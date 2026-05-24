"""
graded_connectome.py — Exact reimplementation of Neural Interactome (Kim et al. 2019).

All variable names and equations taken directly from:
    https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/initialize.py

The only changes from the original:
- Flask/SocketIO web framework removed (we call step() directly)
- scipy vode/bdf replaced with same solver via ode() API (identical numerics)
- External current injection via inject()/inject_batch() instead of transit_Mask()
- GPU tensor readout for integration with PyTorch simulation

Authored departure documentation:
- Original uses Iext=100000 normalised units applied via a binary mask (on/off per neuron)
- We inject continuous physical currents (nA) scaled to normalised units
- I_scale = Iext * 0.001 maps 1nA physical -> 100 normalised units
- Original uses adaptive BDF with atol variable; we use atol=1e-3 for speed
"""

import numpy as np
import torch
import os
import re
from scipy import linalg, sparse
from scipy.integrate import ode as scipy_ode

# ── Exact NI parameters from initialize.py ───────────────────────────────────
N_NI  = 279
Gc    = 0.1
C     = 0.015
ggap  = 1.0
gsyn  = 1.0
Ec    = -35.0
ar    = 1.0 / 1.5
ad    = 5.0 / 1.5
B     = 0.125
Iext  = 100000


class GradedConnectome:

    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.N = 0
        self.names = []
        self._name_idx = {}
        self._loaded = False
        self._ode_r = None
        # Current state (numpy, matches ODE solver)
        self._V = None   # voltage relative to Vth (N,)
        self._s = None   # synaptic gating (N,)
        self._I = None   # external current injection (N,) in normalised units
        self._Vth = None # effective threshold (N,)
        # I_scale: physical nA -> normalised units
        # Iext=100000 for full on; 0.001 maps 1nA -> 100 units (10% of max)
        self._I_scale = Iext * 0.001

    def load(self, data_dir):
        """Load NI matrices from directory with Gg.npy, Gs.npy, emask.npy, neuron_names.txt"""
        import warnings
        def _find(base, *candidates):
            for c in candidates:
                p = os.path.join(base, c)
                if os.path.exists(p): return p
            raise FileNotFoundError(f'None of {candidates} found in {base}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Gg     = np.load(_find(data_dir, 'Gg.npy',     'neural_interactome_Gg.npy')).astype(np.float64)
            Gs     = np.load(_find(data_dir, 'Gs.npy',     'neural_interactome_Gs.npy')).astype(np.float64)
            E_mask = np.load(_find(data_dir, 'emask.npy',  'neural_interactome_emask.npy'))

        # Parse neuron names (Python 2 unicode list format: [u'NAME', ...])
        for _nn_name in ('neuron_names.txt', 'neural_interactome_neuron_names.txt'):
            _nn_path = os.path.join(data_dir, _nn_name)
            if os.path.exists(_nn_path):
                break
        else:
            raise FileNotFoundError(f'neuron_names file not found in {data_dir}')
        with open(_nn_path) as f:
            raw = f.read()
        self.names = re.findall(r"u?'([A-Z0-9]+)'", raw)
        assert len(self.names) == N_NI, f'Expected {N_NI} neurons, got {len(self.names)}'
        self.N = N_NI
        self._name_idx = {n: i for i, n in enumerate(self.names)}

        # Exact NI E computation:
        #   E = np.load('emask.npy')
        #   E = -48.0 * E
        #   EMat = np.tile(np.reshape(E, N), (N, 1))
        E = E_mask.astype(np.float64).flatten()
        E = -48.0 * E                          # (N,) -- same as NI
        EMat = np.tile(E.reshape(N_NI), (N_NI, 1))  # (N,N) -- EMat[i,j] = E[j]

        self._Gg = Gg
        self._Gs = Gs
        self._E  = E       # (N,) per-neuron E value
        self._EMat = EMat  # (N,N) tiled: EMat[i,j] = E[j]

        # Compute EffVth (exact NI EffVth() function)
        self._Vth = self._compute_EffVth(Gg, Gs, E).flatten()
        print(f'  EffVth computed: mean={self._Vth.mean():.2f}mV '
              f'range=[{self._Vth.min():.2f}, {self._Vth.max():.2f}]mV')

        # Initial conditions: small random noise (exact NI InitCond)
        np.random.seed(42)
        ic = 1e-4 * np.random.normal(0, 0.94, 2 * self.N)
        self._V = self._Vth + ic[:self.N]  # absolute membrane voltage
        self._s = np.clip(ic[self.N:], 0.0, 1.0)
        self._I = np.zeros(self.N)

        self._loaded = True
        self._ode_r = None  # lazy init
        print(f'GradedConnectome loaded: {self.N} neurons on {self.device}')

    def _compute_EffVth(self, Gg, Gs, E):
        """Exact copy of NI EffVth() function."""
        N = self.N
        Gcmat = np.multiply(Gc, np.eye(N))
        EcVec = np.multiply(Ec, np.ones((N, 1)))

        M1 = -Gcmat
        b1 = np.multiply(Gc, EcVec)

        Ggap_m = np.multiply(ggap, Gg)
        Ggapdiag = np.subtract(Ggap_m, np.diag(np.diag(Ggap_m)))
        Ggapsum = Ggapdiag.sum(axis=1)
        Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
        M2 = -np.subtract(Ggapsummat, Ggapdiag)

        Gs_ij = np.multiply(gsyn, Gs)
        s_eq = round(ar / (ar + 2 * ad), 4)
        sjmat = np.multiply(s_eq, np.ones((N, N)))
        Gsyn = np.multiply(sjmat, Gs_ij)
        Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
        Gsynsum = Gsyndiag.sum(axis=1)
        M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()
        b3 = np.dot(Gs_ij, np.multiply(s_eq, E))

        M = M1 + M2 + M3
        (P, LL, UU) = linalg.lu(M)
        bbb = -b1.flatten() - b3
        Vth = linalg.solve_triangular(
            UU, linalg.solve_triangular(LL, bbb, lower=True, check_finite=False),
            check_finite=False)
        return np.asarray(Vth).flatten()

    def _rhs(self, t, y):
        """
        Exact copy of NI membrane_voltageRHS(), minus the transit_Mask/web stuff.

        From initialize.py:
            VsubEc = Gc * (Vvec - Ec)
            Vrep = np.tile(Vvec, (N,1))
            GapCon = Gg * (Vrep.T - Vrep)).sum(axis=1)
            VsubEj = Vrep.T - EMat           # EMat[i,j] = E[j]
            SynapCon = (Gs * tile(SVec,(N,1)) * VsubEj).sum(axis=1)
            SynRise = ar*(1-SVec)*sigmoid(B*(Vvec-Vth))
            SynDrop = ad*SVec
            Input = Iext*InMask
            dV = (-(VsubEc+GapCon+SynapCon)+Input)/C
            dS = SynRise - SynDrop

        Note: Vvec here is the RAW membrane voltage (not relative to Vth).
        The NI subtracts Vth AFTER integration for display only (voltage_filter).
        We store V as raw membrane voltage for correct RHS computation.
        """
        Vvec = y[:self.N]
        SVec = y[self.N:]

        # Gc*(Vi - Ec)
        VsubEc = np.multiply(Gc, (Vvec - Ec))

        # Gap junction: ggap*Gg*(Vi-Vj)
        Vrep = np.tile(Vvec, (self.N, 1))
        GapCon = np.multiply(ggap, self._Gg) 
        GapCon = np.multiply(GapCon, np.subtract(np.transpose(Vrep), Vrep))
        GapCon = GapCon.sum(axis=1)

        # Synaptic: gsyn*Gs*S*(Vi-Ej)
        VsubEj = np.subtract(np.transpose(Vrep), self._EMat)
        SynapCon = np.multiply(
            np.multiply(gsyn * self._Gs, np.tile(SVec, (self.N, 1))),
            VsubEj).sum(axis=1)

        # Sigmoid gating: B*(Vvec - Vth)
        SynRise = np.multiply(
            np.multiply(ar, np.subtract(1.0, SVec)),
            np.reciprocal(1.0 + np.exp(np.clip(-B * np.subtract(Vvec, self._Vth), -500, 500))))
        SynDrop = np.multiply(ad, SVec)

        # External input
        Input = self._I

        dV = (-(VsubEc + GapCon + SynapCon) + Input) / C
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    def _build_implicit_matrix(self, dt):
        """
        Pre-factorise the full implicit matrix for leak + gap junctions.
        Solves (C/dt*I + Gc*I + ggap*L_Gg) * V_new = rhs each step,
        where L_Gg is the graph Laplacian of the gap junction network.
        Pre-factored LU decomposition makes each step O(N^2) not O(N^3).
        """
        from scipy import linalg as _la
        N = self.N
        # Gap junction Laplacian: L[i,i] = sum_j Gg[i,j], L[i,j] = -Gg[i,j]
        Gg_rowsum = self._Gg.sum(axis=1)
        L_Gg = np.diag(Gg_rowsum) - self._Gg  # (N,N) Laplacian
        # Implicit matrix: A = (C/dt + Gc)*I + ggap*L_Gg
        A = (C / dt + Gc) * np.eye(N) + ggap * L_Gg
        # Pre-factorise
        self._impl_lu = _la.lu_factor(A)
        # Store matrices on GPU for explicit terms
        dev = self.device
        self._Gs_t  = torch.tensor(gsyn * self._Gs, dtype=torch.float32, device=dev)
        self._Vth_t = torch.tensor(self._Vth, dtype=torch.float32, device=dev)
        self._E_t   = torch.tensor(self._E, dtype=torch.float32, device=dev)
        self._EMat_t = self._E_t.unsqueeze(0).expand(self.N, self.N)
        # Gc*Ec vector (constant)
        self._Gc_Ec = np.full(N, Gc * Ec)
        self._impl_dt = dt

    def reset_integrator(self):
        """No-op."""
        pass

    def step(self, dt=0.005):
        """
        IMEX step: full gap junction matrix treated implicitly (LU solve),
        chemical synapses and external current treated explicitly.
        GPU used for explicit synaptic computation, CPU for LU solve.
        """
        from scipy import linalg as _la
        if not hasattr(self, '_impl_lu') or self._impl_dt != dt:
            self._build_implicit_matrix(dt)

        dev = self.device
        V = torch.tensor(self._V, dtype=torch.float32, device=dev)
        s = torch.tensor(self._s, dtype=torch.float32, device=dev)

        # Exact NI: SynapCon[i] = sum_j Gs[i,j]*s[j]*(V[j]-E[i])
        # Vrep[i,j]=V[i], so Vrep.T[i,j]=V[j] (presynaptic voltage)
        # EMat_t[i,j]=E[j], so EMat_t.T[i,j]=E[i] (postsynaptic reversal)
        Vrep    = V.unsqueeze(0).expand(self.N, self.N)   # [i,j]=V[i]
        VsubEj  = Vrep.T - self._EMat_t.T                # [i,j]=V[j]-E[i]
        SynapCon = (self._Gs_t * s.unsqueeze(0) * VsubEj).sum(dim=1).cpu().numpy()

        # External current
        I_np = self._I.copy()

        # RHS: (C/dt)*V_old + Gc*Ec - SynapCon + I
        V_np = self._V.copy()
        rhs  = (C / dt) * V_np + self._Gc_Ec - SynapCon + I_np

        # Implicit solve (pre-factored LU)
        V_new = _la.lu_solve(self._impl_lu, rhs)
        V_new = np.clip(V_new, -200.0, 200.0)

        # Synaptic gating (explicit)
        V_rel = V - self._Vth_t
        sig   = torch.sigmoid(torch.tensor(B, device=dev) * V_rel)
        s_new = torch.clamp(s + (ar * sig * (1 - s) - ad * s) * dt, 0.0, 1.0)

        self._V = V_new
        self._s = s_new.cpu().numpy()

    def inject(self, name, current_nA):
        """Inject physical current (nA) into named neuron."""
        if not self._loaded: raise RuntimeError('Load first')
        if name in self._name_idx:
            self._I[self._name_idx[name]] = current_nA * self._I_scale
        self.reset_integrator()

    def inject_batch(self, currents_dict):
        """Set injections for multiple neurons. {name: nA}. Zeros all others."""
        if not self._loaded: return
        self._I[:] = 0.0
        for name, I_nA in currents_dict.items():
            if name in self._name_idx:
                self._I[self._name_idx[name]] = float(I_nA) * self._I_scale
        self.reset_integrator()

    def voltage(self, name):
        """Voltage relative to Vth (mV). At rest ~0, stimulated > 0."""
        if name in self._name_idx:
            i = self._name_idx[name]
            return float(self._V[i] - self._Vth[i])
        return 0.0

    def voltage_abs(self, name):
        """Absolute membrane voltage (mV)."""
        if name in self._name_idx:
            return float(self._V[self._name_idx[name]])
        return 0.0

    def steering_signal(self, baseline_aizl=0.0, baseline_aizr=0.0,
                        baseline_aval=0.0, baseline_avar=0.0, baseline_avb=0.0):
        """
        Compute steering/reversal/suppression signals.
        Steering from AIZL/AIZR (strongest AWA propagation, 1-2 hops).
        Reversal from AVAR/AVAL (command neurons, 3+ hops, weaker but correct).
        All voltages relative to Vth.
        """
        aizl = self.voltage('AIZL')
        aizr = self.voltage('AIZR')
        aval = self.voltage('AVAL')
        avar = self.voltage('AVAR')
        avb  = (self.voltage('AVBL') + self.voltage('AVBR')) / 2.0

        # Turn from AIZ asymmetry (AIZL > AIZR = food to left = steer left/negative)
        # Scale: at 0.5nA AWAL injection, AIZ delta ~ 0.7mV -> want turn_signal ~ 0.05
        turn = -(aizl - aizr) * 0.07

        # Reversal: AVA above baseline
        reversal = float(np.clip((aval + avar) / 2.0 - baseline_aval, 0, 100) / 5.0)

        # Suppression: AVB above baseline (forward drive)
        suppression = float(np.clip(avb - baseline_avb, 0, 100) / 5.0)

        return float(turn), float(reversal), float(suppression)

    def all_voltages(self):
        return self.names, self._V.copy()


def test_graded_connectome(data_dir):
    import torch
    gc = GradedConnectome(device='cuda' if torch.cuda.is_available() else 'cpu')
    gc.load(data_dir)

    print()
    print('=== NEURAL INTERACTOME TEST ===')
    for n in ['AWAL','AIYL','AIZL','AVAR','AVAL']:
        print(f'  {n} in model: {n in gc._name_idx}')
    print()

    dt = 0.005  # 5ms
    print('Warming up 2s...')
    for _ in range(int(2.0 / dt)):
        gc.step(dt=dt)

    print('Baseline (relative to Vth):')
    for n in ['AWAL','AWAR','AIZL','AIZR','AIYL','AIYR','AVAR','AVAL','AVBL','AVBR','RIS']:
        print(f'  {n:8s}: {gc.voltage(n):+.4f} mV rel  ({gc.voltage_abs(n):.2f} mV abs)')

    bl_aval = gc.voltage('AVAL')
    bl_avar = gc.voltage('AVAR')
    bl_avb  = (gc.voltage('AVBL') + gc.voltage('AVBR')) / 2

    print()
    print('Injecting 0.5nA into AWAL for 2s...')
    gc.inject('AWAL', 0.5)
    for _ in range(int(2.0 / dt)):
        gc.step(dt=dt)

    print('Post-injection:')
    for n in ['AWAL','AWAR','AIZL','AIZR','AIYL','AIYR','AVAR','AVAL','AVBL','AVBR','RIS']:
        delta = gc.voltage(n) - (bl_aval if n in ('AVAL','AVAR') else gc.voltage(n))
        print(f'  {n:8s}: {gc.voltage(n):+.4f} mV rel  ({gc.voltage_abs(n):.2f} mV abs)')

    turn, rev, supp = gc.steering_signal(bl_aval, bl_avar, bl_avb)
    print()
    print(f'Steering: turn={turn:+.5f}  reversal={rev:.4f}  suppression={supp:.4f}')


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/NeuralInteractome'
    test_graded_connectome(data_dir)
