"""
sensory_mapper.py
=================
Maps environment concentrations → sensory neuron injection currents
using Hill equations calibrated from C. elegans literature.

Hill equation:  I = I_max * C^n / (Kd^n + C^n)
For derivative-sensing neurons (ASER, BAG): applied to dC/dt.

Literature:
  AWA/diacetyl    Albrecht & Bargmann 2011; Larsch et al. 2015
  ASEL/ASER/NaCl  Suzuki et al. 2008; Kunitomo et al. 2022
  AFD/temperature Kimura et al. 2004; Clark et al. 2006
  URX/O2          Zimmer et al. 2009
  BAG/O2+CO2      Zimmer et al. 2009; Bretscher et al. 2011
  AWB             Troemel et al. 1997
  ASH             Hilliard et al. 2004

Field index reference (must match environment_sim.py FIELD_NAMES):
  0  diacetyl      1  benzaldehyde  2  butanone      3  isoamyl_alc
  4  nonanone      5  octanol       6  noxious       7  nacl
  8  osmolarity    9  ph            10 ascarosides   11 oxygen
  12 co2           13 temperature   14 soil_density
"""

import numpy as np
from collections import deque


# Scale factor: IClamp amplitude must match offset_current native amplitude
# offset_current uses 0.005 nA for PLML/PLMR and drives activity ~0.34.
# Our Hill equations output up to 0.6 nA -- scale down by 120x.
# IClamp scale: steady-state v = leakReversal + I/leakConductance
# = -50 + I/1e-4uS. Threshold at -30mV needs I > 0.002nA.
# Use 0.02nA max (10x threshold) for reliable firing without explosion.
# Original I_max values go up to 0.8nA = 400x threshold = voltage explosion.
ICLAMP_SCALE = 0.12
MECH_SCALE   = 1.0   # separate scale for mechanosensory neurons (CEP/IL1/ADE/ALM/PLM)  # Full amplitude — activity clamped at readout in worm_kinematic_sim.py

def _hill(C, I_max, Kd, n):
    """Hill activation. Returns I_max*ICLAMP_SCALE when C >> Kd, 0 when C << Kd."""
    if C <= 0:
        return 0.0
    Cn  = C ** n
    Kdn = Kd ** n
    return float(I_max * ICLAMP_SCALE * Cn / (Kdn + Cn))


class SensoryMapper:
    """
    Converts concentration + derivative arrays to a dict of
    NEURON injection currents {neuron_name: nA}.

    Parameters
    ----------
    dt : float
        Timestep in seconds (used to scale derivative thresholds).
    """

    def __init__(self, dt=0.001):
        self.dt = dt
        self._urx_active = False   # used for O2/CO2 cross-modulation
        self._nacl_history = deque()  # [(time, conc), ...] for temporal derivative
        self._nacl_t = 0.0            # internal clock
        self._dia_history  = deque()  # diacetyl temporal derivative mean (for AWC off-response)
        self._dia_history_L = deque()  # left nostril bilateral derivative (for AWA ON-response)
        self._dia_history_R = deque()  # right nostril bilateral derivative (for AWA ON-response)
        self._dia_deriv_L  = 0.0
        self._dia_deriv_R  = 0.0
        self._dia_t        = 0.0
        self._dia_deriv    = 0.0
        self._temp_history = deque()  # temperature temporal derivative (for AFD isothermal tracking)
        self._temp_t       = 0.0
        self._temp_deriv   = 0.0

    def tail_chemosensory_currents(self, concs_tail: dict) -> dict:
        """
        Compute phasmid sensory currents from chemical concentrations at tail.
        PHA/PHB respond to repellents and noxious chemicals at the tail.
        PHC responds to gentle touch and some chemicals.
        """
        currents = {}

        def scalar(v):
            return float(v[0]) if isinstance(v, (tuple, list)) else float(v)

        # PHA - responds to repellents (nonanone, octanol) at tail
        I_pha = 0.0
        I_pha = max(I_pha, _hill(scalar(concs_tail.get('nonanone', 0.0)),
                                  0.5, 1e-9, 1.0))
        I_pha = max(I_pha, _hill(scalar(concs_tail.get('octanol', 0.0)),
                                  0.5, 5e-8, 1.0))
        I_pha = max(I_pha, _hill(scalar(concs_tail.get('noxious', 0.0)),
                                  0.4, 1e-4, 2.0))
        if I_pha > 1e-6:
            currents['PHAL'] = float(np.clip(I_pha, 0.0, 1.0))
            currents['PHAR'] = float(np.clip(I_pha, 0.0, 1.0))

        # PHB - responds to repellents and some attractants at tail
        I_phb = 0.0
        I_phb = max(I_phb, _hill(scalar(concs_tail.get('noxious', 0.0)),
                                   0.4, 1e-4, 2.0))
        I_phb = max(I_phb, _hill(scalar(concs_tail.get('octanol', 0.0)),
                                   0.4, 5e-8, 1.0))
        if I_phb > 1e-6:
            currents['PHBL'] = float(np.clip(I_phb, 0.0, 1.0))
            currents['PHBR'] = float(np.clip(I_phb, 0.0, 1.0))

        # PHC - responds to gentle touch + diacetyl at tail (food detection)
        I_phc = _hill(scalar(concs_tail.get('diacetyl', 0.0)),
                       0.3, 11e-9, 1.5)
        if I_phc > 1e-6:
            currents['PHCR'] = float(np.clip(I_phc, 0.0, 1.0))

        return currents

    def mechanosensory_currents(self, density_nose: float, density_tail: float,
                               threshold: float = 0.20) -> dict:
        """
        Compute mechanosensory currents from soil density at nose and tail.
        ALM/AVM fire on anterior (nose) contact, PLM/PVM on posterior (tail).
        Threshold 0.20 = ~50th percentile of density distribution (0.037-0.246).
        Worm encounters mechanosensory stimulation ~50% of the time, consistent
        with burrowing through heterogeneous soil.
        """
        currents = {}
        # Anterior touch: ALM (main), AVM (secondary, lower gain)
        if density_nose > threshold:
            excess = density_nose - threshold
            I_alm = float(np.clip(_hill(excess, 0.6, 0.05, 1.5) * MECH_SCALE, 0.0, 1.0))
            I_avm = float(np.clip(_hill(excess, 0.3, 0.05, 1.5) * MECH_SCALE, 0.0, 1.0))
            if I_alm > 1e-6:
                currents['ALML'] = I_alm
                currents['ALMR'] = I_alm
            if I_avm > 1e-6:
                currents['AVM'] = I_avm
        # Posterior touch: PLM (main), PVM (secondary)
        if density_tail > threshold:
            excess = density_tail - threshold
            I_plm = float(np.clip(_hill(excess, 0.6, 0.05, 1.5) * MECH_SCALE, 0.0, 1.0))
            I_pvm = float(np.clip(_hill(excess, 0.3, 0.05, 1.5) * MECH_SCALE, 0.0, 1.0))
            if I_plm > 1e-6:
                currents['PLML'] = I_plm
                currents['PLMR'] = I_plm
            if I_pvm > 1e-6:
                currents['PVM'] = I_pvm
            # PDE: posterior dopaminergic, same threshold as PLM
            I_pde = float(np.clip(_hill(excess, 0.4, 0.05, 1.5), 0.0, 1.0))
            if I_pde > 1e-6:
                currents['PDEL'] = I_pde
                currents['PDER'] = I_pde

        # Nose touch: CEP (cephalic), IL1 (inner labial), ADE (dopaminergic)
        # Respond to substrate contact (density) AND food presence (diacetyl)
        # Threshold raised to 0.22: global density mean=0.191, p95=0.216.
        # 0.15 fired everywhere (below min density). 0.22 fires only in
        # genuinely dense substrate patches (top ~5% of density distribution).
        cep_threshold = 0.22
        if density_nose > cep_threshold:
            excess = density_nose - cep_threshold
            I_cep = float(np.clip(_hill(excess, 0.6, 0.03, 1.5) * MECH_SCALE, 0.0, 1.0))
            I_il1 = float(np.clip(_hill(excess, 0.4, 0.03, 1.5) * MECH_SCALE, 0.0, 1.0))
            I_ade = float(np.clip(_hill(excess, 0.4, 0.03, 1.5) * MECH_SCALE, 0.0, 1.0))
            if I_cep > 1e-6:
                currents['CEPDL'] = I_cep
                currents['CEPDR'] = I_cep
                currents['CEPVL'] = I_cep
                currents['CEPVR'] = I_cep
            if I_il1 > 1e-6:
                currents['IL1DL'] = I_il1
                currents['IL1DR'] = I_il1
                currents['IL1L']  = I_il1
                currents['IL1R']  = I_il1
                currents['IL1VL'] = I_il1
                currents['IL1VR'] = I_il1
            if I_ade > 1e-6:
                currents['ADEL'] = I_ade
                currents['ADER'] = I_ade
        return currents

    def update_nacl_history(self, nacl_conc: float, window_s: float = 2.0):
        """
        Call once per simulation timestep with current NaCl at worm position.
        Maintains a rolling window and updates the nacl derivative in the
        next map() call via self._nacl_deriv.
        """
        self._nacl_t += self.dt
        self._nacl_history.append((self._nacl_t, nacl_conc))
        # Trim to window using O(1) popleft
        cutoff = self._nacl_t - window_s
        while self._nacl_history and self._nacl_history[0][0] < cutoff:
            self._nacl_history.popleft()
        # Derivative: linear regression slope over window
        if len(self._nacl_history) >= 2:
            t0, c0 = self._nacl_history[0]
            t1, c1 = self._nacl_history[-1]
            dt = t1 - t0
            self._nacl_deriv = (c1 - c0) / dt if dt > 1e-6 else 0.0
        else:
            self._nacl_deriv = 0.0

    def update_diacetyl_history(self, dia_conc: float, window_s: float = 3.0):
        """
        Call once per simulation timestep with mean diacetyl at worm nose.
        Maintains a rolling window for temporal derivative computation.
        AWC fires on diacetyl decrease (leaving food) -- this drives
        AIY strongly (w=10/9) and triggers reorientation.
        """
        self._dia_t += self.dt
        self._dia_history.append((self._dia_t, dia_conc))
        cutoff = self._dia_t - window_s
        while self._dia_history and self._dia_history[0][0] < cutoff:
            self._dia_history.popleft()
        if len(self._dia_history) >= 2:
            t0, c0 = self._dia_history[0]
            t1, c1 = self._dia_history[-1]
            dt = t1 - t0
            self._dia_deriv = (c1 - c0) / dt if dt > 1e-6 else 0.0
        else:
            self._dia_deriv = 0.0

    def update_diacetyl_history_bilateral(self, dia_L: float, dia_R: float,
                                           window_s: float = 3.0):
        """
        Update separate L/R diacetyl derivative histories for bilateral AWA ON-response.
        AWA fires when LOCAL concentration is INCREASING -- derivative detector not
        absolute concentration detector. Bilateral asymmetry in derivative drives
        asymmetric AWAL/AWAR currents -> asymmetric AIZL/AIZR -> steering.
        (authored departure: rate-coded IAF cannot produce sparse ON-response firing;
        we inject derivative-based current directly to approximate this mechanism.)
        """
        for side, c, hist, attr in [
            ('L', dia_L, self._dia_history_L, '_dia_deriv_L'),
            ('R', dia_R, self._dia_history_R, '_dia_deriv_R'),
        ]:
            hist.append((self._dia_t, c))
            cutoff = self._dia_t - window_s
            while hist and hist[0][0] < cutoff:
                hist.popleft()
            if len(hist) >= 2:
                t0, c0 = hist[0]; t1, c1 = hist[-1]
                dt = t1 - t0
                setattr(self, attr, (c1-c0)/dt if dt > 1e-6 else 0.0)
            else:
                setattr(self, attr, 0.0)

    def update_temp_history(self, temp_celsius: float, window_s: float = 2.0):
        """
        Call once per simulation timestep with mean temperature at worm nose.
        AFD implements isothermal tracking -- responds to temporal derivative
        of temperature (moving toward/away from Tc) not just absolute deviation.
        Positive deriv = moving away from Tc = fire to reorient back.
        """
        self._temp_t += self.dt
        self._temp_history.append((self._temp_t, temp_celsius))
        cutoff = self._temp_t - window_s
        while self._temp_history and self._temp_history[0][0] < cutoff:
            self._temp_history.popleft()
        if len(self._temp_history) >= 2:
            t0, c0 = self._temp_history[0]
            t1, c1 = self._temp_history[-1]
            dt = t1 - t0
            self._temp_deriv = (c1 - c0) / dt if dt > 1e-6 else 0.0
        else:
            self._temp_deriv = 0.0

    def map(self, concs: dict) -> dict:
        """
        Compute sensory currents from concentration dictionary.

        Parameters
        ----------
        concs : dict with keys matching FIELD_NAMES from environment_sim.py
            Each value is either a scalar (concentration) or
            (concentration, derivative) tuple.

        Returns
        -------
        currents : dict {neuron_name: float}  currents in nA
        """
        # Unpack - accept both scalar and (conc, deriv) tuple per field
        def get(name, default=0.0):
            v = concs.get(name, default)
            if isinstance(v, (tuple, list)):
                return float(v[0]), float(v[1])
            return float(v), 0.0

        c = {}      # concentrations
        d = {}      # derivatives (dC/dt)
        field_names = [
            'diacetyl','benzaldehyde','butanone','isoamyl_alc',
            'nonanone','octanol','noxious','nacl','osmolarity','ph',
            'ascarosides','oxygen','co2','temperature','soil_density'
        ]
        for name in field_names:
            c[name], d[name] = get(name)

        currents = {}

        # ----------------------------------------------------------------
        # AWA - diacetyl (attractant)
        # Kd 11 nM, n=1.5  (Larsch 2015)
        # ----------------------------------------------------------------
        # AWA - diacetyl with neural adaptation.
        # Real AWA is a pulsatile ON neuron that adapts to steady-state
        # concentration and responds transiently to increases.
        # We implement adaptation via a slow variable that tracks the
        # tonic AWA level; the adapted signal is the transient above baseline.
        # This makes AWA respond to concentration CHANGES not absolute levels,
        # enabling the connectome to compute temporal derivative navigation.
        # Adaptation time constant ~5s (real AWA adapts over several seconds).
        # (authored departure: rate-coded model cannot produce spiking adaptation;
        # this approximates the adaptive dynamics of the real AWA neuron)
        # AWA symmetric baseline for map() standalone use only.
        # Adaptation and bilateral asymmetry handled in map_asymmetric().
        # Do NOT advance _awa_adapt_L/R here -- map_asymmetric() owns those.
        I_L_raw = _hill(c['diacetyl'], I_max=0.6, Kd=2e-3, n=1.5)
        I_R_raw = I_L_raw  # symmetric; map_asymmetric overrides with bilateral values
        if I_L_raw > 1e-6:
            currents['AWAL'] = I_L_raw
            currents['AWAR'] = I_R_raw

        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # AWC - diacetyl OFF neuron driven by AWA adaptation signal.
        # When AWA raw < AWA adapted: concentration is falling relative to
        # recent experience = worm is heading away from food = AWC fires.
        # This drives AWC->AIB->RIM->reversal circuit (connectome weights:
        # AWC->AIBL w=12, AWC->AIBR w=18, AIB->RIMR w=56, AIB->RIML w=47)
        # In the real worm AWC adapts independently; here we use AWA adaptation
        # as a proxy since both neurons detect diacetyl concentration changes.\n        # (authored departure: AWC adaptation not modelled separately; AWA\n        # adaptation used as shared temporal derivative signal)\n        # ----------------------------------------------------------------\n        # Long-range diacetyl gradient loss: backup AWC signal for when worm
        # moves into genuinely empty soil far from all patches.
        # Primary food-departure signal is the pharyngeal proxy in kinematic_sim
        # (bacterial density EMA drop). This diacetyl TC is a secondary signal
        # for long-range gradient loss -- TC=3s gives ~12s off-response.
        # Threshold raised (0.003) so it only fires on large gradient drops,
        # not routine fluctuations during inter-patch navigation.
        awa_raw  = (I_L_raw + I_R_raw) * 0.5
        alpha_slow = self.dt * (1.0/3.0)  # 3s TC -- long-range gradient loss backup
        if not hasattr(self, '_awc_adapt'):
            self._awc_adapt = awa_raw
        self._awc_adapt += alpha_slow * (awa_raw - self._awc_adapt)
        awc_drive = float(np.clip(self._awc_adapt - awa_raw, 0.0, 1.0))
        if awc_drive > 0.0003:  # lowered: detect shallow diacetyl gradients
            I_awc = _hill(awc_drive, I_max=0.5, Kd=0.0001, n=1.5)
            if I_awc > 1e-6:
                currents['AWCL'] = currents.get('AWCL', 0.0) + I_awc
                currents['AWCR'] = currents.get('AWCR', 0.0) + I_awc
        # ----------------------------------------------------------------
        # AWB - repellents: nonanone (Kd 1 nM) and octanol (Kd 50 nM)
        # ----------------------------------------------------------------
        # AWB - repellents: nonanone (Kd 1 nM) and octanol (Kd 50 nM)
        # ----------------------------------------------------------------
        I_non = _hill(c['nonanone'], I_max=0.6, Kd=1e-9,  n=1.0)
        I_oct = _hill(c['octanol'],  I_max=0.5, Kd=5e-8,  n=1.0)
        I = max(I_non, I_oct)
        if I > 1e-6:
            currents['AWBL'] = I
            currents['AWBR'] = I

        # ----------------------------------------------------------------
        # ASH - polymodal nociceptor: noxious chemicals, osmolarity, acid
        # ----------------------------------------------------------------
        I_ash = 0.0
        I_ash = max(I_ash, _hill(c['noxious'],    I_max=0.8, Kd=1e-4,  n=2.0))
        # osmolarity threshold ~200 mOsm = 0.1 in normalised units
        if c['osmolarity'] > 0.1:
            I_ash = max(I_ash, _hill(c['osmolarity'] - 0.1,
                                     I_max=0.6, Kd=0.02, n=1.5))
        # H+ concentration: ASH fires when pH is LOW (acidic).
        # Normalised pH: 0.5=neutral, <0.3=pH<5 (ASH threshold).
        # Invert: signal = 0.5 - ph, so acidic zones produce positive drive.
        _ph_acid = max(0.0, 0.5 - c['ph'])
        if _ph_acid > 0.05:
            I_ash = max(I_ash, _hill(_ph_acid - 0.05, I_max=0.7, Kd=0.08, n=1.5))
        if I_ash > 1e-6:
            currents['ASHL'] = I_ash
            currents['ASHR'] = I_ash

        # ----------------------------------------------------------------
        # ASEL - NaCl increase (positive derivative)
        # ASER - NaCl decrease (negative derivative)  <- derivative-sensing
        # ----------------------------------------------------------------
        # Use history-computed derivative if available (set by update_nacl_history)
        # Falls back to instantaneous derivative from concs dict
        nacl_deriv = getattr(self, '_nacl_deriv', d['nacl'])
        if nacl_deriv > 1e-8:
            I = _hill(nacl_deriv, I_max=0.5, Kd=1e-5, n=1.0)
            if I > 1e-6:
                currents['ASEL'] = I
        if nacl_deriv < -1e-8:
            I = _hill(-nacl_deriv, I_max=0.5, Kd=1e-5, n=1.0)
            if I > 1e-6:
                currents['ASER'] = I

        # ----------------------------------------------------------------
        # PQR - tail O2 sensor (high O2, like URX but posterior)
        # Activates above ~10% O2, drives AVA strongly
        o2_pct_pqr = c['oxygen'] * 21.0
        I_pqr = 0.0
        if o2_pct_pqr > 10.0:
            I_pqr = _hill(o2_pct_pqr - 10.0, I_max=0.6, Kd=2.0, n=2.0)
        elif d['oxygen'] > 1e-5:
            I_pqr = _hill(d['oxygen'], I_max=0.3, Kd=1e-3, n=1.0)
        if I_pqr > 1e-6:
            currents['PQR'] = I_pqr

        # ASJ - ascaroside pheromones (dauer signalling, population density)
        I_asj = _hill(c['ascarosides'], I_max=0.5, Kd=1e-9, n=1.5)
        if I_asj > 1e-6:
            currents['ASJL'] = I_asj
            currents['ASJR'] = I_asj

        # ASK - ascarosides + high osmolarity
        I_ask = _hill(c['ascarosides'], I_max=0.4, Kd=1e-9, n=1.5)
        if c['osmolarity'] > 0.05:
            I_ask = max(I_ask, _hill(c['osmolarity'] - 0.05, I_max=0.4,
                                     Kd=0.05, n=1.5))
        if I_ask > 1e-6:
            currents['ASKL'] = I_ask
            currents['ASKR'] = I_ask

        # AFD - isothermal tracking via temporal temperature derivative
        # AFD fires when worm moves away from Tc (|T-Tc| increasing over time).
        # Uses history buffer derivative, not instantaneous spatial comparison.
        # Bilateral injection (symmetric) -- asymmetry comes from map_asymmetric.
        Tc        = float(concs.get('Tc', 20.0))
        T_celsius = c['temperature'] * 40.0
        delta_T   = T_celsius - Tc
        temp_deriv = getattr(self, '_temp_deriv', 0.0)
        # Fire when moving away from Tc: deriv same sign as delta_T
        moving_away = (delta_T > 0 and temp_deriv > 0.01) or (delta_T < 0 and temp_deriv < -0.01)
        if abs(delta_T) > 0.1 and moving_away:
            I = _hill(abs(temp_deriv), I_max=0.6, Kd=0.1, n=1.5)
            if I > 1e-6:
                currents['AFDL'] = I
                currents['AFDR'] = I

        # ----------------------------------------------------------------
        # URX - oxygen upshift / high O2
        # O2 field normalised 0-1 = 0-21%. Preferred range 5-10%.
        # Activates above ~10% or on upshift. (Zimmer 2009)
        # ----------------------------------------------------------------
        o2_pct = c['oxygen'] * 21.0
        I_urx  = 0.0
        if o2_pct > 10.0:
            I_urx = _hill(o2_pct - 10.0, I_max=0.7, Kd=2.0, n=2.0)
        elif d['oxygen'] > 1e-5:
            I_urx = _hill(d['oxygen'], I_max=0.4, Kd=1e-3, n=1.0)
        self._urx_active = I_urx > 0.1   # gate for CO2 avoidance circuit
        if I_urx > 1e-6:
            currents['URXL'] = I_urx
            currents['URXR'] = I_urx

        # ----------------------------------------------------------------
        # BAG - O2 downshift + CO2
        # BAG is a derivative-sensing neuron for O2 (downshift).
        # CO2 avoidance is gated OFF when URX is active (high O2).
        # ----------------------------------------------------------------
        co2_pct = c['co2'] * 5.0   # normalised 0-1 = 0-5%
        I_bag   = 0.0

        if d['oxygen'] < -1e-5:   # O2 dropping
            I_bag = max(I_bag, _hill(-d['oxygen'], I_max=0.6, Kd=1e-3, n=1.5))
        if o2_pct < 7.0:          # absolute low O2
            I_bag = max(I_bag, _hill(7.0 - o2_pct, I_max=0.5, Kd=2.0, n=1.0))
        # CO2: aversive only when URX is NOT active (O2 < ~19%)
        if co2_pct > 1.0:
            if not self._urx_active or co2_pct > 3.0:
                I_bag = max(I_bag, _hill(co2_pct - 1.0, I_max=0.7, Kd=1.0, n=1.5))
        if I_bag > 1e-6:
            currents['BAGL'] = I_bag
            currents['BAGR'] = I_bag

        # ----------------------------------------------------------------
        # Clamp all to [-5.0, 5.0] nA
        # ----------------------------------------------------------------
        for k in currents:
            currents[k] = float(np.clip(currents[k], -5.0, 5.0))

        return currents


    def map_asymmetric(self, concs_L: dict, concs_R: dict) -> dict:
        """
        Asymmetric version: left neurons sample concs_L, right neurons sample concs_R.
        All non-lateralised neurons use the average of the two positions.
        This implements the head-sweep temporal comparison mechanism.
        """
        import numpy as np

        # Average concentrations for non-lateralised processing
        concs_avg = {}
        for k in concs_L:
            vL = concs_L[k]
            vR = concs_R.get(k, vL)
            if isinstance(vL, tuple):
                concs_avg[k] = ((vL[0]+vR[0])/2, (vL[1]+vR[1])/2)
            else:
                concs_avg[k] = (float(vL) + float(vR)) / 2.0

        # Get symmetric currents as baseline
        currents = self.map(concs_avg)

        # Override lateralised neurons with their specific-side concentration
        def _hill(C, I_max, Kd, n):
            if C <= 0: return 0.0
            Cn = C ** n; Kdn = Kd ** n
            return float(I_max * ICLAMP_SCALE * Cn / (Kdn + Cn))

        def scalar(v):
            return float(v[0]) if isinstance(v, (tuple, list)) else float(v)

        # AWA - diacetyl: bilateral adapted signal.
        # Per-side adaptation (TC=3s) tracks slow mean; signal = transient above baseline.
        # This is the canonical AWA ON-response: fires on concentration increase, adapts out.
        # Authored departure: rate-coded IAF cannot produce spiking adaptation; proxied here.
        # _awa_adapt_L/R owned here; map() no longer advances them.
        cL = scalar(concs_L.get('diacetyl', 0.0))
        cR = scalar(concs_R.get('diacetyl', 0.0))
        IL_raw = _hill(cL, I_max=0.6, Kd=2e-3, n=1.5)  # Kd=2e-3: calibrated to soil-phase diacetyl range (peak ~7e-3, gradient detectable from ~1e-4)
        IR_raw = _hill(cR, I_max=0.6, Kd=2e-3, n=1.5)
        alpha_adapt = self.dt * (1.0 / 3.0)  # TC = 3s
        if not hasattr(self, '_awa_adapt_L'):
            self._awa_adapt_L = IL_raw
            self._awa_adapt_R = IR_raw
        self._awa_adapt_L += alpha_adapt * (IL_raw - self._awa_adapt_L)
        self._awa_adapt_R += alpha_adapt * (IR_raw - self._awa_adapt_R)
        IL = float(np.clip(IL_raw - self._awa_adapt_L, 0.0, 0.6))
        IR = float(np.clip(IR_raw - self._awa_adapt_R, 0.0, 0.6))
        # awa_raw_mean: unadapted bilateral mean used by sensory_drive gate in kinematic_sim
        self.awa_raw_mean = (IL_raw + IR_raw) * 0.5
        if IL > 1e-6:
            currents['AWAL'] = float(np.clip(IL, -1.0, 1.0))
        elif 'AWAL' in currents:
            del currents['AWAL']
        if IR > 1e-6:
            currents['AWAR'] = float(np.clip(IR, -1.0, 1.0))
        elif 'AWAR' in currents:
            del currents['AWAR']

        # ASH - noxious: left and right get different concentrations
        def ash_current(concs):
            I = 0.0
            I = max(I, _hill(scalar(concs.get('noxious', 0.0)), 0.8, 1e-4, 2.0))
            osm = scalar(concs.get('osmolarity', 0.0))
            if osm > 0.1: I = max(I, _hill(osm - 0.1, 0.6, 0.02, 1.5))
            ph = scalar(concs.get('ph', 0.0))
            _ph_acid = max(0.0, 0.5 - ph)
            if _ph_acid > 0.05: I = max(I, _hill(_ph_acid - 0.05, 0.7, 0.08, 1.5))
            return I

        IL_ash = ash_current(concs_L)
        IR_ash = ash_current(concs_R)
        if IL_ash > 1e-6: currents['ASHL'] = float(np.clip(IL_ash, -1.0, 1.0))
        elif 'ASHL' in currents: del currents['ASHL']
        if IR_ash > 1e-6: currents['ASHR'] = float(np.clip(IR_ash, -1.0, 1.0))
        elif 'ASHR' in currents: del currents['ASHR']

        # BAG - O2 absolute level + CO2: spatially asymmetric component
        # O2 derivative is temporal (same at both nostrils), so only
        # absolute O2 level and CO2 concentration are lateralised here.
        def bag_current(concs):
            o2_pct  = scalar(concs.get('oxygen', 0.0)) * 21.0
            co2_pct = scalar(concs.get('co2',    0.0)) * 5.0
            I = 0.0
            if o2_pct < 7.0:
                I = max(I, _hill(7.0 - o2_pct, 0.5, 2.0, 1.0))
            if co2_pct > 1.0:
                I = max(I, _hill(co2_pct - 1.0, 0.7, 1.0, 1.5))
            return I

        IL_bag = bag_current(concs_L)
        IR_bag = bag_current(concs_R)
        if IL_bag > 1e-6: currents['BAGL'] = float(np.clip(IL_bag, -1.0, 1.0))
        elif 'BAGL' in currents: del currents['BAGL']
        if IR_bag > 1e-6: currents['BAGR'] = float(np.clip(IR_bag, -1.0, 1.0))
        elif 'BAGR' in currents: del currents['BAGR']

        # URX - high O2: spatially asymmetric component
        def urx_current(concs):
            o2_pct = scalar(concs.get('oxygen', 0.0)) * 21.0
            if o2_pct > 10.0:
                return _hill(o2_pct - 10.0, 0.7, 2.0, 2.0)
            return 0.0

        IL_urx = urx_current(concs_L)
        IR_urx = urx_current(concs_R)
        if IL_urx > 1e-6: currents['URXL'] = float(np.clip(IL_urx, -1.0, 1.0))
        elif 'URXL' in currents: del currents['URXL']
        if IR_urx > 1e-6: currents['URXR'] = float(np.clip(IR_urx, -1.0, 1.0))
        elif 'URXR' in currents: del currents['URXR']

        # AFD - thermosensation: left and right sample different positions
        # Tc is a scalar (whole-animal memory), use average
        def afd_current(concs):
            T_celsius = scalar(concs.get('temperature', 0.0)) * 40.0
            Tc = float(concs_avg.get('Tc', 20.0))
            delta_T = T_celsius - Tc
            temp_deriv = getattr(self, '_temp_deriv', 0.0)
            moving_away = (delta_T > 0 and temp_deriv > 0.01) or (delta_T < 0 and temp_deriv < -0.01)
            if abs(delta_T) > 0.1 and moving_away:
                return _hill(abs(temp_deriv), 0.6, 0.1, 1.5)
            return 0.0

        IL_afd = afd_current(concs_L)
        IR_afd = afd_current(concs_R)
        if IL_afd > 1e-6: currents['AFDL'] = float(np.clip(IL_afd, -1.0, 1.0))
        elif 'AFDL' in currents: del currents['AFDL']
        if IR_afd > 1e-6: currents['AFDR'] = float(np.clip(IR_afd, -1.0, 1.0))
        elif 'AFDR' in currents: del currents['AFDR']

        return currents


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mapper = SensoryMapper(dt=0.001)

    concs = {
        'diacetyl':     50e-9,    # 50 nM - strong AWA signal
        'benzaldehyde': 0.0,
        'butanone':     0.0,
        'isoamyl_alc':  0.0,
        'nonanone':     0.0,
        'octanol':      0.0,
        'noxious':      0.0,
        'nacl':         (0.005, 1e-4),   # 5 mM, increasing -> ASEL
        'osmolarity':   0.0,
        'ph':           0.0,
        'ascarosides':  0.0,
        'oxygen':       (0.48, 0.0),     # 10%, stable
        'co2':          0.01,
        'temperature':  0.5,             # 20°C
        'soil_density': 0.3,
        'Tc':           20.0,
    }

    currents = mapper.map(concs)
    print("Currents near food source (diacetyl + NaCl increasing):")
    for neuron, current in sorted(currents.items()):
        print(f"  {neuron:8s}: {current:+.4f} nA")

