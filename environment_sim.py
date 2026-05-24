"""
environment_sim.py
==================
15-field soil environment for C. elegans simulation.
Fully GPU-accelerated: all field arrays live on GPU permanently.
CPU transfers only for: sample() bilinear interp, bacterial reseeding (every 5s),
HDF5 saving (every save_every_n steps).

Physical scale: 1 wu = 50 μm. World: 960×540 wu. Grid: 320×180 (isotropic 3 wu).

Fields:
  0  diacetyl    1  benzaldehyde  2  butanone     3  isoamyl_alc
  4  nonanone    5  octanol       6  noxious       7  nacl
  8  osmolarity  9  ph           10  ascarosides  11  oxygen
  12 co2        13  temperature  14  soil_density
"""

import os, time
import numpy as np
import torch
import torch.nn.functional as F
import h5py

WU_TO_M = 5e-5  # 1 world unit = 50 μm

FIELD_NAMES = [
    'diacetyl', 'benzaldehyde', 'butanone', 'isoamyl_alc',
    'nonanone', 'octanol', 'noxious', 'nacl',
    'osmolarity', 'ph', 'ascarosides', 'oxygen',
    'co2', 'temperature', 'soil_density',
]
N_FIELDS = len(FIELD_NAMES)
FIELD_IDX = {name: i for i, name in enumerate(FIELD_NAMES)}

# Physically derived D values [wu²/s] — all stable at dt=0.001, dx=6
D_EFF = np.array([
    5.00,   5.00,   5.00,   5.00,   # volatiles: stable gradient, reaches 880wu in ~130s warmup
    3.00,   3.00,                   # larger volatiles
      0.20,   0.50,   0.20,   0.30,  # aqueous -- pH reduced 2.0->0.3: slower spread
      0.06,  50.00,   5.00,          # ascarosides, O2 (fast soil diffusion), CO2
     40.00,   0.00,                   # temperature, soil_density
], dtype=np.float32)

K_DECAY = np.array([
    0.0,  4e-4, 5e-4, 4e-4,   # diacetyl=0: volatilisation handled explicitly in step() source term
    3e-4, 3e-4,
    2e-4, 0.0, 0.0, 0.0, 8e-5,  # noxious decay reduced: 8e-3->2e-4 to allow accumulation
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)


class EnvConfig:
    def __init__(
        self, nx=480, nz=270, world_x=960.0, world_z=540.0,
        dt_env=0.001, save_every_n=2000, use_gpu=True,
        # Multi-patch colony parameters
        n_patches=None,          # None = random 4-6
        patch_radius_wu=None,    # None = random 40-70 wu per patch
        min_separation_wu=150.0, # minimum centre-to-centre distance (wu)
        # Placement zone: patches drawn from this world-space rectangle
        # Default: left-centre of world, leaving right third as open soil
        patch_zone_x=(150.0, 700.0),
        patch_zone_z=(80.0,  460.0),
        nacl_seed=None, colony_seed=None,
        # Legacy single-colony args kept for backwards compat (ignored)
        colony_ix_min=None, colony_ix_max=None,
        colony_iz_min=None, colony_iz_max=None,
    ):
        self.nx, self.nz = nx, nz
        self.world_x, self.world_z = world_x, world_z
        self.world_y = world_z * 0.5
        self.dx = world_x / nx
        self.dz = world_z / nz
        self.dt_env  = dt_env
        self.save_every_n = save_every_n
        self.use_gpu = use_gpu
        self.n_patches         = n_patches
        self.patch_radius_wu   = patch_radius_wu
        self.min_separation_wu = min_separation_wu
        self.patch_zone_x      = patch_zone_x
        self.patch_zone_z      = patch_zone_z
        self.nacl_seed         = nacl_seed
        self.colony_seed       = colony_seed


class EnvironmentSimulator:
    """
    Fully GPU-resident environment. C and B live on GPU permanently.
    sample() does a fast GPU bilinear interp, returns CPU scalars.
    CPU transfers only happen for reseeding (every 5s) and HDF5 saving.
    """

    def __init__(self, output_dir='simulations/current', config=None):
        if config is None:
            config = EnvConfig()
        self.cfg = config
        self.nx, self.nz = config.nx, config.nz
        self.lx, self.lz = config.world_x, config.world_z
        self.ly = config.world_y
        self.dt = config.dt_env
        self.dx = self.lx / self.nx
        self.dz = self.lz / self.nz
        assert abs(self.dx - self.dz) < 0.01, f"Non-isotropic: dx={self.dx} dz={self.dz}"
        self._inv_dx2 = 1.0 / (self.dx ** 2)

        self.output_dir = output_dir
        self.step_count = 0

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[EnvSim] GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[EnvSim] CPU mode")

        # ── persistent GPU tensors ────────────────────────────────────────
        # C: (N_FIELDS, nx, nz) — all chemical fields, lives on GPU
        # B: (nx, nz)           — bacterial density, lives on GPU
        # C_prev: (N_FIELDS, nx, nz) — previous step for derivative
        self.C      = None  # GPU tensor, allocated in reset()
        self.C_prev = None  # GPU tensor
        self.B      = None  # GPU tensor

        # CPU mirror of C_prev for sample() — updated each step
        self._C_prev_cpu = None

        self._seed_timer    = 0.0
        self._seed_interval = 5.0

        # Broadcast-shaped constants on GPU
        self._D_gpu = torch.from_numpy(D_EFF.reshape(N_FIELDS,1,1)).to(self.device)
        self._K_gpu = torch.from_numpy(K_DECAY.reshape(N_FIELDS,1,1)).to(self.device)

        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0,1,0],[1,-4,1],[0,1,0]],
            dtype=torch.float32, device=self.device
        ).view(1,1,3,3)

        # Scalar field indices as tensors for indexing
        self._idx_O2  = FIELD_IDX['oxygen']
        self._idx_dia = FIELD_IDX['diacetyl']
        self._idx_but = FIELD_IDX['butanone']
        self._idx_benz= FIELD_IDX['benzaldehyde']
        self._idx_nox = FIELD_IDX['noxious']
        self._idx_asc = FIELD_IDX['ascarosides']
        self._idx_osm = FIELD_IDX['osmolarity']
        self._idx_ph  = FIELD_IDX['ph']
        self._idx_co2 = FIELD_IDX['co2']
        self._idx_temp= FIELD_IDX['temperature']

        # AFD temperature memory
        self.Tc       = 20.0
        self.alpha_tc = 3e-4

        # HDF5 handles
        self.h5 = None
        self._h5_path = None
        self._h5_chem = self._h5_bact = self._h5_times = None
        self._h5_sensory = self._h5_sensory_t = None
        self._h5_nose    = None
        self._h5_muscle  = self._h5_muscle_v = self._h5_muscle_t = None

    # ── setup ─────────────────────────────────────────────────────────────

    def reset(self):
        self.step_count  = 0
        self._seed_timer = 0.0

        # Allocate GPU tensors
        self.C      = torch.zeros((N_FIELDS, self.nx, self.nz),
                                   dtype=torch.float32, device=self.device)
        self.C_prev = torch.zeros_like(self.C)
        self.B      = torch.zeros((self.nx, self.nz),
                                   dtype=torch.float32, device=self.device)

        self._init_environment()
        self.C_prev.copy_(self.C)
        self._C_prev_cpu = self.C_prev.cpu().numpy()

        self._open_hdf5()
        print(f"[EnvSim] World {self.lx:.0f}×{self.lz:.0f} wu "
              f"({self.lx*WU_TO_M*1000:.1f}×{self.lz*WU_TO_M*1000:.1f} mm), "
              f"grid {self.nx}×{self.nz}, cell {self.dx:.1f} wu, device={self.device}")

    def _init_environment(self):
        """Initialise all fields on GPU."""
        dev = self.device

        # Oxygen: uniform aerobic
        self.C[self._idx_O2] = 0.21

        # CO2: low background
        self.C[self._idx_co2] = 0.005

        # Temperature: smooth patches (generate on CPU, upload)
        trng = torch.Generator()
        if self.cfg.nacl_seed is not None:
            trng.manual_seed(self.cfg.nacl_seed + 1)
        t_raw = torch.rand(1,1,self.nx,self.nz, generator=trng)
        t_sm  = F.avg_pool2d(t_raw, 21, stride=1, padding=10)
        self.C[self._idx_temp] = (0.42 + t_sm.squeeze()*0.18).to(dev)

        # Soil density: multi-scale heterogeneous texture.
        # Real soil has structure at multiple scales:
        #   - Large clods / compacted aggregates (~60wu): coarse layer
        #   - Intermediate pore channels (~15wu): mid layer
        #   - Fine particle variation (~9wu): fine layer
        # Combined noise normalised to mean=0.30, std=0.15:
        #   ~21% of world above 0.4 (dense clods), ~8% below 0.15 (pore spaces)
        # Seeded from colony_seed for reproducibility across runs.
        # Authored departure: multi-scale Gaussian noise approximates soil
        # aggregate structure; real soil texture is fractal (Perlin noise
        # would be more accurate but computationally equivalent for our purposes).
        _soil_seed = self.cfg.colony_seed if self.cfg.colony_seed is not None else 42
        srng_c = torch.Generator(); srng_c.manual_seed(_soil_seed)
        srng_m = torch.Generator(); srng_m.manual_seed(_soil_seed + 1000)
        srng_f = torch.Generator(); srng_f.manual_seed(_soil_seed + 2000)
        s_coarse = F.avg_pool2d(torch.rand(1,1,self.nx,self.nz,generator=srng_c), 21, stride=1, padding=10)
        s_mid    = F.avg_pool2d(torch.rand(1,1,self.nx,self.nz,generator=srng_m),  5, stride=1, padding=2)
        s_fine   = F.avg_pool2d(torch.rand(1,1,self.nx,self.nz,generator=srng_f),  3, stride=1, padding=1)
        s_combined = 0.5 * s_coarse + 0.35 * s_mid + 0.15 * s_fine
        s_sq = s_combined.squeeze()
        s_norm = (s_sq - s_sq.mean()) / (s_sq.std() + 1e-8)  # zero mean, unit std
        soil = (0.30 + s_norm * 0.15).clamp(0.05, 0.80)      # mean=0.30, std~0.12, range 0.05-0.80
        self.C[FIELD_IDX['soil_density']] = soil.to(dev)

        # NaCl: heterogeneous static landscape
        nrng = torch.Generator()
        if self.cfg.nacl_seed is not None:
            nrng.manual_seed(self.cfg.nacl_seed)
        n_raw = torch.rand(1,1,self.nx,self.nz, generator=nrng)
        n_sm  = F.avg_pool2d(n_raw, 19, stride=1, padding=9)
        self.C[FIELD_IDX['nacl']] = (n_sm.squeeze()*0.008).to(dev)

        # Osmolarity: static heterogeneous landscape correlated with soil density.
        # High-osmolarity pockets represent evaporated salt in dense soil regions.
        # C. elegans avoids osmolarity >200 mOsm (threshold=0.1 normalised).
        # Range: 0.0-0.45, mean ~0.08 (most soil below ASH threshold).
        # Seeded independently from NaCl to give uncorrelated repellent landscape.
        # Authored departure: real soil osmolarity is driven by ion concentration
        # and water activity; we use multi-scale noise correlated with soil density.
        # Osmolarity: derived from soil density field (already well-distributed).
        # High-density soil regions retain more dissolved ions -> higher osmolarity.
        # Scaled so ~15% of world exceeds ASH threshold (0.1).
        # Biological basis: soil osmolarity correlates with particle packing and
        # water retention; dense clay regions have higher ion concentration.
        # Authored departure: real soil osmolarity depends on mineralogy and
        # moisture; we use soil density as a tractable proxy.
        osm = (soil * 0.22).clamp(0.0, 0.45)   # soil 0.3->osm 0.066, soil 0.6->osm 0.132  ~15% above ASH threshold
        self.C[self._idx_osm] = osm.to(dev)

        # pH: initialise as neutral (0.5 = pH 7.0 in normalised units).
        # Dynamic: bacterial decomposition produces CO2 -> carbonic acid -> acidification.
        # ASH responds when pH < 5 (normalised < ~0.3).
        # Will develop acidic zones near patch edges where bacteria are dying.
        self.C[self._idx_ph] = torch.full((self.nx, self.nz), 0.5,
                                           dtype=torch.float32, device=dev)

        # Bacterial colonies — multiple circular patches scattered in world.
        # Each patch is a gaussian blob with organic sub-structure.
        # Patches placed by rejection sampling with minimum separation.
        # (authored departure: real soil microbiomes are spatially heterogeneous
        # micro-colony clusters; we use circular gaussians as a tractable proxy)
        B_cpu = np.zeros((self.nx, self.nz), dtype=np.float32)
        rng = np.random.RandomState(
            self.cfg.colony_seed if self.cfg.colony_seed is not None else 0)

        # Pixel coordinate grids
        xi = np.arange(self.nx, dtype=np.float32)
        zi = np.arange(self.nz, dtype=np.float32)
        XX, ZZ = np.meshgrid(xi, zi, indexing='ij')

        # Determine number of patches
        n_patches = self.cfg.n_patches
        if n_patches is None:
            n_patches = rng.randint(4, 7)  # 4-6 inclusive

        # Placement zone in grid coordinates
        x_lo_wu, x_hi_wu = self.cfg.patch_zone_x
        z_lo_wu, z_hi_wu = self.cfg.patch_zone_z
        x_lo = x_lo_wu / self.dx
        x_hi = x_hi_wu / self.dx
        z_lo = z_lo_wu / self.dz
        z_hi = z_hi_wu / self.dz
        min_sep_grid = self.cfg.min_separation_wu / self.dx  # isotropic grid

        # Store patch metadata for reseeding
        self._patches = []  # list of (cx_grid, cz_grid, r_grid)

        # Rejection sampling for patch centres
        max_attempts = 500
        for _ in range(n_patches):
            r_wu = (self.cfg.patch_radius_wu
                    if self.cfg.patch_radius_wu is not None
                    else rng.uniform(40.0, 70.0))
            r_grid = r_wu / self.dx

            placed = False
            for _attempt in range(max_attempts):
                cx = rng.uniform(x_lo + r_grid, x_hi - r_grid)
                cz = rng.uniform(z_lo + r_grid, z_hi - r_grid)
                # Check minimum separation from all existing patches
                too_close = any(
                    np.sqrt((cx - px)**2 + (cz - pz)**2) < min_sep_grid
                    for px, pz, _ in self._patches
                )
                if not too_close:
                    self._patches.append((cx, cz, r_grid))
                    placed = True
                    break
            if not placed:
                # Relax: place anyway (world may be crowded with many patches)
                cx = rng.uniform(x_lo + r_grid, x_hi - r_grid)
                cz = rng.uniform(z_lo + r_grid, z_hi - r_grid)
                self._patches.append((cx, cz, r_grid))

        # Generate bacterial density for each patch
        for cx, cz, r_grid in self._patches:
            # Main gaussian — sigma = 0.4 * radius gives ~80% density at centre,
            # falling smoothly to near-zero at patch edge
            sigma = r_grid * 0.4
            base  = np.exp(-0.5 * (((XX - cx)/sigma)**2 +
                                    ((ZZ - cz)/sigma)**2))
            B_cpu = np.maximum(B_cpu, 0.85 * base)

            # 6-10 sub-blobs per patch for organic texture
            n_sub = rng.randint(6, 11)
            for _ in range(n_sub):
                # Sub-blob centres within patch radius
                angle  = rng.uniform(0, 2*np.pi)
                dist_f = rng.beta(2, 2)  # weighted toward centre
                bx = cx + dist_f * r_grid * 0.8 * np.cos(angle)
                bz = cz + dist_f * r_grid * 0.8 * np.sin(angle)
                sx = rng.uniform(r_grid*0.08, r_grid*0.22)
                sz = rng.uniform(r_grid*0.08, r_grid*0.22)
                amp = rng.uniform(0.15, 0.45)
                blob = amp * np.exp(-0.5 * (((XX - bx)/sx)**2 +
                                             ((ZZ - bz)/sz)**2))
                B_cpu = np.maximum(B_cpu, blob)

            # Hard circular mask with 3-grid-cell soft edge
            dist_from_centre = np.sqrt((XX - cx)**2 + (ZZ - cz)**2)
            taper = np.clip((r_grid - dist_from_centre) / 3.0, 0.0, 1.0)
            # Apply mask only to this patch's contribution region
            # (use a local annular mask rather than a global multiply
            # so overlapping patch edges don't suppress each other)
            outside = dist_from_centre > r_grid + 3.0
            B_cpu[outside] = np.minimum(B_cpu[outside],
                                         B_cpu[outside] * taper[outside])

        B_cpu = np.clip(B_cpu, 0.0, 1.0)
        self.B.copy_(torch.from_numpy(B_cpu))
        print(f"[EnvSim] {len(self._patches)} bacterial patches: "
              + ", ".join(f"({p[0]*self.dx:.0f},{p[1]*self.dz:.0f})wu r={p[2]*self.dx:.0f}wu"
                          for p in self._patches))

    # ── step (fully GPU) ───────────────────────────────────────────────────

    def step(self, dt=None):
        if self.C is None:
            raise RuntimeError("Call reset() before step()")
        if dt is None:
            dt = self.dt
        dt_t = float(dt)

        # Save previous for derivative (GPU copy, cheap)
        self.C_prev.copy_(self.C)

        # ── bacterial dynamics (all GPU) ──────────────────────────────────
        O2  = self.C[self._idx_O2].clamp(0.0, 1.0)   # (nx,nz)
        B   = self.B                                    # (nx,nz)

        K_O2    = 0.05
        O2_lim  = O2 / (O2 + K_O2)                    # Monod term
        dB_grow = 0.08 * B * (1.0 - B) * O2_lim

        O2_crit    = 0.02
        death_mult = 1.0 + 8.0 * (O2_crit - O2).clamp(min=0.0) / O2_crit
        dB_die     = 0.03 * death_mult * B

        self.B = (B + dt_t*(dB_grow - dB_die)).clamp(0.0, 1.0)

        # ── chemical sources (all GPU) ────────────────────────────────────
        metabolic   = self.B * O2_lim
        dead_matter = dB_die * B

        # Diacetyl
        # Production: 2e-4 * B * O2_lim (bacterial metabolic output)
        # Volatilisation: 5e-4 * C (first-order loss, ~23min half-life in moist soil)
        # Bacterial consumption: 1e-2 * B * C (E. coli acetoin reductase pathway)
        #   Creates local sink at colony — steepens near-field gradient without
        #   changing far-field diffusion length (set by volatilisation alone).
        #   K_consume calibrated so production ≈ consumption at steady state:
        #   2e-4 * B = 1e-2 * B * C_steady → C_steady ≈ 2e-2 near dense colony.
        # (authored departure: consumption rate estimated from E. coli diacetyl
        #  catabolism literature; exact value soil-context adjusted)
        dia = self.C[self._idx_dia]
        self.C[self._idx_dia] = (dia + dt_t*(
            2e-4*metabolic           # bacterial production
            - 5e-4*dia               # volatilisation: K=5e-4 -> L=100wu far-field (authored: soil moisture retention slows volatilisation)
            - 1e-2*self.B*dia        # bacterial consumption — local sink at source
        )).clamp(min=0.0)

        # Butanone
        but = self.C[self._idx_but]
        self.C[self._idx_but] = (but + dt_t*(2e-4*metabolic - 5e-5*but)).clamp(min=0.0)

        # Benzaldehyde
        benz = self.C[self._idx_benz]
        self.C[self._idx_benz] = (benz + dt_t*(1e-8*metabolic - 30.0*benz*self.B)).clamp(min=0.0)

        # Noxious
        nox = self.C[self._idx_nox]
        self.C[self._idx_nox] = (nox + dt_t*(8e-2*dead_matter - 12.0*nox*metabolic)).clamp(min=0.0)

        # pH: acidification from bacterial decomposition (dead matter -> CO2 -> H+)
        # At steady state, patch edges (high dead_matter, low metabolic) become acidic.
        # Neutral soil pH ~7 (0.5 normalised). Each unit of dead_matter drops pH ~1 unit.
        # Recovery toward neutral: slow relaxation (soil buffering capacity).
        # ASH threshold: pH < 5 = normalised < 0.3.
        # Authored departure: real soil pH dynamics are complex buffer chemistry;
        # we use a simple production/relaxation model calibrated to produce
        # biologically relevant acidic zones (~pH 5.5) at patch edges after 60s.
        ph = self.C[self._idx_ph]
        self.C[self._idx_ph] = (ph + dt_t*(
            - 0.30 * dead_matter          # acidification from decomposition
            + 0.008 * (0.5 - ph)          # relaxation toward neutral -- faster to prevent corner drift
        )).clamp(0.25, 0.70)             # hard limits: pH 5.5 - pH 8.4

        # Ascarosides
        asc = self.C[self._idx_asc]
        self.C[self._idx_asc] = (asc + dt_t*(3e-9*dead_matter - 4.0*asc*metabolic)).clamp(min=0.0)

        # Oxygen: consumption + uniform background replenishment
        self.C[self._idx_O2] = (
            self.C[self._idx_O2] - dt_t*0.004*metabolic
            + dt_t*0.002*(0.21 - self.C[self._idx_O2])
        ).clamp(min=0.0)

        # CO2: production + venting at x-boundary (left edge)
        self.C[self._idx_co2] = (self.C[self._idx_co2] + dt_t*0.025*metabolic).clamp(min=0.0)
        self.C[self._idx_co2, :3, :] *= (1.0 - dt_t*0.015)

        # ── PDE: diffusion + decay (already on GPU) ───────────────────────
        C_view = self.C.unsqueeze(1)                              # (NF,1,nx,nz)
        lap    = F.conv2d(C_view, self._lap_kernel, padding=1).squeeze(1) * self._inv_dx2
        self.C = (self.C + dt_t*(self._D_gpu*lap - self._K_gpu*self.C)).clamp(min=0.0)

        # ── reseeding (every 5s — CPU, rare) ─────────────────────────────
        self._seed_timer += dt_t
        if self._seed_timer >= self._seed_interval:
            self._seed_timer = 0.0
            self._reseed_bacteria_cpu()

        self.step_count += 1
        if self.step_count % self.cfg.save_every_n == 0:
            self._save_frame()

    def _reseed_bacteria_cpu(self):
        """Pull B to CPU, reseed within patch radii, push back. Happens every 5s."""
        if not hasattr(self, '_patches') or not self._patches:
            return
        B_cpu  = self.B.cpu().numpy()
        O2_cpu = self.C[self._idx_O2].cpu().numpy()
        for cx, cz, r_grid in self._patches:
            # Build mask: within patch radius AND oxygenated
            xi = np.arange(self.nx)
            zi = np.arange(self.nz)
            XX, ZZ = np.meshgrid(xi, zi, indexing='ij')
            dist = np.sqrt((XX - cx)**2 + (ZZ - cz)**2)
            mask = (dist <= r_grid) & (O2_cpu > 0.03)
            candidates = np.argwhere(mask)
            if len(candidates) == 0:
                continue
            n_new = np.random.randint(1, 4)
            chosen = candidates[np.random.choice(
                len(candidates), min(n_new, len(candidates)), replace=False)]
            for ix, iz in chosen:
                r = np.random.randint(1, 4)
                x0, x1 = max(0, ix-r), min(self.nx, ix+r+1)
                z0, z1 = max(0, iz-r), min(self.nz, iz+r+1)
                B_cpu[x0:x1, z0:z1] = np.minimum(
                    1.0, B_cpu[x0:x1, z0:z1] + 0.05*np.random.rand())
        self.B.copy_(torch.from_numpy(B_cpu))

    # ── sample: GPU bilinear interp, return CPU dict ───────────────────────

    def sample(self, x_wu, y_wu, z_wu) -> dict:
        """
        Sample all fields at (x,z). Fully on GPU, returns CPU floats.
        No full array transfer — only 15 interpolated scalars.
        """
        fi = float(np.clip((x_wu/self.lx)*self.nx, 0.0, self.nx-1.001))
        fk = float(np.clip((z_wu/self.lz)*self.nz, 0.0, self.nz-1.001))
        i0,k0 = int(fi), int(fk)
        i1 = min(i0+1, self.nx-1); k1 = min(k0+1, self.nz-1)
        di,dk = fi-i0, fk-k0
        w00,w01,w10,w11 = (1-di)*(1-dk),(1-di)*dk,di*(1-dk),di*dk

        # GPU bilinear interp — no full transfer
        with torch.no_grad():
            c_now  = (self.C[:,i0,k0]*w00 + self.C[:,i0,k1]*w01 +
                      self.C[:,i1,k0]*w10 + self.C[:,i1,k1]*w11)
            c_prev = (self.C_prev[:,i0,k0]*w00 + self.C_prev[:,i0,k1]*w01 +
                      self.C_prev[:,i1,k0]*w10 + self.C_prev[:,i1,k1]*w11)
        c_now_cpu  = c_now.cpu().numpy()
        c_prev_cpu = c_prev.cpu().numpy()
        deriv      = (c_now_cpu - c_prev_cpu) / max(self.dt, 1e-10)

        T_celsius = float(c_now_cpu[self._idx_temp]) * 40.0
        self.Tc   = self.alpha_tc*T_celsius + (1.0-self.alpha_tc)*self.Tc

        result = {name: (float(c_now_cpu[i]), float(deriv[i]))
                  for i, name in enumerate(FIELD_NAMES)}
        result['Tc'] = self.Tc
        return result

    # ── deplete ───────────────────────────────────────────────────────────

    def deplete(self, x_wu, y_wu, z_wu, rate):
        fi = int(np.clip((x_wu/self.lx)*self.nx, 0, self.nx-1))
        fk = int(np.clip((z_wu/self.lz)*self.nz, 0, self.nz-1))
        # Small local operation — do on GPU
        with torch.no_grad():
            for di in range(-1,2):
                for dk in range(-1,2):
                    ii = int(np.clip(fi+di, 0, self.nx-1))
                    kk = int(np.clip(fk+dk, 0, self.nz-1))
                    self.B[ii,kk] = (self.B[ii,kk]*(1.0 - rate*self.dt)).clamp(min=0.0)

    # ── state persistence ─────────────────────────────────────────────────

    def save_state(self, path):
        np.savez_compressed(path,
            C=self.C.cpu().numpy(), B=self.B.cpu().numpy(),
            step_count=np.array([self.step_count]))
        print(f"[EnvSim] Saved state → {path} (step {self.step_count})")

    def load_state(self, path):
        data = np.load(path)
        expected = (N_FIELDS, self.nx, self.nz)
        if tuple(data['C'].shape) != expected:
            raise ValueError(
                f"Cache grid {data['C'].shape} != expected {expected}. "
                f"Delete env_warmup_cache.npz and regenerate.")
        self.C.copy_(torch.from_numpy(data['C']).to(self.device))
        self.B.copy_(torch.from_numpy(data['B']).to(self.device))
        self.step_count = int(data['step_count'][0])
        self.C_prev.copy_(self.C)
        print(f"[EnvSim] Loaded state ← {path} (step {self.step_count})")

    # ── HDF5 ──────────────────────────────────────────────────────────────

    def _open_hdf5(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._h5_path = os.path.join(self.output_dir, 'environment_sim.h5')
        self.h5 = h5py.File(self._h5_path, 'w')
        meta = self.h5.create_group('metadata')
        meta.attrs['grid']       = [self.nx, self.nz]
        meta.attrs['world_size'] = [self.lx, self.lz]
        meta.attrs['wu_to_m']   = WU_TO_M
        meta.attrs['dt_env']    = self.dt
        meta.attrs['field_names'] = FIELD_NAMES
        meta.attrs['D_eff']     = D_EFF.tolist()
        meta.attrs['K_decay']   = K_DECAY.tolist()
        meta.attrs['created']   = time.strftime('%Y-%m-%d %H:%M:%S')

        def ds(name, shape, maxshape, dtype='float32', compress=False, chunks=None):
            kw = {}
            if compress: kw['compression'] = 'gzip'; kw['compression_opts'] = 4
            if chunks:   kw['chunks'] = chunks
            return self.h5.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, **kw)

        cnx, cnz = max(1,self.nx//4), max(1,self.nz//4)
        self._h5_chem  = ds('environment/chem_fields',
                            (0,N_FIELDS,self.nx,self.nz), (None,N_FIELDS,self.nx,self.nz),
                            compress=True, chunks=(1,N_FIELDS,cnx,cnz))
        self._h5_bact  = ds('environment/bacterial_grid',
                            (0,self.nx,self.nz), (None,self.nx,self.nz),
                            compress=True, chunks=(1,cnx,cnz))
        self._h5_times = ds('environment/times', (0,), (None,))
        self._h5_sensory   = ds('sensory/currents',  (0,60), (None,60))
        self._h5_sensory_t = ds('sensory/times',     (0,),   (None,))
        self.h5.create_dataset('sensory/names', data=np.array([], dtype=h5py.string_dtype()))
        self._h5_nose     = ds('worm/nose_position', (0,3), (None,3))
        self._h5_muscle   = ds('body/muscle_dorsal',  (0,24),(None,24))
        self._h5_muscle_v = ds('body/muscle_ventral',(0,24),(None,24))
        self._h5_muscle_t = ds('worm/muscle_times',  (0,), (None,))
        self.h5.flush()
        print(f"[EnvSim] HDF5 → {self._h5_path}")

    def _save_frame(self):
        n = self._h5_chem.shape[0]
        self._h5_chem.resize(n+1, axis=0)
        self._h5_chem[n] = self.C.cpu().numpy()
        self._h5_bact.resize(n+1, axis=0)
        self._h5_bact[n] = self.B.cpu().numpy()
        self._h5_times.resize(n+1, axis=0)
        self._h5_times[n] = self.step_count * self.dt
        self.h5.flush()

    def save_nose_position(self, t, x, y, z):
        n = self._h5_nose.shape[0]
        self._h5_nose.resize(n+1, axis=0); self._h5_nose[n] = [x,y,z]

    def save_sensory_input(self, t, currents_arr, names=None):
        arr = np.zeros(60, dtype=np.float32)
        n_fill = min(len(currents_arr), 60)
        arr[:n_fill] = np.asarray(currents_arr[:n_fill], dtype=np.float32)
        n = self._h5_sensory.shape[0]
        self._h5_sensory.resize(n+1, axis=0); self._h5_sensory[n] = arr
        self._h5_sensory_t.resize(n+1, axis=0); self._h5_sensory_t[n] = t
        if n % 500 == 0: self.h5.flush()

    def save_muscle_activation(self, t, dorsal_arr, ventral_arr=None):
        d = np.zeros(24,dtype=np.float32); v = np.zeros(24,dtype=np.float32)
        nd = min(len(dorsal_arr),24); d[:nd] = np.asarray(dorsal_arr[:nd],dtype=np.float32)
        if ventral_arr is not None:
            nv = min(len(ventral_arr),24); v[:nv] = np.asarray(ventral_arr[:nv],dtype=np.float32)
        n = self._h5_muscle.shape[0]
        self._h5_muscle.resize(n+1,axis=0);   self._h5_muscle[n]   = d
        self._h5_muscle_v.resize(n+1,axis=0); self._h5_muscle_v[n] = v
        self._h5_muscle_t.resize(n+1,axis=0); self._h5_muscle_t[n] = t

    def close(self):
        if self.h5 is not None:
            self.h5.flush(); self.h5.close(); self.h5 = None
            print(f"[EnvSim] Closed {self._h5_path}")


if __name__ == '__main__':
    import time as _time
    print("Testing GPU-accelerated EnvironmentSimulator...")
    cfg = EnvConfig(colony_seed=7, nacl_seed=42)
    env = EnvironmentSimulator(output_dir='/tmp/env_gpu_test', config=cfg)
    env.reset()

    # Benchmark: 1000 steps
    print("Benchmarking 1000 steps...")
    torch.cuda.synchronize() if env.device.type=='cuda' else None
    t0 = _time.perf_counter()
    for _ in range(1000):
        env.step()
    torch.cuda.synchronize() if env.device.type=='cuda' else None
    t1 = _time.perf_counter()
    ms_per_step = (t1-t0)*1000/1000
    print(f"  {ms_per_step:.3f} ms/step  ({1000/ms_per_step:.0f} steps/s)")
    print(f"  Real-time factor: {0.001/((t1-t0)/1000):.1f}x  (target: >1.0x)")

    # Warmup check
    print("\nRunning 30s warmup (30000 steps)...")
    t0 = _time.perf_counter()
    for i in range(30000):
        env.step()
        if i % 5000 == 0:
            o2  = float(env.C[FIELD_IDX['oxygen']].mean())
            bmax= float(env.B.max())
            dia = float(env.C[FIELD_IDX['diacetyl']].max())
            print(f"  t={i*0.001:.0f}s: O2={o2:.3f} bact_max={bmax:.3f} dia_max={dia:.4e}")
    t1 = _time.perf_counter()
    print(f"  Warmup took {t1-t0:.1f}s wall time")

    # Gradient check
    dia_colony = float(env.C[FIELD_IDX['diacetyl'], 113, 45])
    dia_worm   = float(env.C[FIELD_IDX['diacetyl'], 147, 45])
    print(f"\n  Diacetyl at colony (ix=113): {dia_colony:.4e}")
    print(f"  Diacetyl at worm (ix=147):   {dia_worm:.4e}")
    print(f"  Gradient ratio: {dia_colony/max(dia_worm,1e-15):.1f}x")
    env.close()
    print("Done.")
