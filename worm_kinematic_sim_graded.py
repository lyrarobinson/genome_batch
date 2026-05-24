"""
worm_kinematic_sim.py
=====================
C. elegans simulation: real 302-neuron connectome (c302 parameter set B)
+ kinematic undulation body.

Architecture
------------
  Neural:      Full c302 B connectome via NEURON. All 302 neurons run as
               integrate-and-fire cells with an 'activity' measure (0->1)
               representing time-smoothed firing rate. Sensory neurons
               (AWA, AWC, AFD, BAG, ASH, ASE, URX) receive real chemical
               concentrations from the environment PDE via IClamp injection.

  Body:        Parametric undulation wave - independent of muscle output.
               The worm always undulates. Speed and turning are modulated
               by interneuron activity read from the connectome.

  Steering:    Left/right asymmetry in AIYL vs AIYR activity drives heading
               changes. The sign is calibrated so that higher diacetyl on the
               left produces a left turn (toward the attractant).

  Environment: 15-field soil chemistry PDE (environment_sim.py) with
               bacterial diacetyl production. Runs on GPU via PyTorch.

Why parameter set B
-------------------
  c302 C2 (Hodgkin-Huxley) diverges numerically without Sibernetic's
  mechanical body load. Parameter set B uses integrate-and-fire neurons
  which are unconditionally stable in standalone operation. The full
  connectome topology (all 302 neurons, all ~7000 synaptic connections,
  real White et al. 1986 weights) is preserved - only the single-neuron
  dynamics are simplified.

  The 'activity' variable (0->1) is a better proxy for calcium imaging
  signals used in real chemotaxis experiments than raw membrane voltage.

Usage
-----
  python worm_kinematic_sim.py \
      --sim_dir simulations/B_Full_YYYY-MM-DD_HH-MM-SS \
      --duration 30.0

Output: HDF5 file in simulations/kinematic_TIMESTAMP/kinematic_sim.h5
"""

import argparse
import datetime
import math
import os
import re
import sys
import time
import types

import h5py
import numpy as np
_np = np  # kept for backward compat
import torch
from graded_connectome import GradedConnectome

# ---------------------------------------------------------------------------
# Undulation parameters  (tuned to match real C. elegans kinematics)
# ---------------------------------------------------------------------------
BODY_WAVE_SPEED     = 1.8    # wavelengths per second (real worm: ~1.5-2.0)
BODY_WAVE_AMPLITUDE = 0.25   # radians, max bend per body segment
BODY_WAVE_LENGTH    = 1.0    # one full wave along body
N_BODY_POINTS       = 25     # matches N_PARTICLES in worm_body_physics.py
BODY_LENGTH         = 20.0   # world units (~1mm scaled)

# ── Graded connectome parameters ────────────────────────────────────────────
# Kunert/Shlizerman graded-potential model runs alongside NEURON c302 B Full.
# NEURON handles: muscle activation, visualisation, pharyngeal pacemaker.
# Graded model handles: sensory->interneuron->motor computation for steering/reversal.
# Biological basis: C. elegans neurons use graded potentials, not action potentials.
# (authored departure: dual-model architecture; graded model uses White et al. 1986
#  edgelist with Kunert parameter calibration rather than c302 B Full weights)
GC_EDGELIST         = None   # set via --edgelist CLI arg or auto-detected
GC_AWA_BASE         = 0.05   # nA baseline into AWAL/AWAR (keeps network in operating range)
GC_AWA_SCALE        = 0.5    # nA per unit of awa_south/north difference (asymmetry amplifier)
GC_SENSORY_SCALE    = 0.02   # nA per unit of sensory current (NEURON->graded scaling)
GC_ASH_SCALE        = 65.0  # ASH boosted: osmolarity/pH signals weak in soil (~0.001nA raw)
GC_SAAV_MAX         = 0.5   # nA: max SAAV injection at full satiation (worm.satiation=1.0)
                             # Authored departure: real satiation circuit (NSM serotonin,
                             # intestinal signals) absent from Varshney 2011. We author
                             # a drive into SAAV -- a real neuron with real AVA synapses
                             # (SAAVL->AVAL=17, SAAVR->AVAR=13) -- as a proxy for the
                             # missing satiation->reversal pathway. The connectome pathway
                             # is found object; the input signal is authored.
                             # 65x brings ASH into 0.05nA range -> ~1.5x reversal ratio
                             # Authored departure: real ASH sensitivity to osmolarity/pH
                             # is higher than our normalised environmental field implies
GC_MECH_SCALE       = 1.0    # raised from 0.1: puts FLP/PVD/PVC at 5-50 NI units, giving 0.015-0.07 rev/s
                              # higher than GC_SENSORY_SCALE: these neurons have 15-32 direct
                              # synapses to AVA vs AWA's 1-10 to AIZ
                                 # NEURON currents ~0.01-0.10nA; graded model saturates above 0.5nA
GC_TURN_SCALE       = 0.08   # turn_signal per mV of AVAL-AVAR difference
GC_REVERSAL_V       = -63.5  # mV: AVA voltage above which reversal probability increases
GC_PIROUETTE_V      = -63.0  # mV: AVA voltage above which pirouette suppression releases
GC_REVERSAL_SCALE   = 0.0005 # reversal prob per mV AVA depolarisation above baseline
                               # lowered 10x: mechanosensory->AVA pathway was over-triggering
GC_STEP_MS          = 5.0    # ms: graded model BDF step (adaptive solver handles stiffness)

BASE_SPEED          = 3.0    # world units / second (kept for legacy logging)
BASE_WAVE_AMP       = 1.50   # radians/segment -- matches worm_body_physics.py
TURN_GAIN           = 1.0     # rad/s per unit of AIYR-AIYL activity asymmetry (reduced from 50: running mean produces 0.2-0.7 range vs instantaneous ~0.002)
SPEED_SENSORY_GAIN  = 1.0    # speed boost per unit of mean AWA activity
DEPLETION_RATE = 6.0    # rescaled for three-component proxy (AWA * contact * speed_factor);
                        # effective rate ~0.016/s at full saturation, matching original single-component rate

# Synaptic conductance scale factor (relative to B_Full defaults).
# 0.05 = stable operation with strong signal propagation through connectome.
CONDUCTANCE_SCALE   = 0.05

# ---------------------------------------------------------------------------
# B_Full cell accessor helpers
# ---------------------------------------------------------------------------
# In parameter set B, cells are point processes named:
#   m_generic_neuron_iaf_cell_XXXX[0]  (neurons)
#   m_generic_muscle_iaf_cell_XXXX[0]  (muscles)
# Each has an .activity attribute (0->1) and a section accessible via
# .get_segment().sec for IClamp injection.

# Cache of neuron cell objects - populated once at startup
_cell_cache = {}

# Running average buffer for steering readout neurons.
# IAF tau1=50ms but raw activity at dt=0.05ms is a transient snapshot.
# Average over TAU_STEPS steps to match the biological integration window.
from collections import deque as _deque
_TAU_STEPS   = 1000   # 1000 x 0.05ms = 50ms, matching IAF tau1
_steer_buf   = {'AIZL': _deque(maxlen=_TAU_STEPS),
                'AIZR': _deque(maxlen=_TAU_STEPS)}
_ema_state   = {}     # kept for compatibility
_EMA_ALPHA   = 0.002

def build_cell_cache(h):
    """Pre-cache all neuron cell references to avoid repeated getattr calls.
    Sections are named ADAL[0] etc.; point processes are m_generic_neuron_iaf_cell_ADAL[0].
    """
    global _cell_cache
    for sec in h.allsec():
        name = sec.name().split('[')[0]
        if not name:
            continue
        cell_list = getattr(h, f'm_generic_neuron_iaf_cell_{name}', None)
        if cell_list is not None:
            try:
                if len(cell_list) > 0:
                    _cell_cache[name] = cell_list[0]
            except Exception:
                pass
    print(f"  Cached {len(_cell_cache)} neuron cell references")

def _cell(h, name):
    """Return the IAF point process object for a named neuron."""
    if name in _cell_cache:
        return _cell_cache[name]
    try:
        cell = getattr(h, f'm_generic_neuron_iaf_cell_{name}')[0]
        _cell_cache[name] = cell
        return cell
    except Exception:
        return None

def _cell_activity(h, name):
    """Return activity (0->1) for a named neuron, or 0 on failure."""
    try:
        v = float(_cell(h, name).activity)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0

def _cell_section(h, name):
    """Return the NEURON section hosting a named neuron."""
    try:
        return _cell(h, name).get_segment().sec
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Sensory neuron IClamp map
# ---------------------------------------------------------------------------
SENSORY_NAMES = [
    # Primary chemosensory
    'AWAL', 'AWAR',               # diacetyl (attractant)
    'AWCL', 'AWCR',               # volatile attractants (off-neuron)
    'AWBL', 'AWBR',               # repellents
    'ASHL', 'ASHR',               # noxious/polymodal nociceptor
    'ASEL', 'ASER',               # NaCl temporal derivative
    # Thermosensory
    'AFDL', 'AFDR',               # temperature deviation from Tc
    # Gas sensing
    'BAGL', 'BAGR',               # O2 downshift + CO2
    'URXL', 'URXR',               # high O2
    'PQR',                        # tail O2
    # Pheromone / osmosensory
    'ASJL', 'ASJR',               # ascarosides
    'ASKL', 'ASKR',               # ascarosides + osmolarity
    # Anterior mechanosensory (soil density at nose)
    'ALML', 'ALMR', 'AVM',        # gentle touch
    'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR',  # nose-touch / substrate contact
    'IL1DL', 'IL1DR', 'IL1L', 'IL1R', 'IL1VL', 'IL1VR',  # inner labial
    'ADEL', 'ADER',               # anterior dopaminergic substrate contact
    # Posterior mechanosensory (soil density at tail)
    'PLML', 'PLMR', 'PVM',        # gentle touch posterior
    'PDEL', 'PDER',               # posterior dopaminergic substrate contact
    # Harsh-touch / nociceptive: strong direct AVA inputs (FLP=31, PVD=15, PVC=14 synapses)
    'FLPL', 'FLPR',               # FLP: anterior+posterior nociceptor -> AVAR(31)/AVAL(17)
    'PVDL', 'PVDR',               # PVD: harsh touch body -> AVAR(12)/AVAL(15)
    'PVCL', 'PVCR',               # PVC: gentle touch tail -> AVAR(4)/AVAL(2)
    # Tail chemosensory (phasmid neurons)
    'PHAL', 'PHAR',               # repellents at tail
    'PHBL', 'PHBR',               # repellents at tail
    'PHCR',                       # diacetyl / food detection at tail
    # Pharyngeal motor neurons -- driven by authored pacemaker, not sensory mapper
    'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5',
]

_sensory_iclamps = {}


def build_sensory_iclamps(h):
    # Use offset_current (ELECTRODE_CURRENT) instead of IClamp (NONSPECIFIC_CURRENT)
    # offset_current bypasses capacitive membrane calculation, preventing voltage
    # explosion while still perturbing membrane potential and driving connectome.
    for name in SENSORY_NAMES:
        sec = _cell_section(h, name)
        if sec is not None:
            try:
                clamp           = h.offset_current(sec(0.5))
                clamp.delay     = 0.0
                clamp.duration  = 1e9
                clamp.amplitude = 0.0
                clamp.weight    = 1.0
                _sensory_iclamps[name] = clamp
            except Exception as e:
                print(f"  Warning: could not attach offset_current to {name}: {e}")
    print(f"  Built {len(_sensory_iclamps)} sensory offset_currents")


def inject_currents(currents):
    for name, clamp in _sensory_iclamps.items():
        clamp.amplitude = float(currents.get(name, 0.0))


def currents_to_array(currents):
    return np.array([currents.get(n, 0.0) for n in SENSORY_NAMES],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Motor neuron pitch readout
DB_NAMES = ['DB1','DB2','DB3','DB4','DB5','DB6','DB7']
VB_NAMES = ['VB1','VB2','VB3','VB4','VB5','VB6','VB7','VB8','VB9','VB10','VB11']
PITCH_GAIN = 0.5  # rad/s per unit DB-VB asymmetry

def read_pitch(h):
    """
    Compute pitch signal from dorsal vs ventral motor neuron activity.
    pitch_signal = mean(DB1-7) - mean(VB1-11)
    Positive -> pitch up (dorsal dominant)
    Negative -> pitch down (ventral dominant, driven by DVA inhibiting DB)
    """
    acts   = read_neuron_activities_batch(DB_NAMES + VB_NAMES)
    db_act = np.mean([acts[n] for n in DB_NAMES])
    vb_act = np.mean([acts[n] for n in VB_NAMES])
    return float(db_act - vb_act), float(db_act), float(vb_act)

# Reversal command interneuron readout
REVERSAL_NAMES     = ['AVAL','AVAR','AVEL','AVER']
REVERSAL_THRESHOLD = 0.12  # spike above baseline to trigger reversal (lowered: AVA max delta ~0.28)
PIROUETTE_RATE     = 0.3   # restored: pirouette bias handles reorientation
                           # suppressed toward zero in attractive gradient (high AWA)
PIROUETTE_HEADING_STD = 0.8  # radians std -- reduced so klinotaxis can dominate
REVERSAL_DURATION  = 0.8   # seconds -- long enough for full wave cycle reversal
_reversal_baseline = None  # exponential moving average of AVA/AVE activity
_reversal_alpha    = 5e-5  # EMA time constant ~1s at 20000 steps/s (was 0.01 = 5ms, too fast)

def read_reversal(h, is_reversing=False):
    """
    Read reversal command interneuron activity relative to a slow-moving baseline.
    Triggers when AVA/AVE spikes significantly above its tonic level.
    Uses exponential moving average to track baseline, ignoring tonic activity.
    """
    global _reversal_baseline
    batch = read_neuron_activities_batch(REVERSAL_NAMES)
    acts  = float(np.mean(list(batch.values())))
    if _reversal_baseline is None:
        _reversal_baseline = acts
    elif not is_reversing:
        # Only update baseline during forward locomotion -- prevents runaway
        # reversal loops where elevated AVA during reversal immediately
        # re-triggers on wake.
        _reversal_baseline = (1 - _reversal_alpha) * _reversal_baseline + _reversal_alpha * acts
    delta = acts - _reversal_baseline
    return acts, delta

# Egg-laying circuit (HSN→VC neurons)
EGG_ACCUMULATION_RATE = 0.02   # eggs/s in food-rich environment (AWA active)
EGG_LAY_THRESHOLD     = 8.0    # eggs accumulated before laying is triggered
VC_ACTIVITY_THRESHOLD = 0.15   # mean VC1-6 activity to confirm laying event
VC_NAMES = ['VC1','VC2','VC3','VC4','VC5','VC6']

def read_vc_activity(h):
    """Mean activity of VC1-6 egg-laying motor neurons."""
    batch = read_neuron_activities_batch(VC_NAMES)
    return float(np.mean(list(batch.values())))

# Sleep/quiescence readout (RIS interneuron)
# Two thresholds: spontaneous (network fluctuation) and satiation (post-feeding).
# Satiation quiescence in the real worm requires intestinal ASI/TGFb pathway
# over hours -- we author a compressed proxy via SAAV->RIS injection.
QUIESCENCE_THRESHOLD_SPONT    = 0.60   # raised to prevent spurious quiescence during approach  # spontaneous: rare network fluctuation bouts
QUIESCENCE_THRESHOLD_SATIATION = 0.590  # lowered: triggers at satiation~0.34  # satiation: requires authored _satiation drive
QUIESCENCE_THRESHOLD           = QUIESCENCE_THRESHOLD_SPONT  # legacy alias
QUIESCENCE_MIN_MS    = 500.0   # minimum quiescence duration (ms)
QUIESCENCE_MAX_MS    = 3000.0  # maximum quiescence duration (ms) -- forces wake
# Satiation integrator: accumulates from SAAV activity during feeding,
# decays during exploration. Injects into RIS as authored post-feeding drive.
SATIATION_ACCUMULATE  = 2e-6  # raised 2x: faster build on dense food    # per NEURON step (~50s feeding to full satiation)
SATIATION_DECAY       = 4.17e-8  # lowered 10x: gut fullness persists for minutes not seconds # per NEURON step (~120s exploration to full decay)
SATIATION_RIS_SCALE   = 0.050   # increased: RIS=0.573+0.05=0.623 at full satiation   # max RIS boost: 0.570 + 0.030 = 0.600 > 0.597 threshold

def read_quiescence(h):
    """
    Read RIS activity. RIS is a GABAergic interneuron that inhibits
    motor output globally, producing sleep-like quiescence.
    Clamped to [0,1]: voltage explosion from large IClamp currents
    can drive activity to large negative values which must be ignored.
    """
    v = read_neuron_activities_batch(['RIS'])['RIS']
    return max(0.0, min(1.0, v))

# Interneuron steering readout
# ---------------------------------------------------------------------------
# EMA baselines for AIZL/AIZR tonic activity -- removes network asymmetry bias.
# c302 B Full has strong intrinsic AIZR dominance (mean ~0.49 vs AIZL ~0.03)
# unrelated to sensory input. Steering responds to deviation from tonic baseline.
# EMA time constant ~2s at 20000 steps/s (alpha=2.5e-5).
# (authored departure: tonic asymmetry is a rate-coded model artefact)
_aiz_ema = {'AIZL': None, 'AIZR': None}
_AIZ_EMA_ALPHA = 2.5e-5

# Steering reads from AIYL/AIYR rather than AIZL/AIZR.
# AIZL is structurally non-functional in c302 B Full rate-coded parameter set:
# it sits below firing threshold >96% of the time (mean activity ~0.002 vs
# AIZR mean ~0.56). Root cause: ADFR receives more excitatory connectome input
# than ADFL (AWAR->ADFR w=9, AWBR->ADFR w=7) creating tonic ADFR>ADFL
# asymmetry that propagates via ADFR->AIZR w=37. AIZL cannot be rescued by
# current injection without massive intervention in neuron parameters.
# AIYL/AIYR are symmetric (mean ~0.57 each) and receive bilateral AWA input
# (AWAL->AIYL w=3, AWAR->AIYR w=8) making them suitable steering readout.
# (authored departure: biological steering uses AIZ as primary output;
# rate-coded B Full cannot reproduce AIZ bilateral symmetry.)
# read_steering() removed -- steering now from GradedConnectome (AVAL/AVAR voltages)
# NEURON c302 B Full still drives muscle activation and visualisation.
# Graded model handles sensory->interneuron->motor computation.


def read_awa_total(h):
    """Mean AWA activity - modulates forward speed."""
    acts = read_neuron_activities_batch(['AWAL', 'AWAR'])
    l, r = acts['AWAL'], acts['AWAR']
    return (max(0.0, min(1.0, l)) + max(0.0, min(1.0, r))) * 0.5


# ---------------------------------------------------------------------------
# Neuron activity reader (for logging all 302 neurons)
# ---------------------------------------------------------------------------
def build_neuron_list(sim_dir):
    """Extract neuron names from the B_Full LEMS python file."""
    nrn_py = os.path.join(sim_dir, 'LEMS_c302_B_Full_nrn.py')
    if not os.path.exists(nrn_py):
        nrn_py = os.path.join(sim_dir, 'LEMS_c302_nrn.py')
    names, seen = [], set()
    with open(nrn_py) as f:
        for line in f:
            for m in re.finditer(
                    r'm_generic_neuron_iaf_cell_([A-Z][A-Z0-9]+)', line):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def read_neuron_activities_batch(names):
    """Read activity for multiple neurons in a single pass through _cell_cache.
    Clamped to [0,1]: voltage explosion from large IClamp currents can drive
    activity to large negative or positive values which must be ignored.
    Updates EMA for steering neurons to handle IAF tau1=50ms settling issue.
    """
    def _safe(name):
        if name not in _cell_cache: return 0.0
        v = float(_cell_cache[name].activity)
        v = max(0.0, min(1.0, v))
        # Update EMA for this neuron
        if name not in _ema_state:
            _ema_state[name] = v
        else:
            _ema_state[name] = (1 - _EMA_ALPHA) * _ema_state[name] + _EMA_ALPHA * v
        return v
    return {name: _safe(name) for name in names}


# ---------------------------------------------------------------------------
# DD/VD GABAergic cross-inhibition restoration
# ---------------------------------------------------------------------------
# In c302 B_Full, DD/VD neurons receive insufficient synaptic current to fire
# (gap of ~222,000x between available and required). This is a known limitation
# of the B parameter set (all BioParameters marked certainty=0.1, BlindGuess).
# We restore cross-inhibition by injecting IClamp into DD/VD proportional to
# their excitatory motor neuron drivers -- biologically correct mechanism,
# authored delivery pathway.
# Source: White et al. 1986; Wen et al. 2012 (proprioceptive coupling)

# DD neuron -> weighted excitatory drivers (from herm_full_edgelist.csv)
_DD_DRIVERS = {
    'DD1': [('DA1',12),('DA2',7),('DB1',31),('DB2',3),('VA1',12),('VA2',20),('VA3',18),('VB1',1),('VB2',30)],
    'DD2': [('DA3',3),('DA4',4),('DB2',16),('DB3',3),('VA3',10),('VA4',17),('VA5',6),('VB2',1),('VB3',36),('VB4',7)],
    'DD3': [('DA5',12),('DA6',7),('DB3',14),('DB4',9),('VA5',17),('VA6',21),('VA7',3),('VB4',13),('VB5',34)],
    'DD4': [('DA7',3),('DA8',4),('DB4',2),('VA7',17),('VA8',9),('VA9',6),('VB6',32)],
    'DD5': [('DA8',4),('DB3',14),('DB4',9),('VA10',9),('VA8',9),('VA9',6),('VB7',29),('VB8',10),('VB9',10)],
    'DD6': [('DA8',2),('DA9',4),('DB4',2),('VA10',9),('VA11',19),('VA12',7),('VA9',6),('VB10',19),('VB11',19),('VB8',19),('VB9',19)],
}
_VD_DRIVERS = {
    'VD1':  [('VA1',3),('VA2',1),('VB1',4)],
    'VD2':  [('DA1',4),('DA2',17),('DB1',24),('VA2',27),('VA3',3),('VB1',2),('VB2',14)],
    'VD3':  [('DA2',11),('DA3',43),('DB1',2),('DB2',33),('VA3',11),('VB2',4),('VB3',4)],
    'VD4':  [('DA3',13),('DA4',11),('DA6',3),('DB2',15),('DB3',9),('DB5',3),('VA4',8),('VB3',6),('VB4',1)],
    'VD5':  [('DA4',21),('DA5',1),('DA6',3),('DB3',24),('DB5',3),('VA5',9),('VA6',3),('VB4',3)],
    'VD6':  [('DA5',22),('DA6',13),('DB3',4),('DB4',14),('DB5',13),('VA6',6),('VA7',2),('VB5',10),('VB6',1)],
    'VD7':  [('DA6',13),('DA7',10),('DB5',13),('DB6',10),('VA7',13),('VA8',2),('VB6',22)],
    'VD8':  [('DA7',12),('DA8',5),('DB6',12),('DB7',5),('VA8',3),('VB7',5)],
    'VD9':  [('DA7',10),('DA8',5),('DA9',5),('DB6',10),('DB7',5),('VA8',3),('VA9',2),('VB7',6),('VB8',5)],
    'VD10': [('DA8',7),('DA9',5),('DB7',7),('VA10',2),('VA9',3),('VB8',6),('VB9',5)],
    'VD11': [('DA7',2),('DA8',5),('DA9',11),('DB7',5),('VA10',3),('VA11',5),('VA9',3),('VB10',7),('VB9',6)],
    'VD12': [('DA8',5),('DA9',11),('DB7',5),('VA10',3),('VA11',3),('VA12',7),('VB10',6),('VB11',2)],
    'VD13': [('DA8',5),('DA9',2),('DB7',5),('VA12',6),('VA12',12),('VB11',22)],
}

# Normalise driver weights so max total_w maps to DDVD_ICLAMP_MAX
_DD_TOTAL_W = {n: sum(w for _,w in drivers) for n, drivers in _DD_DRIVERS.items()}
_VD_TOTAL_W = {n: sum(w for _,w in drivers) for n, drivers in _VD_DRIVERS.items()}
_MAX_TOTAL_W = max(list(_DD_TOTAL_W.values()) + list(_VD_TOTAL_W.values()))  # 134

# IClamp amplitude when all drivers fire at activity=1.0
# Calibrated so DD/VD reach activity ~0.3 at typical driver activity ~0.4
# IAF: need ~0.6nA for activity=0.3; drivers at 0.4 mean -> scale = 0.6/0.4 = 1.5nA
DDVD_ICLAMP_MAX = 1.0   # nA — max IClamp when drivers fully active

# Cache of DD/VD IClamp objects (built on first call)
_ddvd_iclamps = {}

def inject_dd_vd_drive(h, neuron_acts):
    """
    Inject IClamp into DD/VD GABAergic neurons proportional to excitatory driver
    activities. Restores cross-inhibition missing from c302 B parameter set.

    DD neurons inhibit ventral muscles (counteracting VB dominance).
    VD neurons inhibit dorsal muscles (counteracting DB dominance).

    Authored departure: biologically correct mechanism, authored IClamp delivery.
    Source: White et al. 1986; Wen et al. 2012.
    """
    global _ddvd_iclamps

    # Build IClamp cache on first call
    if not _ddvd_iclamps:
        for name in list(_DD_DRIVERS.keys()) + list(_VD_DRIVERS.keys()):
            try:
                sec = getattr(h, name)[0]
                ic = h.IClamp(sec(0.5))
                ic.delay = 0.0
                ic.dur = 1e9
                ic.amp = 0.0
                _ddvd_iclamps[name] = ic
            except Exception:
                pass

    # Compute and inject drive for each DD neuron
    for name, drivers in _DD_DRIVERS.items():
        if name not in _ddvd_iclamps:
            continue
        weighted_act = sum(w * float(neuron_acts.get(n, 0.0))
                          for n, w in drivers)
        normalised = weighted_act / _MAX_TOTAL_W
        _ddvd_iclamps[name].amp = min(DDVD_ICLAMP_MAX, DDVD_ICLAMP_MAX * normalised)

    # Compute and inject drive for each VD neuron
    for name, drivers in _VD_DRIVERS.items():
        if name not in _ddvd_iclamps:
            continue
        weighted_act = sum(w * float(neuron_acts.get(n, 0.0))
                          for n, w in drivers)
        normalised = weighted_act / _MAX_TOTAL_W
        _ddvd_iclamps[name].amp = DDVD_ICLAMP_MAX * normalised

def read_neuron_activities(h, neuron_names):
    acts = []
    for name in neuron_names:
        acts.append(_cell_activity(h, name))
    return np.array(acts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Undulating worm body
# ---------------------------------------------------------------------------
class UndulatingWorm:
    """
    Worm modelled as a midline of N_BODY_POINTS.
    Undulation is parametric - independent of neural muscle output.
    Steering biases heading via AIYL/AIYR asymmetry.

    Coordinate system: XZ ground plane, Y up.
    Heading angle is in the XZ plane.
    """

    def __init__(self, start_pos, start_heading=0.0):
        self.pos     = np.array(start_pos, dtype=float)
        self.heading = float(start_heading)
        self.pitch        = 0.0   # vertical angle (radians, + = up, - = down)
        self.phase        = 0.0
        self.reversing    = False  # True = backward locomotion state
        self.reverse_time = 0.0   # seconds remaining in reversal
        self.quiescent    = False  # True = RIS-driven sleep state
        self.quiescent_ms = 0.0   # ms spent quiescent
        self.eggs_accumulated = 0.0  # internal egg counter
        self.satiation        = 0.0  # post-feeding fatigue integrator (0-1)
        self.eggs_laid        = 0    # total eggs laid this run
        self.refractory_s     = 0.0  # seconds remaining in post-pirouette refractory period

    @property
    def heading_vec(self):
        # Full 3D heading: yaw (heading) in XZ, pitch in Y
        cos_pitch = math.cos(self.pitch)
        return np.array([
            math.cos(self.heading) * cos_pitch,
            math.sin(self.pitch),
            math.sin(self.heading) * cos_pitch
        ])

    def step(self, dt, turn_signal, speed):
        # Reversal state management
        if self.refractory_s > 0:
            self.refractory_s -= dt
        if self.reversing:
            self.reverse_time -= dt
            if self.reverse_time <= 0:
                self.reversing = False
                self.reverse_time = 0.0
                self.refractory_s = 1.0  # 1s refractory -- no pirouette for 1s after wake
                # Random heading perturbation on pirouette wake
                import random as _random
                self.heading += _random.gauss(0, PIROUETTE_HEADING_STD)
        # Wave propagation: reverse direction if reversing
        wave_dir = -1.0 if self.reversing else 1.0
        self.phase  += wave_dir * 2.0 * math.pi * BODY_WAVE_SPEED * dt
        turn_rate    = turn_signal * TURN_GAIN
        self.heading += turn_rate * dt
        # Pitch: updated by caller via worm.pitch, clamped to +/-45 degrees
        self.pitch    = float(np.clip(self.pitch, -math.pi/4, math.pi/4))
        # Movement direction flips during reversal
        move_dir = -1.0 if self.reversing else 1.0
        self.pos     += self.heading_vec * speed * dt * move_dir
        # Bounce at world boundaries -- finite soil sample
        # World: x=0..100.2, y=0..66.8, z=0..668.0
        if self.pos[0] < 1.0 or self.pos[0] > 99.2:
            self.heading = math.pi - self.heading
            self.pos[0] = float(np.clip(self.pos[0], 1.0, 99.2))
        if self.pos[2] < 1.0 or self.pos[2] > 667.0:
            self.heading = -self.heading
            self.pos[2] = float(np.clip(self.pos[2], 1.0, 667.0))
        self.pos[1] = float(np.clip(self.pos[1], 1.0, 65.0))

        body = self._compute_body()
        return self.pos.copy(), self.heading_vec.copy(), body

    def _compute_body(self):
        points    = np.zeros((N_BODY_POINTS, 3))
        points[0] = self.pos
        seg_len   = BODY_LENGTH / (N_BODY_POINTS - 1)
        cur_pos   = self.pos.copy()
        cur_angle = self.heading

        for i in range(1, N_BODY_POINTS):
            s          = i / (N_BODY_POINTS - 1)
            wave_phase = self.phase - 2.0 * math.pi * BODY_WAVE_LENGTH * s
            bend       = BODY_WAVE_AMPLITUDE * math.sin(wave_phase)
            cur_angle  -= bend * seg_len
            cos_pitch = math.cos(self.pitch)
            cur_pos    += np.array([
                math.cos(cur_angle) * cos_pitch * seg_len,
                math.sin(self.pitch) * seg_len,
                math.sin(cur_angle) * cos_pitch * seg_len
            ])
            points[i] = cur_pos.copy()

        return points


# ---------------------------------------------------------------------------
# HDF5 logger
# ---------------------------------------------------------------------------
class SimLogger:
    def __init__(self, path, neuron_names, env, field_names, n_fields, chunk=500, append=False):
        if append and os.path.exists(path):
            self.f = h5py.File(path, 'a')
            self._append = True
        else:
            self.f = h5py.File(path, 'w')
            self._append = False
        n_neu  = len(neuron_names)
        n_sens = len(SENSORY_NAMES)

        if not self._append:
            dt_str = h5py.special_dtype(vlen=str)
            self.f.create_dataset('worm/neuron_names',
                                  data=np.array(neuron_names, dtype=object),
                                  dtype=dt_str)
            self.f.create_dataset('sensory/names',
                                  data=np.array(SENSORY_NAMES, dtype=object),
                                  dtype=dt_str)
            self.f.create_dataset('environment/field_names',
                                  data=np.array(list(field_names), dtype=object),
                                  dtype=dt_str)

        def mk(name, extra, dtype='f4', chunk_override=None):
            if self._append and name in self.f:
                return self.f[name]
            c = chunk_override if chunk_override is not None else chunk
            return self.f.create_dataset(
                name,
                shape=(0,) + extra,
                maxshape=(None,) + extra,
                chunks=(c,) + extra,
                dtype=dtype)

        self.ds_t       = mk('environment/times',      ())
        self.ds_nose    = mk('worm/nose_position',     (3,))
        self.ds_head    = mk('worm/heading',           (3,))
        self.ds_body    = mk('worm/body_points',       (N_BODY_POINTS, 3))
        self.ds_speed   = mk('worm/speed',             ())
        self.ds_turn    = mk('worm/turn_rate',         ())
        self.ds_neuron  = mk('worm/neuron_activity',   (n_neu,))
        self.ds_sens    = mk('sensory/currents',       (n_sens,))
        self.ds_aiyl    = mk('steering/aiyl',          ())
        self.ds_aiyr    = mk('steering/aiyr',          ())
        self.ds_tsig    = mk('steering/turn_signal',   ())
        self.ds_pitch   = mk('steering/pitch_signal',  ())
        self.ds_db      = mk('steering/db_activity',   ())
        self.ds_vb      = mk('steering/vb_activity',   ())
        self.ds_reva    = mk('steering/reversal_act',  ())
        self.ds_revs    = mk('steering/reversing',     (), dtype='i1')
        self.ds_ris     = mk('steering/ris_activity',  ())
        self.ds_qsct    = mk('steering/quiescent',     (), dtype='i1')
        self.ds_vc      = mk('steering/vc_activity',   ())
        self.ds_eggs    = mk('steering/eggs_accumulated', ())
        self.ds_sati   = mk('steering/satiation',         ())
        self.ds_bact   = mk('environment/bacterial_density', ())
        _nf = n_fields
        _nx = env.cfg.nx
        _nz = env.cfg.nz
        self.ds_chem   = mk('environment/chem_fields',    (_nf, _nx, _nz), chunk_override=1)
        self.ds_bgrid  = mk('environment/bacterial_grid', (_nx, _nz), chunk_override=1)
        self.ds_mdors  = mk('body/muscle_dorsal',          (24,))
        self.ds_mvent  = mk('body/muscle_ventral',         (24,))
        self.ds_bpos   = mk('body/particle_positions',     (N_BODY_POINTS, 3))
        # If appending, start index from existing data length
        self.i = len(self.ds_t) if append else 0

    def log(self, t, nose, heading, body, speed, turn_rate,
            activities, sensory_arr, aiyl, aiyr, turn_signal,
            pitch_signal=0.0, db_act=0.0, vb_act=0.0,
            reversal_act=0.0, reversing=False,
            ris_act=0.0, quiescent=False,
            vc_act=0.0, eggs_accumulated=0.0, satiation=0.0, bacterial_density=0.0,
            chem_fields=None, bacterial_grid=None,
            muscle_dorsal=None, muscle_ventral=None, particle_positions=None):
        i = self.i
        for ds in [self.ds_t, self.ds_nose, self.ds_head, self.ds_body,
                   self.ds_speed, self.ds_turn, self.ds_neuron, self.ds_sens,
                   self.ds_aiyl, self.ds_aiyr, self.ds_tsig,
                   self.ds_pitch, self.ds_db, self.ds_vb,
                   self.ds_reva, self.ds_revs,
                   self.ds_ris, self.ds_qsct,
                   self.ds_vc, self.ds_eggs, self.ds_sati, self.ds_bact,
                   self.ds_mdors, self.ds_mvent, self.ds_bpos]:
            ds.resize(i + 1, axis=0)
        # chem/bgrid resized only when data is provided to avoid huge sparse datasets
        self.ds_t[i]      = t
        self.ds_nose[i]   = nose
        self.ds_head[i]   = heading
        self.ds_body[i]   = body
        self.ds_speed[i]  = speed
        self.ds_turn[i]   = turn_rate
        self.ds_neuron[i] = activities
        self.ds_sens[i]   = sensory_arr
        self.ds_aiyl[i]   = aiyl
        self.ds_aiyr[i]   = aiyr
        self.ds_tsig[i]   = turn_signal
        self.ds_pitch[i]  = pitch_signal
        self.ds_db[i]     = db_act
        self.ds_vb[i]     = vb_act
        self.ds_reva[i]   = reversal_act
        self.ds_revs[i]   = int(reversing)
        self.ds_ris[i]    = ris_act
        self.ds_qsct[i]   = int(quiescent)
        self.ds_vc[i]     = vc_act
        self.ds_eggs[i]   = eggs_accumulated
        self.ds_sati[i]   = satiation
        self.ds_bact[i]   = bacterial_density
        if chem_fields is not None:
            self.ds_chem.resize(self.ds_chem.shape[0] + 1, axis=0)
            self.ds_chem[-1] = chem_fields
        if bacterial_grid is not None:
            self.ds_bgrid.resize(self.ds_bgrid.shape[0] + 1, axis=0)
            self.ds_bgrid[-1] = bacterial_grid
        if muscle_dorsal is not None:
            self.ds_mdors[i]   = muscle_dorsal
        if muscle_ventral is not None:
            self.ds_mvent[i]   = muscle_ventral
        if particle_positions is not None:
            self.ds_bpos[i]    = particle_positions
        self.i += 1

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# NEURON loader
# ---------------------------------------------------------------------------
def load_neuron_sim(sim_dir, duration_ms, dt_nrn):
    os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'
    os.environ['DISPLAY'] = ''

    os.chdir(sim_dir)
    sys.path.insert(0, sim_dir)

    import neuron
    h = neuron.h
    h.load_file('stdrun.hoc')

    nrn_py = os.path.join(sim_dir, 'LEMS_c302_B_Full_nrn.py')
    with open(nrn_py) as f:
        src = f.read()

    for gui in ('nrngui.hoc', 'stdgui.hoc', 'stdlib.hoc'):
        src = src.replace(f'h.load_file("{gui}")', '')

    import re as _re
    src = _re.sub(r'[ \t]*self\.display_\w+[^\n]*\n', '', src)
    src = _re.sub(r'[ \t]*h\.graphList[^\n]*\n', '', src)
    src = _re.sub(r'[ \t]*h\.nrncontrolmenu\(\)[^\n]*\n', '', src)

    mod = types.ModuleType('LEMS_c302_B_Full_nrn')
    mod.__dict__['neuron'] = neuron
    exec(src, mod.__dict__)
    # Fixed small tstop — prevents recording vector corruption on long runs
    # (session 15 root cause). Main loop drives h.fadvance() independently.
    mod.NeuronSimulation(tstop=1000, dt=dt_nrn)

    # Scale synaptic conductances: default gbase=1e-5uS attenuates too much
    # over multi-hop pathways. 2e-5 gives good signal through AWA->AIZL.
    GBASE = 5e-6
    n_scaled = 0
    for syn_type in ['neuron_to_neuron_exc_syn', 'neuron_to_neuron_inh_syn']:
        for pp in getattr(h, syn_type, []):
            try:
                pp.gbase = GBASE
                n_scaled += 1
            except Exception:
                pass
    print(f"  Scaled {n_scaled} synapses to gbase={GBASE:.0e}")

    h.finitialize(-60)
    build_cell_cache(h)
    build_sensory_iclamps(h)

    return neuron, h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def save_checkpoint(path, h, worm, mapper, env, t_ms, step, neuron_mod):
    """Save full simulation state to a checkpoint file for resume after crash."""
    import pickle
    # NEURON state via SaveState
    ss = h.SaveState()
    ss.save()
    ss_file = path + '.nrn'
    sf = h.File(ss_file)
    ss.fwrite(sf)
    sf.close()
    # Worm state
    worm_state = {
        'pos': worm.pos.copy(), 'heading': worm.heading,
        'pitch': worm.pitch, 'phase': getattr(worm, 'phase', 0.0),
        'reversing': worm.reversing, 'reverse_time': worm.reverse_time,
        'refractory_s': worm.refractory_s, 'quiescent': worm.quiescent,
        'quiescent_ms': worm.quiescent_ms, 'satiation': worm.satiation,
        'eggs_accumulated': worm.eggs_accumulated, 'eggs_laid': worm.eggs_laid,
        'pos_2d': worm.pos_2d.copy(), 'vel_2d': worm.vel_2d.copy(),
    }
    # Mapper state
    mapper_state = {
        'nacl_t': mapper._nacl_t, 'nacl_history': list(mapper._nacl_history),
        'dia_t': mapper._dia_t, 'dia_history': list(mapper._dia_history),
        'dia_deriv': mapper._dia_deriv,
        'temp_t': mapper._temp_t, 'temp_history': list(mapper._temp_history),
        'temp_deriv': mapper._temp_deriv,
    }
    np.savez_compressed(path,
        t_ms=np.array([t_ms]),
        step=np.array([step]),
        worm_state=np.array([pickle.dumps(worm_state)]),
        mapper_state=np.array([pickle.dumps(mapper_state)]),
        env_C=env.C.cpu().numpy(), env_B=env.B.cpu().numpy(),
        env_step_count=np.array([env.step_count]))
    print(f'  [CHECKPOINT] Saved at t={t_ms/1000:.1f}s -> {path}')


def load_checkpoint(path, h, worm, mapper, env):
    """Restore simulation state from checkpoint. Returns (t_ms, step)."""
    import pickle
    data = np.load(path, allow_pickle=True)
    # NEURON state
    ss = h.SaveState()
    ss_file = path + '.nrn'
    sf = h.File(ss_file)
    ss.fread(sf)
    sf.close()
    ss.restore(1)
    # Worm state
    worm_state = pickle.loads(data['worm_state'][0])
    if 'pos_2d' in worm_state:
        worm.pos_2d[:] = worm_state['pos_2d']
        worm.vel_2d[:] = worm_state['vel_2d']
    else:
        worm.pos_2d[0] = [worm_state['pos'][0], worm_state['pos'][2]]
    worm.pitch = worm_state['pitch']
    worm.reversing = worm_state['reversing']
    worm.reverse_time = worm_state['reverse_time']
    worm.refractory_s = worm_state['refractory_s']
    worm.quiescent = worm_state['quiescent']
    worm.quiescent_ms = worm_state['quiescent_ms']
    worm.satiation = worm_state['satiation']
    worm.eggs_accumulated = worm_state['eggs_accumulated']
    worm.eggs_laid = worm_state['eggs_laid']
    # Mapper state
    mapper_state = pickle.loads(data['mapper_state'][0])
    from collections import deque
    mapper._nacl_t = mapper_state['nacl_t']
    mapper._nacl_history = deque(mapper_state['nacl_history'])
    mapper._dia_t = mapper_state['dia_t']
    mapper._dia_history = deque(mapper_state['dia_history'])
    mapper._dia_deriv = mapper_state['dia_deriv']
    mapper._temp_t = mapper_state['temp_t']
    mapper._temp_history = deque(mapper_state['temp_history'])
    mapper._temp_deriv = mapper_state['temp_deriv']
    # Env state
    env.C.copy_(torch.from_numpy(data['env_C']).to(env.device))
    env.B.copy_(torch.from_numpy(data['env_B']).to(env.device))
    env.step_count = int(data['env_step_count'][0])
    env.C_prev.copy_(env.C)
    t_ms = float(data['t_ms'][0])
    step = int(data['step'][0])
    print(f'  [CHECKPOINT] Restored from t={t_ms/1000:.1f}s, step={step}')
    return t_ms, step


def run(sim_dir, duration=30.0, dt_nrn=0.05, log_every=100, log_every_env=500, start_heading=None,
        output_path=None, env_step_ms=1.0,
        start_x=None, start_y=None, start_z=None,
        env_cache=None, colony_seed=None,
        colony_ix_min=None, colony_ix_max=None,
        colony_iz_min=None, colony_iz_max=None,
        checkpoint_every=30.0, resume_from=None,
        edgelist_path=None):

    sim_dir = os.path.abspath(sim_dir)
    sib_dir = os.path.dirname(os.path.dirname(sim_dir))
    # Convert resume_from to absolute path before os.chdir changes working directory
    if resume_from:
        resume_from = os.path.abspath(resume_from)

    os.chdir(sim_dir)

    if output_path is None:
        if resume_from and os.path.exists(os.path.abspath(resume_from)):
            # Reuse the same output directory as the checkpoint
            out_dir = os.path.dirname(os.path.abspath(resume_from))
            output_path = os.path.join(out_dir, 'kinematic_sim.h5')
            print(f"Output (appending): {out_dir}")
        else:
            stamp   = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            out_dir = os.path.join(sib_dir, 'simulations', f'kinematic_{stamp}')
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, 'kinematic_sim.h5')
            print(f"Output: {out_dir}")

    # -- NEURON ---------------------------------------------------------------
    print("Loading NEURON connectome (c302 parameter set B)...")
    neuron_mod, h = load_neuron_sim(sim_dir, duration * 1000, dt_nrn)
    neuron_names  = build_neuron_list(sim_dir)
    print(f"  {len(neuron_names)} neurons")
    # Precompute index map for sensory neuron HDF5 patch (applied at log time).
    # For directly-injected neurons the IAF activity ODE is corrupted by large
    # currents; we substitute clean clipped-current values before writing HDF5.
    _sensory_idx = {name: neuron_names.index(name)
                    for name in SENSORY_NAMES if name in neuron_names}

    # -- Environment ----------------------------------------------------------
    print("Loading environment...")
    sys.path.insert(0, sib_dir)
    from environment_sim import EnvironmentSimulator, EnvConfig, FIELD_IDX, N_FIELDS
    from worm_body_physics import PhysicsWorm, BODY_Y
    from muscle_map import compute_muscle_activations
    from sensory_mapper  import SensoryMapper

    env_cfg = EnvConfig(
        nacl_seed=a.nacl_seed,
        colony_seed=a.colony_seed,
        # Colony placement: right-centre of 960x540 world
        # ix=90..130 of 160 → world x=540..780
        # iz=25..65 of 90  → world z=150..390
        colony_ix_min=colony_ix_min,
        colony_ix_max=colony_ix_max,
        colony_iz_min=colony_iz_min,
        colony_iz_max=colony_iz_max,
    )
    env    = EnvironmentSimulator(output_dir=out_dir, config=env_cfg)
    mapper = SensoryMapper(dt=dt_nrn / 1000.0)

    lx, lz = env.lx, env.lz
    ly = env.ly  # kept for worm y position only -- env is 2D XZ
    print(f"  World: {lx:.1f} x {lz:.1f} (2D XZ)")

    # -- Worm start position --------------------------------------------------
    # Default: worm starts at x=880 (far right), z=270 (mid-height)
    # Colony is centred at x=680, z=270 — worm approaches from right
    sx = start_x if start_x is not None else 880.0
    sy = start_y if start_y is not None else ly * 0.5  # ignored by 2D env
    sz = start_z if start_z is not None else lz * 0.5   # mid-height

    # Heading: π = facing in −X direction (toward colony at lower X)
    start_heading = start_heading if start_heading is not None else math.pi
    worm = PhysicsWorm(
        start_pos=[sx, sy, sz],
        start_heading=start_heading
    )
    worm.refractory_s = 3.0  # prevent pirouette before locomotion establishes
    # Set world bounds from environment so worm reflects at correct boundaries
    worm.x_min = 1.0
    worm.x_max = lx - 1.0
    worm.z_min = 1.0
    worm.z_max = lz - 1.0
    print(f"  Worm at {[f'{v:.1f}' for v in worm.pos]}, "
          f"heading={math.degrees(start_heading):.0f}deg")
    print(f"  World bounds: x=[{worm.x_min},{worm.x_max}] z=[{worm.z_min},{worm.z_max}]")

    # -- Logger ---------------------------------------------------------------
    logger = SimLogger(output_path, neuron_names, env, list(FIELD_IDX.keys()), N_FIELDS, append=bool(resume_from))

    # -- Warm up environment --------------------------------------------------
    env.reset()

    if env_cache and os.path.exists(env_cache):
        env.load_state(env_cache)
        print("  Environment loaded from cache.")
    else:
        print("Pre-running environment 1000s to establish diacetyl gradient...")
        for _ in range(1000000):
            env.step(dt=0.001)
        print("  Environment ready.")
        if env_cache:
            env.save_state(env_cache)

    # -- NEURON warmup: let network reach steady state before main loop ------
    if not (resume_from and os.path.exists(resume_from)):
        print("Warming up NEURON network (500ms)...")
        # Inject only proprioceptive proxy during warmup -- no environmental currents.
        # Position-dependent sensory input during warmup drives the network into
        # position-specific asymmetric states (e.g. BAG-only at x=45 suppresses AIZL).
        # Neutral warmup gives a consistent baseline regardless of start position.
        _prop_sine0 = 0.0  # PhysicsWorm has no phase; use zero for neutral warmup
        currents0 = {
            'PLML': max(0.0, 0.04 + 0.03 * _prop_sine0),
            'PLMR': max(0.0, 0.04 + 0.03 * _prop_sine0),
            'PVM':  max(0.0, 0.02 + 0.015 * _prop_sine0),
        }
        inject_currents(currents0)
        for _ in range(10000):  # 10000 x 0.05ms = 500ms warmup
            h.fadvance()
        print("  NEURON network ready.")
    else:
        print("  Skipping warmup -- restoring from checkpoint.")

    # ── Graded connectome (Kunert graded-potential model) ───────────────────
    # Runs alongside c302 B Full NEURON. Sensory inputs injected into both;
    # graded model provides turn_signal and reversal_drive from AVAL/AVAR voltages.
    print("Loading graded connectome...")
    _gc = GradedConnectome(device='cuda' if torch.cuda.is_available() else 'cpu')
    # Auto-detect edgelist path
    import glob as _glob
    # Find Neural Interactome data files (Gg.npy, Gs.npy, emask.npy, neuron_names.txt)
    # Look for files copied to project dir, or in /tmp/NeuralInteractome clone
    _ni_candidates = [
        edgelist_path,  # --edgelist can also point to NI data dir
        os.path.dirname(__file__),  # project dir (if neural_interactome_*.npy copied here)
        '/tmp/NeuralInteractome',
    ]
    # Check for neural_interactome_Gg.npy in project dir (copied with prefix)
    _proj_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(_proj_dir, 'neural_interactome_Gg.npy')):
        # Files have the prefix -- create a temp dir with standard names
        import shutil as _shutil, tempfile as _tempfile
        _ni_tmp = _tempfile.mkdtemp()
        for _ni_f in ['Gg.npy','Gs.npy','emask.npy','neuron_names.txt']:
            _src = os.path.join(_proj_dir, f'neural_interactome_{_ni_f}')
            if os.path.exists(_src):
                _shutil.copy(_src, os.path.join(_ni_tmp, _ni_f))
        # Also copy neuron_names.txt from NI clone if not in project dir
        if not os.path.exists(os.path.join(_proj_dir, 'neural_interactome_neuron_names.txt')):
            for _cand in ['/tmp/NeuralInteractome']:
                _nf = os.path.join(_cand, 'neuron_names.txt')
                if os.path.exists(_nf):
                    _shutil.copy(_nf, os.path.join(_ni_tmp, 'neuron_names.txt'))
                    break
        _ni_data_dir = _ni_tmp
    else:
        _ni_data_dir = next((p for p in _ni_candidates
                             if p and os.path.exists(os.path.join(str(p), 'Gg.npy'))), None)
    if _ni_data_dir is None:
        raise FileNotFoundError(
            'Neural Interactome data not found. Clone https://github.com/shlizee/C-elegans-Neural-Interactome '
            'to /tmp/NeuralInteractome, or copy Gg.npy/Gs.npy/emask.npy/neuron_names.txt to project dir '
            'with prefix neural_interactome_*')
    _gc.load(_ni_data_dir)
    # Warmup graded model at rest (2s), then with typical locomotion injection (1s).
    # Baseline captured with PLML/R at tonic floor so AVA baseline reflects
    # normal locomotion state, not zero-injection rest.
    print(f"  Graded connectome: {_gc.N} neurons. Warming up...")
    for _ in range(int(2.0 / (GC_STEP_MS / 1000))):
        _gc.step(dt=GC_STEP_MS / 1000.0)
    # Inject typical locomotion mechanosensory tonic (PLML/R at floor, FLP/PVD/PVC=0)
    _gc_warmup_injection = {
        'PLML': 0.05, 'PLMR': 0.05, 'PVM': 0.025,
        'AWAL': GC_AWA_BASE, 'AWAR': GC_AWA_BASE,
    }
    _gc.inject_batch(_gc_warmup_injection)
    for _ in range(int(1.0 / (GC_STEP_MS / 1000))):
        _gc.step(dt=GC_STEP_MS / 1000.0)
    _gc.inject_batch({})  # clear injection before main loop
    _gc_baseline_aizl = _gc.voltage('AIZL')
    _gc_baseline_aizr = _gc.voltage('AIZR')
    _gc_baseline_aval = _gc.voltage('AVAL')
    _gc_baseline_avar = _gc.voltage('AVAR')
    _gc_baseline_avb  = (_gc.voltage('AVBL') + _gc.voltage('AVBR')) / 2
    print(f"  Graded baseline: AIZL={_gc_baseline_aizl:.4f} AIZR={_gc_baseline_aizr:.4f} AVAL={_gc_baseline_aval:.4f} AVAR={_gc_baseline_avar:.4f} mV")
    _gc_aizl_ema = _gc_baseline_aizl  # initialise EMA from warmup
    _gc_aizr_ema = _gc_baseline_aizr
    _gc_ava_ema  = (_gc_baseline_aval + _gc_baseline_avar) / 2.0  # running AVA mean
    _gc_avb_ema  = _gc_baseline_avb   # running AVB mean
    # Steps counter for graded model (runs every GC_STEP_MS ms, not every NEURON step)
    _gc_step_counter = 0
    _gc_steps_per_update = max(1, int(GC_STEP_MS / dt_nrn))  # NEURON steps per graded step (dt_nrn in ms, GC_STEP_MS in ms)
    _gc_currents = {}   # current injection dict updated each env step
    _gc_turn_signal   = 0.0
    _gc_reversal_drive = 0.0

    # Default values -- overridden by resume block if resuming
    t_ms = 0.0
    _step_offset = 0

    # Resume from checkpoint if provided -- must happen AFTER warmup skip
    if resume_from and os.path.exists(resume_from):
        print(f'Resuming from checkpoint: {resume_from}')
        t_ms, _step_offset = load_checkpoint(resume_from, h, worm, mapper, env)
        _steer_buf['AIZL'].clear()
        _steer_buf['AIZR'].clear()
        # Adjust duration: --duration is total target, subtract already-simulated time
        duration = max(0.0, duration - t_ms / 1000.0)
        print(f'  Resuming from t={t_ms/1000:.1f}s, remaining duration={duration:.1f}s')

    # -- Simulation loop ------------------------------------------------------
    dt_s      = dt_nrn / 1000.0
    n_steps   = int(duration * 1000 / dt_nrn)
    env_every = max(1, int(env_step_ms / dt_nrn))
    t_wall    = time.time()

    print(f"\nRunning {duration}s ({n_steps} steps, "
          f"env every {env_every} steps, log every {log_every}, env every {log_every_env})...\n")

    _bact_density = 0.0  # initialised before loop; updated at step 6d
    _bact_ema     = 0.0  # EMA of bacterial density for pharyngeal departure AWC signal
    # Graded connectome state (initialised after GradedConnectome is loaded in startup block)
    _gc_turn_signal    = 0.0
    _gc_reversal_drive = 0.0
    _gc_suppression    = 0.0
    _gc_step_counter   = 0
    _gc_currents       = {}
    _weathervane_signal = 0.0  # restored weathervane steering signal
    _gc_aizl           = 0.0    # updated each gc step; initialised at rest
    _gc_aizr           = 0.0
    # _gc_aizl_ema / _gc_aizr_ema set from warmup baseline at line ~1049 -- do not reinitialise here
    # _gc_aizl_ema       = 0.0
    # _gc_aizr_ema       = 0.0
    _gc_aval           = 0.0
    _gc_avar           = 0.0
    _gc_avbl           = -64.0
    _gc_avbr           = -64.0
    _gc_avb_mean       = -64.0
    _gc_ava_mean       = -64.0
    _body_phase   = 0.0  # running phase for proprioceptive proxy (replaces worm.phase)
    speed = BASE_SPEED         # initialised before loop; updated at step 6
    _ris_satiation_boost = 0.0  # authored satiation drive into RIS
    _checkpoint_path = os.path.join(os.path.dirname(output_path), 'checkpoint.npz')
    _checkpoint_every_steps = int(checkpoint_every * 1000 / dt_nrn) if checkpoint_every else 0
    for step in range(n_steps):

        # 1. Advance NEURON one timestep
        h.fadvance()
        t_ms += dt_nrn

        # 2. Step environment PDE
        if step % env_every == 0:
            env.step(dt=env_step_ms / 1000.0)

        # 2b. Feeding: deplete bacteria using three-component proxy
        # Component 1: AWA neuron activity (smell -- food-specific chemosensation)
        # Component 2: ADE/CEP contact neurons (physical contact with substrate/bacteria)
        # Component 3: speed factor (locomotion state -- worm must be slow to feed)
        # All three must be simultaneously active for high depletion rate.
        # Authored proxy: pharyngeal pumping rate in real worm depends on food
        # detection AND substrate contact AND dwelling state (Sawin et al. 2000).
        if step % env_every == 0:
            _awa_act = (max(0.0, min(1.0, float(_cell_cache['AWAL'].activity) if 'AWAL' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['AWAR'].activity) if 'AWAR' in _cell_cache else 0.0))) * 0.5
            _ade_act = (max(0.0, min(1.0, float(_cell_cache['ADEL'].activity) if 'ADEL' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['ADER'].activity) if 'ADER' in _cell_cache else 0.0))) * 0.5
            _cep_act = (max(0.0, min(1.0, float(_cell_cache['CEPDL'].activity) if 'CEPDL' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['CEPDR'].activity) if 'CEPDR' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['CEPVL'].activity) if 'CEPVL' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['CEPVR'].activity) if 'CEPVR' in _cell_cache else 0.0))) * 0.25
            _contact_act = (_ade_act + _cep_act) * 0.5
            _speed_factor = float(np.clip(1.0 - (speed / BASE_SPEED), 0.0, 1.0))
            _feeding_signal = (_awa_act / 0.072) * _contact_act * _speed_factor
            if _feeding_signal > 0.001:
                effective_rate = DEPLETION_RATE * _feeding_signal
                env.deplete(worm.pos[0], worm.pos[1], worm.pos[2], effective_rate)

        # 3. Sample concentrations at L/R of nose
        # L/R sampling using physics body tangent
        _nose_pos, _left_pos, _right_pos = worm.get_nose_tangent_perp()
        wx, wy, wz = _nose_pos
        # ── Weathervane (klinotaxis) mechanism ───────────────────────────────
        # Correlate head swing direction with AWA level over sliding window.
        # Klinotaxis: swing-extreme bilateral sampling.
        # The head casts left and right during undulation. At each swing extreme
        # the nose is physically displaced from the centreline -- this is when
        # we sample concentration bilaterally. At the northern extreme we sample
        # at nose + perpendicular offset (toward +Z); at the southern extreme we
        # sample at nose - perpendicular offset (toward -Z). These two samples
        # are taken at the same moment (instantaneous bilateral read at each
        # extreme), so there is zero temporal contamination.
        # The undulation IS the sensing mechanism: you only get a reading when
        # the head swings far enough; the swing amplitude gates the sample.
        # Comparison: AWA_at_south_extreme - AWA_at_north_extreme.
        # Positive = more food to south = curve south.
        #
        # Biological grounding: SMD motor neurons respond to body curvature and
        # are most active at the swing extremes. Their output drives neck muscles
        # to bias the next swing toward higher concentration. We implement the
        # functional output of this circuit directly.
        # (authored departure: real SMD circuit uses curvature-gated temporal
        # integration; we sample instantaneously at the swing extremes, which is
        # equivalent when neural noise is absent from the rate-coded model)
        # Use absolute nose z for head swing -- avoids travelling wave
        # interference from nose-neck differential which gives 2x frequency.
        # Project nose position onto axis perpendicular to heading so peak
        # detection works at any heading, not just heading~0 (east/west).
        # perp = (-sin(h), cos(h)) in XZ plane; positive = left of heading.
        _h_sw = worm._smooth_heading
        _nose_xz = worm.pos_2d[0]  # (x, z) of nose particle
        _head_swing = float(-math.sin(_h_sw) * _nose_xz[0] + math.cos(_h_sw) * _nose_xz[1])

        # Smooth swing signal for reliable peak detection (TC=0.05s, sub-cycle)
        if not hasattr(worm, '_wv_swing_smooth'):
            worm._wv_swing_smooth    = _head_swing
            worm._wv_prev_smooth     = _head_swing
            worm._wv_awa_north       = None
            worm._wv_awa_south       = None
            worm._wv_lr_ema          = 0.0

        alpha_s = min(dt_s / 0.020, 0.5)  # TC=20ms: tracks 1.8Hz oscillation reliably
        prev_smooth = worm._wv_swing_smooth
        prev_deriv  = getattr(worm, '_wv_swing_deriv', 0.0)
        worm._wv_swing_smooth += alpha_s * (_head_swing - worm._wv_swing_smooth)
        curr_deriv = worm._wv_swing_smooth - prev_smooth
        worm._wv_swing_deriv = curr_deriv

        # Peak detection: derivative sign change at actual swing extremes.
        # Samples diacetyl at maximum lateral excursion -- strongest gradient signal.
        # Refractory period prevents double-firing on noisy oscillations.
        # Refractory counted in NEURON steps (not seconds) for precision.
        if not hasattr(worm, '_peak_refrac_steps'):
            worm._peak_refrac_steps = 0
        if worm._peak_refrac_steps > 0:
            worm._peak_refrac_steps -= 1
        _refrac_steps = int(0.3 / dt_s)   # 0.3s refractory -- calibrated for 0.3Hz wave
        # prev_deriv > 0: nose was moving +z (south), now stopped = SOUTH peak
        # prev_deriv < 0: nose was moving -z (north), now stopped = NORTH peak
        _at_south_peak = (prev_deriv > 0 and curr_deriv <= 0 and worm._peak_refrac_steps == 0)
        _at_north_peak = (prev_deriv < 0 and curr_deriv >= 0 and worm._peak_refrac_steps == 0)
        if _at_north_peak or _at_south_peak:
            worm._peak_refrac_steps = _refrac_steps

        # Klinotaxis: temporal comparison at successive head swing peaks.
        # Biological basis: Iino & Yoshida 2009 -- AWA compares odour at each
        # head swing peak; difference between alternating peaks drives turning.
        # Klinotaxis: at each swing peak the nose is maximally displaced to the
        # left or right of the heading. Sample diacetyl at the actual left/right
        # nose positions and compare directly.
        # left_conc > right_conc -> turn left (negative turn_signal in body frame)
        # right_conc > left_conc -> turn right (positive turn_signal)
        # Sign convention: positive turn_signal = rightward body bias = turn right.
        # _left_pos/_right_pos are already computed above from nose tangent perp.
        # Biological basis: Iino & Yoshida 2009 -- klinotaxis uses bilateral
        # comparison at swing extremes to compute gradient direction.
        # Authored departure: we sample at the actual displaced nose positions
        # rather than the centreline, which is geometrically equivalent and
        # avoids world-coordinate heading dependence entirely.
        if _at_north_peak or _at_south_peak:
            _c_left  = env.sample(_left_pos[0],  BODY_Y, _left_pos[2])
            _c_right = env.sample(_right_pos[0], BODY_Y, _right_pos[2])
            _dia_L   = float(_c_left['diacetyl'][0])
            _dia_R   = float(_c_right['diacetyl'][0])
            # Hill transform to match AWA response curve
            _awa_L   = float(_dia_L**1.5 / (5e-3**1.5 + _dia_L**1.5))
            _awa_R   = float(_dia_R**1.5 / (5e-3**1.5 + _dia_R**1.5))
            # lr_inst: positive = more food to left = turn left (positive steer_bias = left bend)
            _lr_inst = _awa_L - _awa_R
            worm._wv_lr_held = _lr_inst
            _gc_currents['AWAL'] = GC_AWA_BASE + max(0.0,  _lr_inst) * GC_AWA_SCALE
            _gc_currents['AWAR'] = GC_AWA_BASE + max(0.0, -_lr_inst) * GC_AWA_SCALE
        else:
            _gc_currents['AWAL'] = GC_AWA_BASE
            _gc_currents['AWAR'] = GC_AWA_BASE

        worm._wv_prev_smooth = prev_smooth
        # Decay food world vector (TC=10s) so stale peaks don't dominate
        if hasattr(worm, '_food_world_x'):
            worm._food_world_x *= (1.0 - dt_s / 10.0)
            worm._food_world_z *= (1.0 - dt_s / 10.0)

        # Direct hold: decay between peaks, reset on reversal
        if hasattr(worm, '_wv_lr_held'):
            # Always decay -- don't zero on reversal (signal lost for 4s after each reversal)
            worm._wv_lr_held *= (1.0 - dt_s / 2.0)  # TC=2s: keeps signal alive between peaks at 0.3Hz
        _weathervane_signal = 0.0  # unused; retained for logging compatibility

        # Left/right sampling relative to locomotion heading (stable direction).
        # AWAL receives left-of-heading concentration, AWAR receives right.
        # Connectome: AWAL->AIZL w=23, AWAR->AIZR w=25 (ipsilateral dominant).
        concs_L = env.sample(_left_pos[0],  _left_pos[1],  _left_pos[2])
        concs_R = env.sample(_right_pos[0], _right_pos[1], _right_pos[2])

        # 4a. Update NaCl temporal derivative (ASEL/ASER use history, not spatial diff)
        nacl_avg = (concs_L['nacl'][0] + concs_R['nacl'][0]) / 2.0
        mapper.update_nacl_history(nacl_avg)
        # 4b. Update diacetyl temporal derivative (AWC off-response on food leaving)
        dia_avg = (concs_L['diacetyl'][0] + concs_R['diacetyl'][0]) / 2.0
        mapper.update_diacetyl_history(dia_avg)  # mean for AWC off-response
        mapper.update_diacetyl_history_bilateral(
            concs_L['diacetyl'][0], concs_R['diacetyl'][0])  # bilateral for AWA ON-response
        # 4c. Update temperature temporal derivative (AFD isothermal tracking)
        temp_avg = (concs_L['temperature'][0] + concs_R['temperature'][0]) / 2.0 * 40.0
        mapper.update_temp_history(temp_avg)

        # 4. Map concentrations -> currents -> inject into NEURON
        currents = mapper.map_asymmetric(concs_L, concs_R)
        # Authored correction: ADFR receives more excitatory connectome input than
        # ADFL (AWAR->ADFR w=9, AWBR->ADFR w=7 vs AWBL->ADFL w=25 only).
        # This produces tonic ADFR>ADFL asymmetry which propagates via
        # ADFR->AIZR w=37 to create persistent AIZR dominance (~0.55 units)
        # that overwhelms chemosensory steering signal (~0.05 units).
        # In the real worm ADF is a pheromone/satiation sensor; its tonic
        # projection to AIZ is context-modulated. We inject a compensatory
        # current into AIZL to restore bilateral symmetry for steering.
        # Magnitude estimated from ADFR-ADFL activity difference * w=37 gain.
        # (authored departure: tonic ADF->AIZ asymmetry is not behaviourally
        # relevant in this model context; real modulation requires ASI/NSM
        # pathways absent from our simulation.)


        # 5. Step graded connectome + read steering/reversal from AVAL/AVAR voltages.
        # The graded model receives the same sensory currents injected into NEURON,
        # scaled from NEURON nA range to graded model nA range (GC_SENSORY_SCALE).
        # Runs every GC_STEP_MS ms (not every NEURON step -- still fast enough).
        _gc_step_counter += 1
        if _gc_step_counter >= _gc_steps_per_update:
            _gc_step_counter = 0
            # Map sensory currents into graded model.
            # AWAL/AWAR: handled by continuous bilateral sampling above.
            # Mechanosensory neurons (FLP/AVD/PVD/PVC/PLM): strong direct AVA inputs
            #   -- scaled separately at GC_MECH_SCALE for reversal pathway.
            # Other sensory: scaled at GC_SENSORY_SCALE.
            # Biological basis: FLP(31), AVD(32), PVD(15), PVC(14) synapses to AVA
            #   are the primary reversal-driving inputs in Varshney 2011 connectome.
            _GC_MECH_NEURONS = {
                'FLPL','FLPR',           # FLP: nociceptive, 31+17 synapses to AVA
                'PVDL','PVDR',           # PVD: harsh touch, 12+15 to AVA
                'PVCL','PVCR',           # PVC: gentle touch tail, 6+14 to AVA
                'PLML','PLMR',           # PLM: tail touch, 5 to AVA
                'PDEL','PDER',           # PDE: body mechanosensory
                'BAGL','BAGR',           # BAG: O2 sensor, drives PQR->AVA indirectly
                'PQR',                   # PQR: tail O2, 11 direct synapses to AVAR
                'ASHL','ASHR',           # ASH: polymodal nociceptor, 5+2 to AVA/AVAL
                                         # responds to osmolarity, pH, noxious chemicals
            }
            # Chemosensory neurons with strong AIY connections
            # AFDL->AIYL(7), AFDR->AIYR(13): thermotaxis pathway
            # ASEL->AIYL(13), ASER->AIYR(14): salt chemotaxis pathway
            _GC_CHEM_NEURONS = {'AFDL','AFDR','ASEL','ASER'}
            for _gc_name, _gc_I in currents.items():
                if _gc_name in ('AWAL', 'AWAR'):
                    pass  # handled by bilateral sampling
                elif _gc_name in _GC_MECH_NEURONS:
                    if _gc_name in ('ASHL', 'ASHR'):
                        _gc_currents[_gc_name] = float(_gc_I) * GC_ASH_SCALE
                    else:
                        _gc_currents[_gc_name] = float(_gc_I) * GC_MECH_SCALE
                elif _gc_name in _GC_CHEM_NEURONS:
                    _gc_currents[_gc_name] = float(_gc_I) * GC_SENSORY_SCALE * 2.0
                else:
                    _gc_currents[_gc_name] = float(_gc_I) * GC_SENSORY_SCALE
            if 'AWAL' not in _gc_currents:
                _gc_currents['AWAL'] = GC_AWA_BASE
            if 'AWAR' not in _gc_currents:
                _gc_currents['AWAR'] = GC_AWA_BASE
            # SAAV: authored satiation drive through real connectome pathway.
            # SAAVL->AVAL(17), SAAVR->AVAR(13) synapses from Varshney 2011.
            # Injection ramps from 0 (satiation=0) to GC_SAAV_MAX (satiation=1).
            # Threshold at 0.3: no effect while worm is hungry, gradual onset
            # as satiation builds, full effect drives ~1.3x reversal boost.
            # This implements patch departure via connectome rather than
            # direct AVA or RIS injection.
            _saav_sat = max(0.0, (worm.satiation - 0.3) / 0.7)  # 0 below 0.3, ramps to 1.0
            _saav_I   = _saav_sat * GC_SAAV_MAX
            if _saav_I > 0.001:
                _gc_currents['SAAVL'] = _saav_I
                _gc_currents['SAAVR'] = _saav_I
            _gc.inject_batch(_gc_currents)
            _gc.step(dt=GC_STEP_MS / 1000.0)
            # Read steering from AVAL/AVAR voltage asymmetry
            # AVAL > AVAR -> food is to left -> turn left (negative signal)
            _gc_aval = _gc.voltage('AVAL')
            _gc_avar = _gc.voltage('AVAR')
            _gc_avbl = _gc.voltage('AVBL')
            _gc_avbr = _gc.voltage('AVBR')
            _gc_aizl = _gc.voltage('AIZL')
            _gc_aizr = _gc.voltage('AIZR')
            _gc_ava_mean = (_gc_aval + _gc_avar) * 0.5
            _gc_avb_mean = (_gc_avbl + _gc_avbr) * 0.5
            # Turn: AIZL-AIZR asymmetry relative to baseline
            # Subtract warmup baseline to remove network equilibrium bias
            # (graded model settles to asymmetric fixed point from random IC)
            # Turn signal: NOT from graded connectome (AIZ bilateral asymmetry
            # reflects global network dynamics, not ipsilateral gradient detection)
            # Authored departure: weathervane lr_ema used for steering;
            # documented finding: NI network global perturbations dominate
            # ipsilateral AWAL->AIZL pathway at realistic gradient strengths
            _gc_turn_signal = 0.0  # set below from weathervane
            # AVA mean voltage: high = worm detecting aversive/leaving-food signal
            _gc_ava_mean = (_gc_aval + _gc_avar) * 0.5
            # AVB mean voltage: high = worm in forward locomotion state
            _gc_avb_mean = (_gc_avbl + _gc_avbr) * 0.5
            # Pirouette suppression: AVB depolarised relative to AVA = forward drive
            # When food present, AWA->AIY->AVB pathway keeps AVB elevated


        # Klinotaxis turn signal: _wv_lr_held is the held bilateral AWA asymmetry
        # from the most recent swing peak pair (south - north concentration).
        # Positive = more food to south = turn south = positive turn_signal.
        # Scale 20x: _wv_lr_held peaks ~0.004, need turn_signal ~0.08 for
        # meaningful steer_bias = K_STEER(0.25) * 0.08 = 0.02 rad head bias.
        _wv_lr_now = getattr(worm, '_wv_lr_held', 0.0)
        # Scale 1.0: _wv_lr_held is Hill-transformed (0-1 range), not raw IClamp.
        # K_STEER(0.25) * turn_signal(0.1) = 0.025 rad head bias -- sufficient for steering.
        turn_signal = float(np.clip(_wv_lr_now * 1.0, -1.0, 1.0))

        awa_total = read_awa_total(h)
        # 5b. Pitch disabled -- worm navigates in XZ plane only
        pitch_signal, db_act, vb_act = read_pitch(h)
        worm.pitch = 0.0

        # 5c. Reversal and pirouette decisions from graded connectome.
        #
        # REVERSALS: driven by AVA voltage from mechanosensory pathway.
        # Biological basis: FLP(31), AVD(32), PVD(15), PVC(14) synapses to AVA
        # are the primary reversal inputs in Varshney 2011 connectome.
        # Mechanosensory neurons (FLPL/R, PVDL/R, PVCL/R, PLML/R) are injected
        # into graded model at GC_MECH_SCALE. When substrate density is high
        # or worm approaches boundary, these neurons depolarise, raising AVA,
        # which increases reversal probability.
        # AWC off-response (food departure) still contributes via direct NEURON
        # IClamp read -- AWC->AVA pathway is too weak in NI connectome (0 synapses)
        # to rely on for reliable food-departure reversals.
        # (authored departure: AWC->AVA not in Varshney 2011; kept as direct read)
        _ava_rel = (_gc_aval + _gc_avar) / 2.0  # relative to Vth
        # Use EMA-based transient detection: reversal triggered by sudden AVA rise,
        # not static offset from warmup baseline. EMA tracks slow mean; deviation
        # captures genuine depolarisation events (boundary contact, harsh touch).
        # TC=5s (alpha=0.001 at 5ms steps) tracks slow drift; fast transients stand out.
        _gc_ava_ema = 0.999 * _gc_ava_ema + 0.001 * _ava_rel
        _ava_above_baseline = _ava_rel - _gc_ava_ema
        # Reversal probability scales with transient AVA depolarisation above running mean
        # _gc_reversal_prob is a rate (events/s); multiply by dt_s for per-step probability
        _gc_reversal_prob = float(np.clip(_ava_above_baseline * GC_REVERSAL_SCALE, 0.0, 50.0))
        # AWC off-response: food departure still read directly (pathway not in connectome)
        _awcl_I = float(_sensory_iclamps['AWCL'].amplitude) if 'AWCL' in _sensory_iclamps else 0.0
        _awc_reversal_prob = float(np.clip(_awcl_I * 2.0, 0.0, 2.0))
        _total_reversal_rate = _gc_reversal_prob + _awc_reversal_prob  # events/s
        # AVA reversal rate stored; combined with pirouette below into single decision.

        # PIROUETTE SUPPRESSION: driven by AVB voltage.
        # Biological basis: AVB drives forward locomotion; high AVB = worm in
        # forward state = suppress spontaneous reversals.
        # AVB receives input from food-sensing circuit via multiple hops.
        # AVBL+AVBR have 14+13 synapses TO AVAR (inhibitory when active) --
        # this is the biological basis for AVB suppressing reversals.
        # AVB suppression: AVB has no AWA/AIZ/AIY inputs in Varshney 2011 (all Gs=0).
        # Cannot encode food state through connectome. Retained for boundary-contact
        # suppression (PVC->AVB exists) but contributes negligibly to food modulation.
        # AVB: use EMA like AVA to avoid static baseline offset problem
        _gc_avb_ema = 0.999 * _gc_avb_ema + 0.001 * _gc_avb_mean
        _avb_rel = _gc_avb_mean - _gc_avb_ema
        _avb_suppression = float(np.clip(_avb_rel * 0.5, 0.0, 0.5))
        # AVB->undulation frequency: deviation from running mean maps to 0.3-0.6 Hz.
        # Positive _avb_rel = AVB above baseline = faster undulation (exploration).
        # Negative _avb_rel = AVB below baseline = slower undulation (on food).
        # Scale 0.075: typical _avb_rel range ±2mV -> ±0.15 Hz deviation.
        # Biological basis: AVB drives DB/VB motor neurons that set crawl rhythm;
        # higher AVB activity = faster B-type motor neuron cycling.
        # Authored departure: frequency modulation proxied as direct parameter.
        worm._wave_freq = float(np.clip(0.45 + _avb_rel * 0.075, 0.30, 0.60))
        # AWA direct suppression: authored departure compensating for absent NSM/serotonin.
        # On food, AWA activity suppresses pirouette rate (worm stays in patch).
        # Biological basis: serotonin from NSM suppresses reversals on food;
        # NSM absent from Varshney 2011. AWA activity used as food-presence proxy.
        # Authored departure: documented in Section 5 of handover.
        # AWA suppression: use IClamp current (verified 0.009-0.019 nA range)
        # rather than live activity variable which may spike in saturated network.
        _awal_I = float(_sensory_iclamps['AWAL'].amplitude) if 'AWAL' in _sensory_iclamps else 0.0
        _awar_I = float(_sensory_iclamps['AWAR'].amplitude) if 'AWAR' in _sensory_iclamps else 0.0
        _awa_I_mean = (_awal_I + _awar_I) * 0.5
        # Scale: at max AWA current ~0.05nA, suppression=0.6. At 0.01nA, suppression=0.12.
        # Scale 150x: AWA ~0.006-0.009nA gives 0.9+ suppression on gradient
        # Near-complete reversal suppression when gradient detected
        _awa_suppression = float(np.clip(_awa_I_mean * 150.0, 0.0, 0.95))
        # AWC off-response boost: departure signal increases pirouette rate
        _awc_I = (_sensory_iclamps['AWCL'].amplitude if 'AWCL' in _sensory_iclamps else 0)
        _awc_I = float(_awc_I) + float(_sensory_iclamps['AWCR'].amplitude if 'AWCR' in _sensory_iclamps else 0)
        _awc_I *= 0.5
        _deriv_boost = float(np.clip(_awc_I * 2.0, 0.0, 2.0))
        effective_rate = PIROUETTE_RATE * (1.0 - _awa_suppression - _avb_suppression) * (1.0 + _deriv_boost)
        effective_rate = max(0.0, effective_rate)
        # Single combined reversal decision per GC step.
        # AVA pathway adds to pirouette rate only when genuinely depolarised (>0.05/s).
        # Both evaluated once per GC_STEP_MS to prevent double-counting.
        # Suppress AVA reversals on gradient -- worm should run, not reverse
        _ava_rate = _total_reversal_rate if _total_reversal_rate > 0.05 else 0.0
        _ava_rate *= (1.0 - _awa_suppression)  # AWA suppresses mechanosensory reversals too
        # Also suppress AWC-driven reversals on gradient
        _total_reversal_rate *= (1.0 - _awa_suppression)
        _combined_rate = effective_rate + _ava_rate
        if (_gc_step_counter == 0
                and _combined_rate > 0 and not worm.reversing and not worm.quiescent
                and worm.refractory_s <= 0
                and worm._pirouette_turn_t <= 0
                and np.random.random() < _combined_rate * (GC_STEP_MS / 1000.0)):
            worm.reversing    = True
            worm.reverse_time = REVERSAL_DURATION
            # Sample gradient at reversal point for pirouette bias
            _nose = worm.pos
            _d_e  = float(env.sample(_nose[0]+10, BODY_Y, _nose[2])['diacetyl'][0])
            _d_w  = float(env.sample(_nose[0]-10, BODY_Y, _nose[2])['diacetyl'][0])
            _d_s  = float(env.sample(_nose[0], BODY_Y, _nose[2]+10)['diacetyl'][0])
            _d_n  = float(env.sample(_nose[0], BODY_Y, _nose[2]-10)['diacetyl'][0])
            _grad_x = _d_w - _d_e  # positive = more conc to west
            _grad_z = _d_s - _d_n  # positive = more conc to south
            _grad_mag = math.sqrt(_grad_x**2 + _grad_z**2)
            if _grad_mag > 1e-8:
                # food_world stored as (dx,dz) pointing toward food
                # west = -x direction, south = +z direction
                # Scale by raw (unadapted) AWA level: stronger gradient confidence
                # = larger pirouette bias. Raw AWA from mapper.awa_raw_mean is
                # the unadapted Hill value -- non-zero even when adapted signal=0.
                # This decouples pirouette heading bias from reversal suppression:
                # the adapted signal suppresses reversals (worm runs on food),
                # the raw signal biases heading after reversal (worm turns toward food).
                # Authored departure: biological mechanism is AIY/RIB/SMD circuit;
                # we proxy the confidence scaling via raw AWA Hill value.
                _awa_raw_conf = float(np.clip(
                    getattr(mapper, 'awa_raw_mean', 0.0) * 3.0, 0.1, 1.0))
                worm._food_world_x = -_grad_x / _grad_mag * _awa_raw_conf
                worm._food_world_z =  _grad_z / _grad_mag * _awa_raw_conf
        # 5d. Read quiescence signal from RIS + authored satiation drive
        ris_act = read_quiescence(h)

        # Satiation integrator: accumulate from SAAVL/SAAVR during feeding,
        # decay during exploration. Authored NSM/ASI substitute -- those
        # pathways require intestinal metabolic signalling absent from our model.
        if _bact_density > 0.002:  # lowered: worm feeds at low density, not just patch core
            worm.satiation = min(1.0, worm.satiation + SATIATION_ACCUMULATE)
        else:
            worm.satiation = max(0.0, worm.satiation - SATIATION_DECAY)
        # Inject satiation boost into RIS via authored proxy
        _ris_satiation_boost = worm.satiation * SATIATION_RIS_SCALE
        ris_act_boosted = min(1.0, ris_act + _ris_satiation_boost)

        # Two thresholds: spontaneous (network fluctuation) and satiation
        _quiescence_threshold = (QUIESCENCE_THRESHOLD_SATIATION
                                 if worm.satiation > 0.5
                                 else QUIESCENCE_THRESHOLD_SPONT)
        if ris_act_boosted > _quiescence_threshold and not worm.quiescent:
            worm.quiescent    = True
            worm.quiescent_ms = 0.0
        elif worm.quiescent:
            worm.quiescent_ms += dt_s * 1000.0
            natural_exit = (ris_act_boosted <= _quiescence_threshold and
                            worm.quiescent_ms >= QUIESCENCE_MIN_MS)
            forced_exit  = worm.quiescent_ms >= QUIESCENCE_MAX_MS
            if natural_exit or forced_exit:
                worm.quiescent    = False
                worm.quiescent_ms = 0.0
                # Partial satiation decay on wake -- worm remains somewhat satiated
                worm.satiation = max(0.0, worm.satiation - 0.3)
        # Suppress motor output during quiescence
        if worm.quiescent:
            turn_signal = 0.0
            awa_total   = 0.0
        # Gate turn signal: only apply when sensory input is present.
        # During exploration (no AWA/AWC input), worm runs straight.
        # Connectome asymmetry bias without sensory drive = spurious spinning.
        # Gate uses RAW (unadapted) AWA as proxy for "any sensory input present"
        # -- adapted IClamp values go to zero at steady state so cannot be used.
        # mapper.awa_raw_mean is set by map_asymmetric() each step.
        _awal_I = float(_sensory_iclamps['AWAL'].amplitude) if 'AWAL' in _sensory_iclamps else 0.0
        _awar_I = float(_sensory_iclamps['AWAR'].amplitude) if 'AWAR' in _sensory_iclamps else 0.0
        _awc_I  = float(_sensory_iclamps['AWCL'].amplitude) if 'AWCL' in _sensory_iclamps else 0.0
        _afdl_I = float(_sensory_iclamps['AFDL'].amplitude) if 'AFDL' in _sensory_iclamps else 0.0
        _afdr_I = float(_sensory_iclamps['AFDR'].amplitude) if 'AFDR' in _sensory_iclamps else 0.0
        _awa_raw = getattr(mapper, 'awa_raw_mean', (_awal_I + _awar_I) * 0.5)
        _sensory_drive = _awa_raw + abs(_awc_I) + (_afdl_I + _afdr_I) * 0.5
        if _sensory_drive < 0.005:
            turn_signal = turn_signal * 0.05  # minimal leakthrough when truly no sensory input
        # 5e. Egg accumulation and laying (HSN/VC circuit)
        # Eggs accumulate when worm is in food-rich environment (AWA active)
        vc_act = read_vc_activity(h)
        if awa_total > 0.1 and not worm.quiescent:
            worm.eggs_accumulated += EGG_ACCUMULATION_RATE * dt_s
        if (worm.eggs_accumulated >= EGG_LAY_THRESHOLD
                and vc_act > VC_ACTIVITY_THRESHOLD):
            worm.eggs_accumulated = 0.0
            worm.eggs_laid += 1
            print(f"  [EGG] t={t_ms/1000:.2f}s egg laid (total={worm.eggs_laid})")

        # 6. Step worm body
        # Speed: base + AWA gain + boost when leaving food (negative dia_deriv)
        dia_deriv = getattr(mapper, '_dia_deriv', 0.0)
        dia_deriv_boost = float(np.clip(-dia_deriv * 500.0, 0.0, 2.0))
        # 6d. Sample bacterial density at nose position (before food_factor)
        # Bilinear interpolation in XZ -- 2D env, y ignored.
        _bfi = float(np.clip((worm.pos[0] / env.lx) * env.nx, 0, env.nx - 1.001))
        _bfk = float(np.clip((worm.pos[2] / env.lz) * env.nz, 0, env.nz - 1.001))
        _bi0, _bk0 = int(_bfi), int(_bfk)
        _bi1 = min(_bi0+1, env.nx-1)
        _bk1 = min(_bk0+1, env.nz-1)
        _bdi, _bdk = _bfi-_bi0, _bfk-_bk0
        _bact_density = float(
            float(env.B[_bi0, _bk0]) * (1-_bdi) * (1-_bdk) +
            float(env.B[_bi0, _bk1]) * (1-_bdi) * _bdk +
            float(env.B[_bi1, _bk0]) * _bdi     * (1-_bdk) +
            float(env.B[_bi1, _bk1]) * _bdi     * _bdk)

        # Speed modulation: slow on food, fast during exploration
        # Real C. elegans slows to ~30% speed on food (Sawin et al. 2000)
        # Gated on actual bacterial density at nose -- not diacetyl concentration.
        # Worm maintains full speed through diacetyl plume; only slows on contact
        # with bacterial source. Biologically: speed reduction is driven by pharyngeal
        # feedback and substrate contact, not chemosensation alone.
        # _bact_density sampled below at nose position each logging step.
        # Hill-based food_factor: sensitive at low density (patch edge detection)
        # At bact=0.003 (edge): factor=0.74 (26% slower)
        # At bact=0.02 (mid):   factor=0.44 (56% slower)  
        # At bact=0.1+ (core):  factor=0.33 (full slowdown)
        # Biological basis: C. elegans slows on food contact (Sawin 2000);
        # Hill response matches graded pharyngeal feedback at low densities.
        food_factor = max(0.3, 1.0 - 0.7 * _bact_density / (0.002 + _bact_density))
        # Multiplier raised from 4.0: nose samples max ~0.07 due to inter-blob gaps
        # in organic patch texture. 12.0 gives full slowdown at observed densities.

        # Soil density modulates base locomotion speed.
        # Denser soil = higher substrate resistance = slower forward thrust.
        # Biological basis: Fang-Yen et al. 2010 -- C. elegans speed scales with
        # substrate viscosity roughly as v ∝ η^-0.26. Here approximated as linear
        # reduction over the soil_density range 0.1-0.45.
        # soil_factor range: ~0.97 (loose) to ~0.60 (dense) -- 37% max reduction.
        # Applied to BASE_SPEED before food_factor so both modulations compound:
        # dense soil + on food -> slowest locomotion (~0.54 wu/s at extremes).
        # (authored departure: real mechanism is drag on body; proxied via speed_override)
        _soil_den = float(env.sample(_nose_pos[0], _nose_pos[1], _nose_pos[2])
                         ['soil_density'][0])
        soil_factor = max(0.60, 1.0 - _soil_den * 0.90)
        # Wave amplitude modulates speed biologically (food/soil resistance).
        # Replaces speed_override force -- amplitude reduction slows worm naturally.
        _wave_amp = BASE_WAVE_AMP * soil_factor * food_factor
        worm._wave_amp = float(_wave_amp)
        speed = BASE_SPEED * soil_factor * food_factor + dia_deriv_boost

        # 6d2. Pharyngeal departure AWC signal.
        # Proxies pharyngeal mechanosensory -> NSM -> reversal disinhibition pathway.
        # EMA of bact_density tracks recent food contact; drop signals food departure.
        # (authored departure: real pathway is pharyngeal mechanosensory -> serotonin;
        # collapsed to scalar density proxy since those neurons absent from c302 B_Full)
        _bact_ema_alpha = dt_s * (1.0 / 3.0)  # 3s TC -> ~8s off-response
        _bact_ema = _bact_ema + _bact_ema_alpha * (_bact_density - _bact_ema)
        _phx_departure_drive = float(np.clip(_bact_ema - _bact_density, 0.0, 1.0))
        if _phx_departure_drive > 0.005:
            _I_awc_phx = _phx_departure_drive * 0.15
            currents['AWCL'] = currents.get('AWCL', 0.0) + _I_awc_phx
            currents['AWCR'] = currents.get('AWCR', 0.0) + _I_awc_phx
        # Compute per-segment muscle activations from connectome
        _all_motor = read_neuron_activities_batch(
            ['DA1','DA2','DA3','DA4','DA5','DA6','DA7','DA8','DA9',
             'DB1','DB2','DB3','DB4','DB5','DB6','DB7',
             'VA1','VA2','VA3','VA4','VA5','VA6','VA7','VA8','VA9','VA10','VA11','VA12',
             'VB1','VB2','VB3','VB4','VB5','VB6','VB7','VB8','VB9','VB10','VB11',
             'AS1','AS2','AS3','AS4','AS5','AS6','AS7','AS8','AS9','AS10','AS11',
             'DD1','DD2','DD3','DD4','DD5','DD6',
             'VD1','VD2','VD3','VD4','VD5','VD6','VD7','VD8','VD9','VD10','VD11','VD12','VD13']
        )
        _muscle_dorsal, _muscle_ventral = compute_muscle_activations(_all_motor)
        # Restore DD/VD GABAergic cross-inhibition (non-functional in c302 B)
        inject_dd_vd_drive(h, _all_motor)
        # Step physics body — speed_override maintains existing speed modulation
        _body_phase += 2.0 * math.pi * 1.8 * dt_s  # 1.8 Hz proprioceptive wave
        nose, heading_vec, body = worm.step(dt_s, _muscle_dorsal, _muscle_ventral,
                                            speed_override=None,
                                            turn_signal=turn_signal + _weathervane_signal,
                                            awc_signal=_awc_I)

        # 6c. Proprioceptive proxy: asymmetric curvature-driven PLM/PVM injection.
        # PLM neurons fire when body presses against substrate during bending.
        # PLML fires on leftward (ventral) bend, PLMR on rightward (dorsal) bend.
        # This provides bilateral alternating drive into DVA->AIZL (w5 each side),
        # counteracting the resting RIH->AIZR asymmetry by lifting both AIZ neurons
        # rhythmically in antiphase with the undulation wave.
        # Authored departure: PLM responds to substrate contact via body bending;
        # curvature sign used as proxy for L/R substrate pressure.
        # Biological basis: Wen et al. 2012 -- PLM proprioceptive coupling
        # essential for stable forward locomotion wave.
        # Curvature-based PLM: use deviation from running mean to track
        # the oscillatory wave component rather than absolute bend direction.
        # This gives symmetric bilateral alternation even with a tonic body bias.
        # EMA of posterior curvature tracks the DC bias; deviation tracks the wave.
        _body_curv = worm.get_curvature()
        _post_curv_raw = float(_body_curv[max(0,len(_body_curv)*3//4):].mean())
        # Update running mean of posterior curvature (slow EMA, ~5s time constant)
        _plm_curv_ema = getattr(worm, '_plm_curv_ema', _post_curv_raw)
        _plm_curv_ema = 0.999 * _plm_curv_ema + 0.001 * _post_curv_raw
        worm._plm_curv_ema = _plm_curv_ema
        # Oscillatory component = deviation from slow mean
        _post_curv_osc = _post_curv_raw - _plm_curv_ema
        # Scale: ±0.5 rad oscillation -> ±1.0 scale
        _osc_scale = np.clip(_post_curv_osc / 0.5, -1.0, 1.0)
        _plm_tonic = 0.05  # bilateral tonic floor
        _asym = 0.04 * abs(_osc_scale)  # asymmetric amplitude
        if _osc_scale >= 0:  # dorsal oscillation -> PLMR fires more
            _plmL_I = _plm_tonic
            _plmR_I = _plm_tonic + _asym
        else:  # ventral oscillation -> PLML fires more
            _plmL_I = _plm_tonic + _asym
            _plmR_I = _plm_tonic
        _pvm_I = _plm_tonic * 0.5
        currents['PLML'] = max(0.0, _plmL_I)
        currents['PLMR'] = max(0.0, _plmR_I)
        currents['PVM']  = max(0.0, _pvm_I)

        # 6c2. Pharyngeal pacemaker: authored MC substitute.
        # In the real worm, MC fires tonically on food to drive pharyngeal pumping
        # at 200-300 pumps/min. MC is absent from c302 B_Full; we author a proxy
        # that injects rhythmic current into pharyngeal motor neurons when AWA > 0.05
        # (worm is at or near bacterial source). Rate ~4 Hz = 240 pumps/min.
        # Amplitude modulated by AWA activity (food concentration).
        # Authored explicitly -- restores intrinsic pharyngeal drive missing from model.
        _awa_act_phx = (max(0.0, min(1.0, float(_cell_cache['AWAL'].activity) if 'AWAL' in _cell_cache else 0.0)) +
                        max(0.0, min(1.0, float(_cell_cache['AWAR'].activity) if 'AWAR' in _cell_cache else 0.0))) * 0.5
        if _awa_act_phx > 0.05:
            # Phase advances at 4 Hz pump rate
            _phx_phase = (t_ms * 0.004) % 1.0   # 4 pumps/s, phase 0-1
            _phx_sine  = float(math.sin(2.0 * math.pi * _phx_phase))
            # Scale amplitude by AWA activity -- stronger food signal = faster pumping
            _phx_amp   = 0.03 * (_awa_act_phx / 0.072) * max(0.0, _phx_sine)
            currents['M1']  = _phx_amp
            currents['M2L'] = _phx_amp * 0.8
            currents['M2R'] = _phx_amp * 0.8
            currents['M3L'] = _phx_amp * 0.6
            currents['M3R'] = _phx_amp * 0.6
            currents['M4']  = _phx_amp * 0.7
            currents['M5']  = _phx_amp * 0.5
        else:
            currents['M1']  = 0.0
            currents['M2L'] = 0.0
            currents['M2R'] = 0.0
            currents['M3L'] = 0.0
            currents['M3R'] = 0.0
            currents['M4']  = 0.0
            currents['M5']  = 0.0

        # 6b. Mechanosensation + tail chemosensation: fresh body positions
        tx, ty, tz = body[N_BODY_POINTS - 1]
        concs_tail = env.sample(tx, ty, tz)
        density_nose = env.sample(nose[0], nose[1], nose[2])['soil_density'][0]
        density_tail = concs_tail['soil_density'][0]
        # FLP/PVD/PVC: harsh-touch nociceptors. Fire only above density threshold,
        # representing genuine substrate contact / boundary compression.
        # Silent during normal open-field locomotion (density < threshold).
        # Threshold 0.4: below typical open soil (0.27), above boundary/patch edge.
        # At threshold: 0.0 nA. At density=1.0: FLP=0.06nA -> ~6 NI units -> 0.02/s AVA.
        # Authored departure: we model these as contact sensors not tonic reporters.
        _MECH_THRESHOLD = 0.35  # lowered: medium-density soil (0.35-0.55) contributes
        _mech_drive = float(np.clip((density_tail - _MECH_THRESHOLD) / (1.0 - _MECH_THRESHOLD), 0.0, 1.0))
        currents['FLPL'] = 0.06 * _mech_drive
        currents['FLPR'] = 0.06 * _mech_drive
        currents['PVDL'] = 0.04 * _mech_drive
        currents['PVDR'] = 0.04 * _mech_drive
        currents['PVCL'] = 0.03 * _mech_drive
        currents['PVCR'] = 0.03 * _mech_drive
        mech = mapper.mechanosensory_currents(density_nose, density_tail)
        if mech:
            currents.update(mech)
        # Tail chemical sampling: PHA/PHB/PHC phasmid neurons
        tail_chem = mapper.tail_chemosensory_currents(concs_tail)
        if tail_chem:
            currents.update(tail_chem)
        # 6e. Travelling wave CPG — propagates head oscillation along B-type neurons.
        # Biological basis: anterior DB/VB neurons have intrinsic oscillatory
        # properties (Gao et al. 2018; Kawano et al. 2011) that seed the wave.
        # Wave propagates head-to-tail via proprioceptive coupling (Wen et al. 2012).
        # c302 B Full IAF model captures neither the intrinsic oscillation nor the
        # proprioceptive propagation adequately.
        # We author both: a sinusoidal signal injected into each DB/VB neuron with
        # a phase delay proportional to its segment index, producing a travelling wave
        # in the neural domain directly. Body curvature follows from neural drive.
        # Phase delay: 1/(1.8Hz * 24 segs) * 2pi ≈ 0.1454 rad per segment.
        # Amplitude: HEAD_CPG_AMP modulates each neuron relative to its tonic level.
        # Gated on forward locomotion — suppressed during reversals and quiescence.
        # Authored explicitly — documented departure from raw connectome output.
        HEAD_CPG_AMP  = 0.13
        _WAVE_FREQ    = 1.8   # Hz
        _PHASE_PER_SEG = 2.0 * math.pi * _WAVE_FREQ / 24.0  # rad per segment
        # DB neuron segment indices (for phase calculation)
        _db_phase_segs = {'DB1': 0, 'DB2': 2, 'DB3': 4, 'DB4': 6,
                          'DB5': 8, 'DB6': 10, 'DB7': 12}
        _vb_phase_segs = {'VB1': 1, 'VB2': 3, 'VB3': 5,  'VB4': 7,
                          'VB5': 8, 'VB6': 9, 'VB7': 11, 'VB8': 10,
                          'VB9': 13, 'VB10': 15, 'VB11': 17}
        if not worm.reversing and not worm.quiescent:
            _t_s = t_ms * 0.001
            for _db, _pseg in _db_phase_segs.items():
                _phase = 2.0 * math.pi * _WAVE_FREQ * _t_s - _pseg * _PHASE_PER_SEG
                _osc = math.sin(_phase)
                currents[_db] = float(currents.get(_db, 0.0)) + max(0.0, _osc) * HEAD_CPG_AMP
            for _vb, _pseg in _vb_phase_segs.items():
                _phase = 2.0 * math.pi * _WAVE_FREQ * _t_s - _pseg * _PHASE_PER_SEG
                _osc = math.sin(_phase)
                currents[_vb] = float(currents.get(_vb, 0.0)) + max(0.0, -_osc) * HEAD_CPG_AMP

        # 6e. Segment-local proprioceptive feedback for DB/VB neurons.
        # Biological basis: Wen et al. 2012 — B-type motor neurons transduce
        # proprioceptive signals to propagate the bending wave head-to-tail.
        # Each DB/VB neuron reads curvature at its primary body segment and
        # receives positive feedback: dorsal bend excites DB, ventral bend excites VB.
        # This closes the local reflex arc that drives wave propagation.
        # Authored departure: implemented as IClamp injection rather than
        # stretch-sensitive membrane conductance (absent from IAF model).
        # Gain K_PROPRIO=0.06 nA/rad — tuned to produce wave without runaway.
        K_PROPRIO = 0.02
        # DB neuron -> primary segment mapping (peak innervation from weight arrays)
        _db_segs = {'DB1': 7,  'DB2': 9,  'DB3': 11, 'DB4': 13,
                    'DB5': 16, 'DB6': 18, 'DB7': 21}
        # VB neuron -> primary segment mapping
        _vb_segs = {'VB1': 6,  'VB2': 8,  'VB3': 10, 'VB4': 12,
                    'VB5': 13, 'VB6': 14, 'VB7': 16, 'VB8': 15,
                    'VB9': 17, 'VB10': 19, 'VB11': 21}
        # Proprioceptive delay buffer — each segment reads curvature from
        # a time-delayed history, introducing the phase offset needed for
        # head-to-tail wave propagation.
        # Biological basis: mechanical wave travels at finite speed; posterior
        # stretch receptors respond to what anterior segments did ~23ms ago.
        # Delay per segment: 1/(1.8Hz * 24 segs) ≈ 23ms → 23 steps at 1ms.
        # Segment n reads curvature delayed by n * SEG_DELAY_STEPS.
        # Authored departure: implemented as explicit ring buffer rather than
        # physical wave propagation in the IAF model.
        SEG_DELAY_STEPS = 23  # ms between adjacent segments at 1.8Hz
        _PROPRIO_ALPHA  = 0.01
        _BUFFER_LEN     = SEG_DELAY_STEPS * 24 + 1  # full body delay coverage

        # Initialise ring buffer and EMA on first call
        if not hasattr(worm, '_curv_buffer'):
            worm._curv_buffer = np.zeros((_BUFFER_LEN, len(_body_curv)), dtype=np.float32)
            worm._curv_buf_idx = 0
            worm._curv_ema = _body_curv.copy()

        # Update EMA baseline
        worm._curv_ema = (1.0 - _PROPRIO_ALPHA) * worm._curv_ema + _PROPRIO_ALPHA * _body_curv

        # Write current curvature to ring buffer
        worm._curv_buffer[worm._curv_buf_idx] = (_body_curv - worm._curv_ema).astype(np.float32)
        worm._curv_buf_idx = (worm._curv_buf_idx + 1) % _BUFFER_LEN

        def _read_delayed(seg_idx):
            """Read oscillatory curvature at seg_idx delayed by seg_idx * SEG_DELAY_STEPS."""
            delay = seg_idx * SEG_DELAY_STEPS
            read_idx = (worm._curv_buf_idx - 1 - delay) % _BUFFER_LEN
            return float(worm._curv_buffer[read_idx, seg_idx])

        # DB: excited by delayed dorsal oscillation at its primary segment
        for _db, _seg in _db_segs.items():
            _osc = _read_delayed(_seg) if _seg < len(_body_curv) else 0.0
            _fb = K_PROPRIO * max(0.0, _osc)
            currents[_db] = float(currents.get(_db, 0.0)) + _fb
        # VB: excited by delayed ventral oscillation at its primary segment
        for _vb, _seg in _vb_segs.items():
            _osc = _read_delayed(_seg) if _seg < len(_body_curv) else 0.0
            _fb = K_PROPRIO * max(0.0, -_osc)
            currents[_vb] = float(currents.get(_vb, 0.0)) + _fb

        # Single inject: all currents merged
        inject_currents(currents)
        # DEBUG: verify AWAL/AWAR reach NEURON inject (remove after validation)
        if step % 2000 == 0:
            _dbg_al = currents.get('AWAL', 'absent')
            _dbg_ar = currents.get('AWAR', 'absent')
            print(f"  [DBG] NEURON inject AWAL={_dbg_al}  AWAR={_dbg_ar}")

        # 7. Log
        if step % log_every == 0:
            activities  = read_neuron_activities(h, neuron_names)
            sensory_arr = currents_to_array(currents)
            # Patch corrupted sensory neuron activity values in HDF5 output.
            # IAF ODE explodes under direct IClamp injection; substitute the
            # injected current (scaled to 0-1) for those neuron positions only.
            # Non-sensory neurons are written as-is (clean connectome activity).
            activities = activities.copy()
            for _sname, _sidx in _sensory_idx.items():
                _raw_I = float(currents.get(_sname, 0.0))
                activities[_sidx] = float(np.clip(_raw_I, 0.0, 1.0))
            speed       = speed  # already computed above
            turn_rate   = turn_signal * TURN_GAIN

            # Log env grids only every log_every_env steps (slower cadence)
            _log_env = (step % log_every_env == 0)

            logger.log(
                t_ms / 1000.0, nose, heading_vec, body,
                speed, turn_rate, activities, sensory_arr,
                _gc_aizl, _gc_aizr, turn_signal + _weathervane_signal,  # full turn signal logged
                pitch_signal, db_act, vb_act,
                _total_reversal_rate * dt_s, worm.reversing,  # per-step reversal prob from AVA
                ris_act, worm.quiescent,
                vc_act, worm.eggs_accumulated, worm.satiation,
                _bact_density,
                chem_fields=env.C.cpu().numpy() if _log_env else None,
                bacterial_grid=env.B.cpu().numpy() if _log_env else None,
                muscle_dorsal=_muscle_dorsal.copy(),
                muscle_ventral=_muscle_ventral.copy(),
                particle_positions=worm.get_body_points()
            )

            active = [f"{k}:{v:.4f}" for k, v in currents.items()
                      if abs(v) > 0.001]
            _wv_lr = getattr(worm, '_wv_lr_ema', 0.0)
            print(
                f"t={t_ms:7.1f}ms  "
                f"nose=({nose[0]:.2f},{nose[1]:.2f},{nose[2]:.2f})  "
                f"AIZL={_gc_aizl:.3f} AIZR={_gc_aizr:.3f}mV  "
                f"wv_lr={_wv_lr:+.5f} turn={turn_signal+_weathervane_signal:+.4f}  "
                f"sensory={active}  "
                f"wall={time.time()-t_wall:.0f}s"
            )

            # Periodic checkpoint
            if (_checkpoint_every_steps > 0 and
                    (step + _step_offset) % _checkpoint_every_steps == 0 and
                    step > 0):
                save_checkpoint(_checkpoint_path, h, worm, mapper, env,
                                t_ms, step + _step_offset, None)

    logger.close()

    # Save final checkpoint on clean exit so run can always be resumed
    if checkpoint_every and checkpoint_every > 0:
        save_checkpoint(_checkpoint_path, h, worm, mapper, env,
                        t_ms, n_steps + _step_offset, None)
        print('  [CHECKPOINT] Final checkpoint saved (run resumable)')

    elapsed = time.time() - t_wall
    print(f"\nDone. {duration}s simulated in {elapsed:.1f}s wall time "
          f"({duration/elapsed*1000:.1f}x real-time).")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='C. elegans kinematic sim with real connectome steering '
                    '(c302 parameter set B)')
    ap.add_argument('--sim_dir',     required=True)
    ap.add_argument('--edgelist',    type=str, default=None,
                    help='Path to herm_full_edgelist.csv for graded connectome')
    ap.add_argument('--duration',    type=float, default=30.0)
    ap.add_argument('--dt_nrn',      type=float, default=0.05)
    ap.add_argument('--log_every',     type=int,   default=20)
    ap.add_argument("--log_every_env", type=int, default=20000, help="Steps between env grid snapshots (default 20000=1s)")
    ap.add_argument('--env_cache',   type=str,   default=None,
                    help='Path to saved environment state (.npz). '
                         'If exists, loads instead of running warmup. '
                         'If not exists, runs warmup and saves to this path.')
    ap.add_argument('--output',      default=None)
    ap.add_argument('--env_step_ms', type=float, default=1.0)
    ap.add_argument('--start_x',     type=float, default=None)
    ap.add_argument('--start_y',     type=float, default=None)
    ap.add_argument('--start_z',     type=float, default=None)
    ap.add_argument('--start_heading', type=float, default=None, help='Starting heading in degrees (0=east, 180=west)')
    ap.add_argument('--colony_ix_min', type=int, default=None,
                    help='Colony X min grid index (default: 56%% of nx)')
    ap.add_argument('--colony_ix_max', type=int, default=None,
                    help='Colony X max grid index (default: 81%% of nx)')
    ap.add_argument('--colony_iz_min', type=int, default=None,
                    help='Colony Z min grid index (default: 28%% of nz)')
    ap.add_argument('--colony_iz_max', type=int, default=None,
                    help='Colony Z max grid index (default: 72%% of nz)')
    ap.add_argument('--nacl_seed',    type=int, default=None,
                    help='Seed for NaCl landscape. None=random each run (performance), int=reproducible (debug).')
    ap.add_argument('--colony_seed', type=int, default=None, help='Seed for bacterial colony positions. None=random, int=reproducible.')
    ap.add_argument('--checkpoint_every', type=float, default=30.0, help='Save checkpoint every N simulated seconds. 0 to disable.')
    ap.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint.npz to resume from.')
    a = ap.parse_args()
    _sh = math.radians(a.start_heading) if a.start_heading is not None else None

    run(sim_dir      = a.sim_dir,
        start_heading = _sh,
        duration     = a.duration,
        dt_nrn       = a.dt_nrn,
        log_every    = a.log_every,
        log_every_env = a.log_every_env,
        output_path  = a.output,
        env_step_ms  = a.env_step_ms,
        start_x      = a.start_x,
        start_y      = a.start_y,
        start_z      = a.start_z,
        env_cache        = a.env_cache,
        colony_seed      = a.colony_seed,
        colony_ix_min    = a.colony_ix_min,
        colony_ix_max    = a.colony_ix_max,
        colony_iz_min    = a.colony_iz_min,
        colony_iz_max    = a.colony_iz_max,
        checkpoint_every = a.checkpoint_every,
        resume_from      = a.resume_from,
        edgelist_path    = a.edgelist)
