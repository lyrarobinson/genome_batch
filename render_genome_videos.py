#!/usr/bin/env python3
"""
render_genome_videos.py
-----------------------
Headless batch renderer for SOIL genome simulation runs.
Produces MP4 videos with the same three-panel layout as visualise_dashboard.py:

    LEFT  (60%): vispy 3D — worm body + environment (offscreen OpenGL)
    TOP-RIGHT (40%): matplotlib — 302-neuron connectome activation
    BOT-RIGHT (40%): matplotlib — kymograph

Standalone — does NOT import or depend on visualise_dashboard.py.
All rendering code is self-contained.

Usage:
    conda activate worm
    cd ~/projects/worm/simulate/sibernetic

    # Render all completed runs from genome_batch output:
    python3 render_genome_videos.py

    # Render a specific HDF5:
    python3 render_genome_videos.py path/to/kinematic_sim.h5

    # Resume: completed videos are skipped automatically.
    # Stop at any time with Ctrl+C — partial MP4 is discarded cleanly.

Configuration is at the top of this file.
"""

import os, sys, glob, math, time, csv, traceback, signal, argparse
import numpy as np
import h5py
import torch
import xml.etree.ElementTree as ET
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, gaussian_filter
from scipy.interpolate import CubicSpline

# ── Headless OpenGL for vispy ─────────────────────────────────────────────────
# Must be set BEFORE importing vispy
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('VISPY_BACKEND',     'egl')   # vispy >= 0.13

import vispy
vispy.use('egl')
from vispy import scene
from vispy.scene import visuals
from vispy.app import use_app
use_app('egl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── Output encoding ───────────────────────────────────────────────────────────
import cv2

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Paths
SIM_ROOT   = os.path.expanduser("~/projects/worm/simulate/sibernetic")
NML_PATH   = os.path.expanduser("~/projects/worm/c302/examples/c302_B_Full.net.nml")
GENOME_DIR = os.path.join(SIM_ROOT, "genome_runs")         # where genome_batch wrote outputs
VIDEO_DIR  = os.path.join(SIM_ROOT, "genome_videos")       # where we write MP4s
COMPLETED_CSV = os.path.join(SIM_ROOT, "render_completed.csv")
GENOME_BATCH_CSV = os.path.join(SIM_ROOT, "completed.csv") # from genome_batch.py

# Video
OUTPUT_FPS     = 30       # MP4 frames per second
SIM_FPS        = 20       # how many simulation body frames per rendered frame (downsampling)
                           # 1 = every body frame; higher = faster, smaller file
VIDEO_WIDTH    = 1920
VIDEO_HEIGHT   = 1080
VIDEO_QUALITY  = 22       # CRF: lower = better quality / larger file

# Layout ratios (must sum to 1.0)
LEFT_FRAC  = 0.60   # vispy panel
RIGHT_FRAC = 0.40   # matplotlib panels (connectome top, kymo bottom)
CONN_FRAC  = 0.70   # fraction of right panel for connectome
KYMO_FRAC  = 0.30   # fraction of right panel for kymograph

# World
LX, LZ = 960.0, 540.0

# Worm rendering
N_SPLINE  = 150
TUBE_PTS  = 24

# Colours (match dashboard)
BG_RGB = (2, 1, 9)   # #020109 in uint8

FIELD_NAMES = ['diacetyl','benzaldehyde','butanone','isoamyl_alc',
               'nonanone','octanol','noxious','nacl','osmolarity',
               'ph','ascarosides','oxygen','co2','temperature','soil_density']

FIELD_COL = {
    'diacetyl':    (1.00, 0.55, 0.10),
    'nacl':        (0.20, 0.75, 0.90),
    'oxygen':      (0.15, 0.50, 1.00),
    'co2':         (0.65, 0.25, 0.85),
    'temperature': (1.00, 0.20, 0.10),
    'soil_density':(0.60, 0.42, 0.22),
    'bacteria':    (0.15, 0.95, 0.35),
}
DEF_COL = (0.80, 0.60, 0.20)
BG_HEX = "#020109"

COLOR_CLASSES = [
    ("#04161E","#4FC3F7",["AWA","AWB","AWC","ASH","ASE","ASI","ASJ","ASK","AFD","BAG",
                           "URX","URB","URA","ADF","ADL","ALM","AVM","PLM","PVM",
                           "CEP","IL1","IL2","OLL","OLQ","ADE","PDE","PHA","PHB","PHC","PQR","ADA"]),
    ("#041A0C","#69D98C",["AIY","AIZ","AIB","AIA","AIE","AIM","AIN",
                           "RIA","RIB","RIG","RIM","RIS","RIF","RIC","RIH",
                           "AVA","AVB","AVD","AVE","AVF","AVG","AVH","AVJ","AVK","AVL",
                           "DVA","DVC","PVC","PVN","PVP","PVQ","PVR","LUA","PVD","PLN","CAN","SDQ"]),
    ("#1A0D00","#FFB74D",["DA","DB","VA","VB","AS"]),
    ("#1A0404","#FF6B6B",["DD","VD"]),
    ("#0E0418","#CE93D8",["RMD","SMD","SMB","SIA","SIB","RIV","SAA","SAB","RMF","RMG","RME","RMH"]),
    ("#1A0410","#F48FB1",["VC","HSN"]),
    ("#1A1A00","#FFF176",["I1","I2","I3","I4","I5","I6","M1","M2","M3","M4","M5","MCL","MCR","MI","NSM"]),
    ("#0C0C14","#8899CC",[]),
]

FIELD_STYLE = {
    'diacetyl':    dict(alpha_scale=0.30, alpha_max=0.55, size=13, sigma=1.5, y_range=2.0, y_bias=0.0),
    'butanone':    dict(alpha_scale=0.25, alpha_max=0.50, size=12, sigma=1.5, y_range=2.0, y_bias=0.0),
    'benzaldehyde':dict(alpha_scale=0.22, alpha_max=0.45, size=11, sigma=1.8, y_range=2.0, y_bias=0.0),
    'isoamyl_alc': dict(alpha_scale=0.20, alpha_max=0.42, size=11, sigma=1.8, y_range=2.0, y_bias=0.0),
    'noxious':     dict(alpha_scale=0.18, alpha_max=0.38, size=7,  sigma=1.2, y_range=1.5, y_bias=0.0),
    'ascarosides': dict(alpha_scale=0.15, alpha_max=0.32, size=7,  sigma=1.2, y_range=1.5, y_bias=0.0),
    'nacl':        dict(alpha_scale=0.10, alpha_max=0.20, size=6,  sigma=4.0, y_range=3.0, y_bias=0.0),
    'osmolarity':  dict(alpha_scale=0.08, alpha_max=0.16, size=5,  sigma=4.5, y_range=3.0, y_bias=0.0),
    'ph':          dict(alpha_scale=0.08, alpha_max=0.16, size=5,  sigma=4.0, y_range=3.0, y_bias=0.0),
    'oxygen':      dict(alpha_scale=0.06, alpha_max=0.12, size=18, sigma=8.0, y_range=4.0, y_bias=0.5),
    'co2':         dict(alpha_scale=0.06, alpha_max=0.12, size=18, sigma=8.0, y_range=4.0, y_bias=0.5),
    'soil_density':dict(alpha_scale=0.0,  alpha_max=0.0,  size=1,  sigma=1.0, y_range=1.0, y_bias=0.0),
    'temperature': dict(alpha_scale=0.0,  alpha_max=0.0,  size=1,  sigma=1.0, y_range=1.0, y_bias=0.0),
}
_DEFAULT_STYLE = dict(alpha_scale=0.10, alpha_max=0.20, size=8, sigma=3.0, y_range=2.5, y_bias=0.0)

_FIELD_RGB = np.array([FIELD_COL.get(n, DEF_COL) for n in FIELD_NAMES], dtype=np.float32)
N_CHEM_PTS = 500
N_BACT_PTS = 600

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPLICIT = [
    ("CEPDL",0.30,0.97),("CEPDR",0.70,0.97),("CEPVL",0.33,0.95),("CEPVR",0.67,0.95),
    ("IL1DL",0.27,0.98),("IL1DR",0.73,0.98),("IL1L",0.37,0.99),("IL1R",0.63,0.99),
    ("AWAL",0.22,0.88),("AWAR",0.78,0.88),("AWBL",0.24,0.86),("AWBR",0.76,0.86),
    ("AWCL",0.26,0.87),("AWCR",0.74,0.87),("ASHL",0.17,0.85),("ASHR",0.83,0.85),
    ("ASEL",0.19,0.83),("ASER",0.81,0.83),("ASJL",0.12,0.84),("ASJR",0.88,0.84),
    ("ASKL",0.10,0.86),("ASKR",0.90,0.86),("BAGL",0.08,0.88),("BAGR",0.92,0.88),
    ("URXL",0.06,0.90),("URXR",0.94,0.90),("ADEL",0.29,0.82),("ADER",0.71,0.82),
    ("ALML",0.14,0.78),("ALMR",0.86,0.78),("AVM",0.50,0.76),("AFDL",0.15,0.90),("AFDR",0.85,0.90),
    ("AIYL",0.28,0.77),("AIYR",0.72,0.77),("AIZL",0.31,0.75),("AIZR",0.69,0.75),
    ("AIBL",0.25,0.73),("AIBR",0.75,0.73),("AIAL",0.23,0.74),("AIAR",0.77,0.74),
    ("RIAL",0.33,0.79),("RIAR",0.67,0.79),("RIBL",0.35,0.78),("RIBR",0.65,0.78),
    ("RIS",0.50,0.73),("RICL",0.38,0.71),("RICR",0.62,0.71),
    ("RMDL",0.24,0.80),("RMDR",0.76,0.80),("SMDL",0.20,0.77),("SMDR",0.80,0.77),
    ("SIAL",0.16,0.74),("SIAR",0.84,0.74),("RIVL",0.21,0.73),("RIVR",0.79,0.73),
    ("AVAL",0.20,0.62),("AVAR",0.80,0.62),("AVBL",0.22,0.60),("AVBR",0.78,0.60),
    ("AVDL",0.24,0.58),("AVDR",0.76,0.58),("AVEL",0.26,0.56),("AVER",0.74,0.56),
    ("AVHL",0.33,0.50),("AVHR",0.67,0.50),("AVL",0.50,0.44),("DVA",0.50,0.42),
    ("PVCL",0.44,0.30),("PVCR",0.56,0.30),
    ("HSNL",0.22,0.28),("HSNR",0.78,0.28),
    ("PLML",0.18,0.14),("PLMR",0.82,0.14),("PVM",0.50,0.13),
    ("PDEL",0.23,0.12),("PDER",0.77,0.12),
    ("PHCL",0.23,0.07),("PHCR",0.77,0.07),("PQR",0.50,0.065),
    ("SAAVL",0.19,0.67),("SAAVR",0.81,0.67),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to dashboard)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(h5_path):
    print(f"  Loading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        times     = f['environment/times'][:]
        body_all  = f['worm/body_points'][:]
        muscle_d  = f['body/muscle_dorsal'][:]
        muscle_v  = f['body/muscle_ventral'][:]
        reversing = f['steering/reversing'][:]
        quiescent = f['steering/quiescent'][:]
        aiyl      = f['steering/aiyl'][:]
        aiyr      = f['steering/aiyr'][:]
        total_env = f['environment/chem_fields'].shape[0]
        chem_env  = f['environment/chem_fields'][:]
        bact_env  = f['environment/bacterial_grid'][:]
        has_act   = 'worm/neuron_activity' in f
        if has_act:
            neuron_names = list(f['worm/neuron_names'][:].astype(str))
            activity_raw = f['worm/neuron_activity'][:]
        else:
            neuron_names = []
            activity_raw = None

    T = len(times)
    env_times = np.linspace(times[0], times[-1], total_env)
    print(f"  {T} body frames, {total_env} env snapshots")
    NX, NZ = bact_env.shape[1], bact_env.shape[2]
    gpu_particles = _build_gpu_particles(NX, NZ)
    env_clouds    = _build_env_clouds_gpu(chem_env, bact_env, gpu_particles)
    return dict(
        times=times, body_all=body_all, muscle_d=muscle_d, muscle_v=muscle_v,
        reversing=reversing, quiescent=quiescent, aiyl=aiyl, aiyr=aiyr,
        chem_env=chem_env, bact_env=bact_env, env_times=env_times,
        neuron_names=neuron_names, activity_raw=activity_raw, T=T,
        env_clouds=env_clouds, NX=NX, NZ=NZ,
    )


def process_activity(activity_raw):
    if activity_raw is None:
        return None
    SILENT_THRESHOLD = 0.008
    SIGMA_CLIP       = 0.25
    CONTRAST_GAMMA   = 1.5
    raw_max = activity_raw.max(axis=0)
    active  = raw_max > SILENT_THRESHOLD
    mu      = activity_raw.mean(axis=0)
    std     = activity_raw.std(axis=0)
    std_safe= np.where(std > 1e-6, std, 1.0)
    zscored = (activity_raw - mu[None,:]) / std_safe[None,:]
    normed  = np.clip((zscored + SIGMA_CLIP) / (2*SIGMA_CLIP), 0, 1).astype(np.float32)
    activity = np.power(normed, CONTRAST_GAMMA).astype(np.float32)
    activity[:, ~active] = 0.0
    return activity


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLOUD (GPU)  —  identical to dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gpu_particles(NX, NZ):
    g = torch.Generator(device='cpu')
    g.manual_seed(42)
    particles = {}
    for fi, name in enumerate(FIELD_NAMES):
        sty = FIELD_STYLE.get(name, _DEFAULT_STYLE)
        if sty['alpha_scale'] == 0.0:
            particles[name] = None
            continue
        yr, yb = sty['y_range'], sty['y_bias']
        px = torch.rand(N_CHEM_PTS, generator=g)
        pz = torch.rand(N_CHEM_PTS, generator=g)
        py = torch.empty(N_CHEM_PTS).uniform_(yb-yr, yb+yr)
        particles[name] = (px.to(_DEVICE), py.to(_DEVICE), pz.to(_DEVICE))
    particles['bacteria'] = None
    return particles


def _gpu_sample_field(field_np, px_gpu, pz_gpu, sigma):
    NX, NZ = field_np.shape
    fmax = max(float(np.percentile(field_np, 99)) if field_np.max() > 0 else 1.0, 1e-9)
    fn   = gaussian_filter(np.clip(field_np / fmax, 0, 1).astype(np.float32), sigma=sigma)
    field_t = torch.from_numpy(fn).to(_DEVICE)
    gx = (px_gpu * (NX - 1)).clamp(0, NX - 1)
    gz = (pz_gpu * (NZ - 1)).clamp(0, NZ - 1)
    x0 = gx.long().clamp(0, NX - 2); z0 = gz.long().clamp(0, NZ - 2)
    x1 = x0 + 1;                      z1 = z0 + 1
    fx = gx - x0.float();             fz = gz - z0.float()
    intensity = (field_t[x0, z0]*(1-fx)*(1-fz) + field_t[x1, z0]*fx*(1-fz) +
                 field_t[x0, z1]*(1-fx)*fz      + field_t[x1, z1]*fx*fz)
    return intensity.cpu().numpy()


def _build_env_clouds_gpu(chem_env_all, bact_env_all, gpu_particles):
    n_snaps = len(chem_env_all)
    NX, NZ  = bact_env_all[0].shape
    results = []
    print(f"  GPU cloud precompute: {n_snaps} snapshots on {_DEVICE}...")
    for si in range(n_snaps):
        chem_frame = chem_env_all[si]
        bact_frame = bact_env_all[si]
        all_chem_pos = np.zeros((15 * N_CHEM_PTS, 3), dtype=np.float32)
        all_chem_col = np.zeros((15 * N_CHEM_PTS, 4), dtype=np.float32)
        for fi, name in enumerate(FIELD_NAMES):
            sl  = slice(fi * N_CHEM_PTS, (fi + 1) * N_CHEM_PTS)
            sty = FIELD_STYLE.get(name, _DEFAULT_STYLE)
            if sty['alpha_scale'] == 0.0 or gpu_particles[name] is None:
                continue
            px_g, py_g, pz_g = gpu_particles[name]
            intensity = _gpu_sample_field(chem_frame[fi], px_g, pz_g, sty['sigma'])
            alpha = np.clip(intensity * sty['alpha_scale'] + 0.003, 0, sty['alpha_max']).astype(np.float32)
            col = np.zeros((N_CHEM_PTS, 4), dtype=np.float32)
            col[:, :3] = _FIELD_RGB[fi]
            col[:, 3]  = alpha
            all_chem_pos[sl, 0] = px_g.cpu().numpy() * LX
            all_chem_pos[sl, 1] = py_g.cpu().numpy()
            all_chem_pos[sl, 2] = pz_g.cpu().numpy() * LZ
            all_chem_col[sl]    = col
        B_THRESHOLD = 0.01
        NX_b, NZ_b = bact_frame.shape
        xi_all = np.arange(NX_b, dtype=np.float32)
        zi_all = np.arange(NZ_b, dtype=np.float32)
        XX_b, ZZ_b = np.meshgrid(xi_all, zi_all, indexing='ij')
        mask_b = bact_frame > B_THRESHOLD
        bx_wu  = XX_b[mask_b] * (LX / NX_b)
        bz_wu  = ZZ_b[mask_b] * (LZ / NZ_b)
        bvals  = bact_frame[mask_b]
        xi_m   = XX_b[mask_b].astype(np.int32)
        zi_m   = ZZ_b[mask_b].astype(np.int32)
        cell_hash = (xi_m * 1664525 + zi_m * 1013904223) & 0xFFFF
        by_wu  = (cell_hash.astype(np.float32) / 65535.0 * 2.0 - 1.0) * 1.5
        bpos   = np.stack([bx_wu, by_wu, bz_wu], axis=1).astype(np.float32)
        bcol   = np.zeros((len(bvals), 4), dtype=np.float32)
        bcol[:, 0] = np.clip(0.02 + (1.0 - bvals) * 0.20, 0, 0.22)
        bcol[:, 1] = np.clip(0.72 + bvals * 0.23, 0, 0.95)
        bcol[:, 2] = np.clip(0.04 + (1.0 - bvals) * 0.08, 0, 0.12)
        bcol[:, 3] = np.clip(bvals * 0.90 + 0.08, 0, 0.92).astype(np.float32)
        soil_fi = FIELD_NAMES.index('soil_density')
        sd = chem_frame[soil_fi].astype(np.float32)
        sdmax = max(float(sd.max()), 1e-9)
        soil_intensity = np.clip(sd / sdmax, 0, 1)
        results.append(dict(
            chem_pos=all_chem_pos, chem_col=all_chem_col,
            bact_pos=bpos,         bact_col=bcol,
            soil_intensity=soil_intensity,
        ))
    print("  GPU cloud precompute done.")
    return results


def body_to_env_idx(fi, times, env_times):
    t = times[fi]
    i = np.searchsorted(env_times, t, side='right') - 1
    return int(np.clip(i, 0, len(env_times) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# WORM BODY HELPERS  —  identical to dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def radius_profile(s):
    head = np.clip(s/0.08, 0, 1)
    tail = np.clip((1-s)/0.12, 0, 1)
    return 0.05 + 0.60 * head * tail

def muscle_at_s(s, md, mv):
    if s < 0.20: return 0.0, 0.0
    seg = min(int((s-0.20)/0.80*24), 23)
    return float(md[seg]), float(mv[seg])

def build_worm_tube(bp, md, mv):
    from vispy.visuals.tube import _frenet_frames
    pts = np.zeros_like(bp, dtype=np.float64)
    for ax in range(3):
        pts[:,ax] = uniform_filter1d(bp[:,ax].astype(np.float64), size=4)
    pts3 = np.column_stack([pts[:,0], np.zeros(len(pts)), pts[:,2]])
    diffs = np.diff(pts3, axis=0)
    seg_l = np.linalg.norm(diffs, axis=1)
    seg_l = np.where(seg_l<1e-8, 1e-8, seg_l)
    u = np.concatenate([[0], np.cumsum(seg_l)]); u /= u[-1]
    cs = CubicSpline(u, pts3)
    u_new = np.linspace(0, 1, N_SPLINE)
    spl_pts = cs(u_new).astype(np.float32)

    md_n = md / max(md.max(), 1e-6)
    mv_n = mv / max(mv.max(), 1e-6)
    radii_raw = np.array([max(radius_profile(s), 0.05) for s in u_new])
    radii = gaussian_filter1d(radii_raw, sigma=3.0)

    tang, nf_raw, bf_raw = _frenet_frames(spl_pts, False)
    nf = np.array(nf_raw, dtype=np.float32)
    bf = np.array(bf_raw, dtype=np.float32)
    r_arr = radii.astype(np.float32)
    r_y = r_arr*0.65; r_z = r_arr*1.15
    N = N_SPLINE; R = TUBE_PTS
    angles = np.linspace(0, 2*np.pi, R, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    cos_R = cos_a[None,:,None]; sin_R = sin_a[None,:,None]
    ry_N=r_y[:,None,None]; rz_N=r_z[:,None,None]
    vgrid = spl_pts[:,None,:] + cos_R*nf[:,None,:]*ry_N + sin_R*bf[:,None,:]*rz_N
    verts = vgrid.reshape(-1,3).astype(np.float32)

    engrid = cos_R*nf[:,None,:]/ry_N + sin_R*bf[:,None,:]/rz_N
    elen = np.linalg.norm(engrid,axis=2,keepdims=True)
    engrid = engrid/np.where(elen<1e-8,1,elen)
    for ax in range(3):
        engrid[:,:,ax] = gaussian_filter1d(engrid[:,:,ax], sigma=2.0, axis=0)
    nlen = np.linalg.norm(engrid,axis=2,keepdims=True)
    norms = (engrid/np.where(nlen<1e-8,1,nlen)).reshape(-1,3).astype(np.float32)

    key_dir = np.array([0.2,0.7,0.5],dtype=np.float32); key_dir/=np.linalg.norm(key_dir)
    rim_dir = np.array([-0.3,-0.5,-0.6],dtype=np.float32); rim_dir/=np.linalg.norm(rim_dir)
    view_dir= np.array([0.1,1.0,0.2],dtype=np.float32)
    half_v  = key_dir+view_dir; half_v/=np.linalg.norm(half_v)
    key2  = np.clip(norms@key_dir,0,1)*0.75
    rim2  = np.clip(norms@rim_dir,0,1)*0.20
    spec2 = np.clip(norms@half_v, 0,1)**40*0.55
    base2 = 0.72+key2*0.15
    d_acts = np.array([muscle_at_s(u_new[i],md_n,mv_n)[0] for i in range(N)],dtype=np.float32)
    v_acts = np.array([muscle_at_s(u_new[i],md_n,mv_n)[1] for i in range(N)],dtype=np.float32)
    d_pv = np.repeat(d_acts,R); v_pv = np.repeat(v_acts,R)
    ca_pv = np.tile(cos_a,N)
    def _hash(x): return (np.sin(x*127.1+311.7)*43758.5453)%1.0
    noise = (0.5*_hash(verts[:,0]*3.1+verts[:,2]*1.7)+
             0.3*_hash(verts[:,0]*6.3-verts[:,2]*2.9)+
             0.2*_hash(verts[:,1]*11.0+verts[:,0]*4.1))
    noise = (noise-0.5)*0.08
    back_lit = np.clip(-(norms@key_dir),0,1)
    sss_r = back_lit*(0.35*d_pv+0.15); sss_g = back_lit*(0.20*d_pv+0.08)
    dt = np.maximum(0,ca_pv)*d_pv; vt = np.maximum(0,-ca_pv)*v_pv
    trans_d = np.maximum(0,-ca_pv)*d_pv*0.18
    trans_v = np.maximum(0, ca_pv)*v_pv*0.18
    colors = np.stack([
        np.clip(base2+dt*0.30-vt*0.12+spec2+rim2*0.05+sss_r+trans_d*0.30+noise,0,1),
        np.clip(base2+dt*0.08+vt*0.18+spec2*0.9+rim2*0.08+sss_g+trans_v*0.12+noise*0.8,0,1),
        np.clip(base2-dt*0.20+vt*0.30+spec2*0.8+rim2*0.18+noise*0.6,0,1),
        np.full(N*R,0.88,dtype=np.float32)],axis=1).astype(np.float32)

    if not hasattr(build_worm_tube,'_faces') or build_worm_tube._NR != (N,R):
        fi2,fj = np.mgrid[0:N-1,0:R]
        fj1=(fj+1)%R
        a=fi2*R+fj; b=fi2*R+fj1; c=(fi2+1)*R+fj; d=(fi2+1)*R+fj1
        build_worm_tube._faces = np.stack([
            np.stack([a,b,d],axis=2),np.stack([a,d,c],axis=2)
        ],axis=2).reshape(-1,3).astype(np.uint32)
        build_worm_tube._NR = (N,R)
    return verts, build_worm_tube._faces, colors, spl_pts, u_new, md_n, mv_n, nf, bf, r_y, r_z


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTOME HELPERS  —  identical to dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _hex_rgb(h):
    h=h.lstrip("#")
    return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)],dtype=np.float32)

def build_colour_arrays(neuron_names):
    dim_rgb  = np.zeros((len(neuron_names),3),dtype=np.float32)
    glow_rgb = np.zeros((len(neuron_names),3),dtype=np.float32)
    dd,dg = _hex_rgb(COLOR_CLASSES[-1][0]),_hex_rgb(COLOR_CLASSES[-1][1])
    for i,nm in enumerate(neuron_names):
        assigned=False
        for dim_h,glow_h,prefixes in COLOR_CLASSES[:-1]:
            if any(nm.startswith(p) for p in prefixes):
                dim_rgb[i]=_hex_rgb(dim_h); glow_rgb[i]=_hex_rgb(glow_h)
                assigned=True; break
        if not assigned:
            dim_rgb[i]=dd; glow_rgb[i]=dg
    return dim_rgb, glow_rgb

def parse_nml(nml_path, name_set, min_weight=3.0):
    if not os.path.exists(nml_path):
        return []
    tree=ET.parse(nml_path); root=tree.getroot()
    ns=(root.tag.split("}")[0]+"}") if root.tag.startswith("{") else ""
    edges=[]
    for proj in root.iter(f"{ns}projection"):
        pre=proj.attrib.get("presynapticPopulation","")
        post=proj.attrib.get("postsynapticPopulation","")
        syn=proj.attrib.get("synapse","")
        inh="inh" in syn.lower()
        for conn in proj.iter(f"{ns}connectionWD"):
            w=float(conn.attrib.get("weight",1.0))
            if pre in name_set and post in name_set and w>=min_weight:
                edges.append((pre,post,w,inh))
            break
    return edges

def build_layout(neuron_names):
    pos={}
    for e in EXPLICIT:
        x=0.5+(e[1]-0.5)*1.1; pos[e[0]]=(x,e[2])
    for i in range(1,10): pos[f"DA{i}"]=(0.79,0.63-(i-1)*(0.41/8))
    for i in range(1,8):  pos[f"DB{i}"]=(0.75,0.63-(i-1)*(0.41/6))
    for i in range(1,7):  pos[f"DD{i}"]=(0.70,0.60-(i-1)*(0.38/5))
    for i in range(1,13): pos[f"VA{i}"]=(0.21,0.63-(i-1)*(0.41/11))
    for i in range(1,12): pos[f"VB{i}"]=(0.25,0.63-(i-1)*(0.41/10))
    for i in range(1,14): pos[f"VD{i}"]=(0.30,0.60-(i-1)*(0.38/12))
    for i in range(1,12): pos[f"AS{i}"]=(0.50,0.63-(i-1)*(0.41/10))
    rng=np.random.default_rng(42)
    for nm in neuron_names:
        if nm not in pos:
            a=rng.uniform(0,2*np.pi); r=0.06+rng.uniform(0,0.03)
            pos[nm]=(0.50+r*np.cos(a),0.50+r*np.sin(a))
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# OFFSCREEN RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class OffscreenRenderer:
    """
    Renders one frame of the three-panel dashboard to a numpy RGB array.
    Constructed once per HDF5; call render_frame(fi) repeatedly.

    Panel pixel sizes are fixed at construction time.
    """

    def __init__(self, data, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.data   = data
        self.W      = width
        self.H      = height
        self.activity = process_activity(data['activity_raw'])

        # Pixel budgets
        self.left_w  = int(width * LEFT_FRAC)
        self.right_w = width - self.left_w
        self.conn_h  = int(height * CONN_FRAC)
        self.kymo_h  = height - self.conn_h

        self._build_vispy()
        self._build_connectome()
        self._build_kymograph()

    # ── vispy (left panel) ──────────────────────────────────────────────────

    def _build_vispy(self):
        self.canvas = scene.SceneCanvas(
            size=(self.left_w, self.H),
            bgcolor=BG_HEX,
            show=False,
            keys=None,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=50, distance=180, elevation=-60, azimuth=0,
            center=(820, 0, 270))

        # Soil background particles
        _r = np.random.default_rng(42)
        N_SOIL = 4000
        sp = np.zeros((N_SOIL,3),dtype=np.float32)
        sp[:,0]=_r.uniform(0,LX,N_SOIL)
        sp[:,1]=_r.uniform(-4.5,8,N_SOIL)
        sp[:,2]=_r.uniform(0,LZ,N_SOIL)
        sc=np.zeros((N_SOIL,4),dtype=np.float32)
        rv=_r.random(N_SOIL)
        sc[rv>0.97]=[0.55,0.50,0.44,0.75]
        sc[(rv>0.87)&(rv<=0.97)]=[0.24,0.10,0.04,0.65]
        sc[rv<=0.87]=[0.06,0.04,0.08,0.55]
        self._soil_pos      = sp
        self._soil_col_base = sc.copy()
        self.soil_vis = visuals.Markers()
        self.soil_vis.set_data(sp, face_color=sc, size=1.8, edge_width=0)
        self.view.add(self.soil_vis)

        # Chemical / bacteria clouds
        self.cloud_chem = visuals.Markers()
        self.cloud_chem.set_gl_state('additive', depth_test=False)
        self.cloud_bact = visuals.Markers()
        self.cloud_bact.set_gl_state('translucent', depth_test=False)
        snap0 = self.data['env_clouds'][0]
        self.cloud_chem.set_data(snap0['chem_pos'], face_color=snap0['chem_col'], size=10, edge_width=0)
        self.cloud_bact.set_data(snap0['bact_pos'], face_color=snap0['bact_col'], size=10, edge_width=0)
        self.view.add(self.cloud_chem)
        self.view.add(self.cloud_bact)

        # Patch centre markers
        _patch_centres = np.array([
            [399, 0, 327], [249, 0, 214], [615, 0, 142], [572, 0, 341]
        ], dtype=np.float32)
        pm = visuals.Markers()
        pm.set_data(_patch_centres, face_color=(0.08, 0.55, 0.18, 0.30), size=8, edge_width=0)
        self.view.add(pm)

        # Worm mesh (persistent)
        _NR, _R = N_SPLINE, TUBE_PTS
        _fi2, _fj = np.mgrid[0:_NR-1, 0:_R]
        _fj1 = (_fj+1) % _R
        _a=_fi2*_R+_fj; _b=_fi2*_R+_fj1; _c=(_fi2+1)*_R+_fj; _d=(_fi2+1)*_R+_fj1
        self._worm_faces = np.stack(
            [np.stack([_a,_b,_d],axis=2),np.stack([_a,_d,_c],axis=2)],
            axis=2).reshape(-1,3).astype(np.uint32)
        self.worm_vis = visuals.Mesh(
            vertices=np.zeros((_NR*_R, 3), dtype=np.float32),
            faces=self._worm_faces,
            vertex_colors=np.ones((_NR*_R, 4), dtype=np.float32),
            shading=None)
        self.view.add(self.worm_vis)

        # Pharynx mesh
        _PH = 24
        _pfi2, _pfj = np.mgrid[0:_PH-1, 0:_R]
        _pfj1 = (_pfj+1) % _R
        _pa=_pfi2*_R+_pfj; _pb=_pfi2*_R+_pfj1
        _pc=(_pfi2+1)*_R+_pfj; _pd=(_pfi2+1)*_R+_pfj1
        self._pharynx_faces = np.stack(
            [np.stack([_pa,_pb,_pd],axis=2),np.stack([_pa,_pd,_pc],axis=2)],
            axis=2).reshape(-1,3).astype(np.uint32)
        self.pharynx_vis = visuals.Mesh(
            vertices=np.zeros((_PH*_R, 3), dtype=np.float32),
            faces=self._pharynx_faces,
            vertex_colors=np.ones((_PH*_R, 4), dtype=np.float32),
            shading=None)
        self.view.add(self.pharynx_vis)

        # HUD text
        self.frame_txt = scene.visuals.Text(
            '', color='#446677', font_size=9,
            anchor_x='left', anchor_y='top',
            parent=self.canvas.scene)
        self.frame_txt.transform = scene.transforms.STTransform(translate=(12, 18, 0))

        self.state_txt = scene.visuals.Text(
            '', color='#88bbcc', font_size=10,
            anchor_x='left', anchor_y='top',
            parent=self.canvas.scene)
        self.state_txt.transform = scene.transforms.STTransform(translate=(12, 38, 0))

    def _render_vispy(self, fi):
        bp  = self.data['body_all'][fi]
        md  = self.data['muscle_d'][fi]
        mv  = self.data['muscle_v'][fi]
        verts,faces,colors,spl_pts,u_new,md_n,mv_n,nf,bf,r_y,r_z = build_worm_tube(bp,md,mv)
        self.worm_vis.set_data(vertices=verts, faces=faces, vertex_colors=colors)

        # Pharynx
        Np = min(max(4, int(len(u_new)*0.16)), 24)
        R  = TUBE_PTS
        angles = np.linspace(0, 2*np.pi, R, endpoint=False)
        cos_a  = np.cos(angles).astype(np.float32)
        sin_a  = np.sin(angles).astype(np.float32)
        ph_r   = gaussian_filter1d(
            np.array([max(radius_profile(u_new[i]),0.05) for i in range(Np)],
                     dtype=np.float32), sigma=3.0) * 0.75
        ry_p = (ph_r*0.55)[:,None,None]; rz_p = (ph_r*0.90)[:,None,None]
        vg_p = (spl_pts[:Np,None,:] +
                cos_a[None,:,None]*nf[:Np,None,:]*ry_p +
                sin_a[None,:,None]*bf[:Np,None,:]*rz_p)
        vp   = vg_p.reshape(-1,3).astype(np.float32)
        en   = cos_a[None,:,None]*nf[:Np,None,:]/ry_p + sin_a[None,:,None]*bf[:Np,None,:]/rz_p
        el   = np.linalg.norm(en,axis=2,keepdims=True)
        en   = en/np.where(el<1e-8,1,el)
        key_dir = np.array([0.2,0.7,0.5],dtype=np.float32); key_dir/=np.linalg.norm(key_dir)
        ph_nf  = en.reshape(-1,3).astype(np.float32)
        ph_key = np.clip(ph_nf@key_dir,0,1)*0.6
        pb     = 0.28+ph_key*0.15
        ph_cols= np.stack([
            np.clip(pb+0.05,0,1), np.clip(pb,0,1),
            np.clip(pb-0.05,0,1), np.full(Np*R,0.95)
        ], axis=1).astype(np.float32)
        if Np < 24:
            pad = 24-Np
            vp      = np.vstack([vp,      np.tile(vp[-R:],      (pad,1))])
            ph_cols = np.vstack([ph_cols, np.tile(ph_cols[-R:],  (pad,1))])
        self.pharynx_vis.set_data(vertices=vp, faces=self._pharynx_faces, vertex_colors=ph_cols)

        # Camera follows worm nose
        self.view.camera.center = (float(bp[0,0]), 0, float(bp[0,2]))

        # Update environment
        ei   = body_to_env_idx(fi, self.data['times'], self.data['env_times'])
        snap = self.data['env_clouds'][ei]
        self.cloud_chem.set_data(snap['chem_pos'], face_color=snap['chem_col'], size=10, edge_width=0)
        self.cloud_bact.set_data(snap['bact_pos'], face_color=snap['bact_col'], size=10, edge_width=0)
        if snap['soil_intensity'] is not None:
            NX, NZ = self.data['NX'], self.data['NZ']
            sp = self._soil_pos
            xi = np.clip((sp[:,0]/LX*NX).astype(int), 0, NX-1)
            zi = np.clip((sp[:,2]/LZ*NZ).astype(int), 0, NZ-1)
            density = snap['soil_intensity'][xi, zi].astype(np.float32)
            sc = self._soil_col_base.copy()
            sc[:,:3] = np.clip(self._soil_col_base[:,:3] * (0.7+0.5*density[:,None]), 0, 1)
            self.soil_vis.set_data(sp, face_color=sc, size=1.8, edge_width=0)

        t   = self.data['times'][min(fi, len(self.data['times'])-1)]
        rev = bool(self.data['reversing'][min(fi, len(self.data['reversing'])-1)])
        qui = bool(self.data['quiescent'][min(fi, len(self.data['quiescent'])-1)])
        self.frame_txt.text = f'FRAME {fi:05d}  /  t={t:.2f}s'
        parts = []
        if qui: parts.append("QUIESCENT")
        if rev: parts.append("REVERSING")
        self.state_txt.text = "  ".join(parts) or "FORAGING"

        # Offscreen render → numpy
        img = self.canvas.render(alpha=False)   # (H, W, 3) uint8
        return img

    # ── connectome (top-right) ──────────────────────────────────────────────

    def _build_connectome(self):
        nn = self.data['neuron_names']
        if not nn:
            self._conn_ready = False
            return
        self._conn_ready = True
        self.pos       = build_layout(nn)
        self.dim_rgb, self.glow_rgb = build_colour_arrays(nn)
        name_set  = set(nn)
        edges     = parse_nml(NML_PATH, name_set)
        self.node_x = np.array([self.pos[n][0] for n in nn], dtype=np.float32)
        self.node_y = np.array([self.pos[n][1] for n in nn], dtype=np.float32)
        name_to_idx = {n:i for i,n in enumerate(nn)}
        exc_segs, exc_pre = [], []
        inh_segs, inh_pre = [], []
        for pre, post, w, inh in edges:
            if pre not in name_to_idx or post not in name_to_idx: continue
            p0, p1 = self.pos[pre], self.pos[post]
            seg = [(p0[0],p0[1]), (p1[0],p1[1])]
            if inh: inh_segs.append(seg); inh_pre.append(name_to_idx[pre])
            else:   exc_segs.append(seg); exc_pre.append(name_to_idx[pre])
        self.exc_segs = exc_segs; self.exc_pre = np.array(exc_pre, dtype=int)
        self.inh_segs = inh_segs; self.inh_pre = np.array(inh_pre, dtype=int)
        # Colour look-up tables
        exc_dim_r = self.dim_rgb[self.exc_pre]
        exc_bri_r = self.glow_rgb[self.exc_pre]
        self.exc_dim  = np.hstack([exc_dim_r, np.full((len(exc_dim_r),1), 0.12)])
        self.exc_bri  = np.hstack([exc_bri_r, np.full((len(exc_bri_r),1), 0.80)])
        inh_dim_r = self.dim_rgb[self.inh_pre]
        inh_bri_r = np.tile(np.array([[0.55,0.10,0.18]],dtype=np.float32), (len(inh_dim_r),1))
        self.inh_dim  = np.hstack([inh_dim_r, np.full((len(inh_dim_r),1), 0.12)])
        self.inh_bri  = np.hstack([inh_bri_r, np.full((len(inh_bri_r),1), 0.70)])

        # Figure — Agg backend, fixed pixels
        dpi = 100
        fw = self.right_w / dpi
        fh = self.conn_h  / dpi
        self.fig_conn, self.ax_conn = plt.subplots(1, 1, figsize=(fw, fh),
            facecolor=BG_HEX, subplot_kw=dict(facecolor="#030D18"))
        ax = self.ax_conn
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("CONNECTOME  302 neurons", color="#304050",
                     fontsize=7, fontfamily="monospace", pad=2)
        # Edges (drawn once; colours updated each frame)
        self.lc_exc = LineCollection(exc_segs, linewidths=0.3, zorder=1)
        self.lc_inh = LineCollection(inh_segs, linewidths=0.3, zorder=1)
        ax.add_collection(self.lc_exc)
        ax.add_collection(self.lc_inh)
        # Glow nodes
        self.scat_glow = ax.scatter(
            self.node_x, self.node_y, s=8, c=self.glow_rgb,
            zorder=3, linewidths=0)
        # Ring nodes
        self.scat_ring = ax.scatter(
            self.node_x, self.node_y, s=8,
            facecolors='none', edgecolors=self.glow_rgb, linewidths=0.5, zorder=4)
        self.time_txt  = ax.text(0.02, 0.98, '', color="#226633",
                                 fontsize=7, fontfamily="monospace",
                                 va='top', transform=ax.transAxes)
        self.state_conn_txt = ax.text(0.50, 0.98, '', color="#88bbcc",
                                 fontsize=7, fontfamily="monospace",
                                 va='top', ha='center', transform=ax.transAxes)
        self.fig_conn.tight_layout(pad=0.2)
        self._conn_canvas = FigureCanvasAgg(self.fig_conn)

    def _render_connectome(self, fi):
        if not self._conn_ready:
            img = np.zeros((self.conn_h, self.right_w, 3), dtype=np.uint8)
            img[:] = BG_RGB
            return img
        act = self.activity[min(fi, len(self.activity)-1)] if self.activity is not None \
              else np.zeros(len(self.data['neuron_names']), dtype=np.float32)

        exc_col = self.exc_dim + (self.exc_bri - self.exc_dim) * act[self.exc_pre, None]
        self.lc_exc.set_colors(exc_col)
        self.lc_exc.set_linewidths(0.22 + 0.28*act[self.exc_pre])
        inh_col = self.inh_dim + (self.inh_bri - self.inh_dim) * act[self.inh_pre, None]
        self.lc_inh.set_colors(inh_col)
        self.lc_inh.set_linewidths(0.22 + 0.28*act[self.inh_pre])

        glow_rgba = np.hstack([self.glow_rgb, act[:,None]])
        self.scat_glow.set_color(glow_rgba)
        self.scat_glow.set_sizes((8 + 30*act**0.7).astype(float))
        ring_rgb  = self.glow_rgb*0.6 + 0.4
        ring_rgba = np.hstack([ring_rgb, (act**0.5)[:,None]])
        self.scat_ring.set_edgecolors(ring_rgba)
        self.scat_ring.set_sizes((8 + 30*act**0.7).astype(float))

        t   = self.data['times'][min(fi, len(self.data['times'])-1)]
        rev = bool(self.data['reversing'][min(fi, len(self.data['reversing'])-1)])
        qui = bool(self.data['quiescent'][min(fi, len(self.data['quiescent'])-1)])
        self.time_txt.set_text(f"t = {t:.2f}s")
        parts = []
        if qui: parts.append("QUIESCENT")
        if rev: parts.append("REVERSING")
        self.state_conn_txt.set_text("  ".join(parts) or "FORAGING")

        self._conn_canvas.draw()
        buf = np.frombuffer(self._conn_canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(self.fig_conn.canvas.get_width_height()[::-1] + (3,))
        # Resize to exact pixel budget if needed
        if img.shape[1] != self.right_w or img.shape[0] != self.conn_h:
            import cv2
            img = cv2.resize(img, (self.right_w, self.conn_h))
        return img

    # ── kymograph (bottom-right) ─────────────────────────────────────────────

    def _build_kymograph(self):
        dpi = 100
        fw = self.right_w / dpi
        fh = self.kymo_h  / dpi
        self.fig_kymo, self.ax_kymo = plt.subplots(1, 1, figsize=(fw, fh),
            facecolor=BG_HEX, subplot_kw=dict(facecolor="#060810"))
        ax = self.ax_kymo
        if self.activity is not None and self.data['activity_raw'] is not None:
            raw   = self.data['activity_raw']
            order = np.argsort(raw.std(axis=0))[::-1][:80]
            raw_s = raw[:, order]
            r_min = raw_s.min(axis=0, keepdims=True)
            r_max = raw_s.max(axis=0, keepdims=True)
            r_rng = np.where((r_max-r_min)>1e-6, r_max-r_min, 1.0)
            raw_norm = np.clip((raw_s - r_min) / r_rng, 0, 1).astype(np.float32)
            ax.imshow(raw_norm.T, aspect='auto', cmap='magma',
                      origin='upper', interpolation='nearest', vmin=0, vmax=1,
                      extent=[0, self.data['times'][-1], 80, 0])
            ax.set_xlabel("time (s)", color="#446", fontsize=7, fontfamily="monospace")
            ax.set_ylabel("neuron", color="#446", fontsize=7, fontfamily="monospace")
            ax.tick_params(colors="#446", labelsize=6)
            for sp in ax.spines.values(): sp.set_color("#1a2030")
            self.kymo_line = ax.axvline(0, color="#FF8844", linewidth=1.0, alpha=0.8)
            self._kymo_xmax = float(self.data['times'][-1])
        else:
            ax.text(0.5, 0.5, "No activity data", color="#446",
                    ha="center", va="center", transform=ax.transAxes)
            self.kymo_line = None
        ax.set_title("NEURAL ACTIVITY KYMOGRAPH", color="#304050",
                     fontsize=7, fontfamily="monospace", pad=2)
        self.fig_kymo.tight_layout(pad=0.3)
        self._kymo_canvas = FigureCanvasAgg(self.fig_kymo)
        # Pre-render static background
        self._kymo_canvas.draw()
        self._kymo_bg_buf = np.frombuffer(
            self._kymo_canvas.tostring_rgb(), dtype=np.uint8).copy()
        self._kymo_bg_buf = self._kymo_bg_buf.reshape(
            self.fig_kymo.canvas.get_width_height()[::-1] + (3,)).copy()

    def _render_kymograph(self, fi):
        if self.kymo_line is None:
            img = np.zeros((self.kymo_h, self.right_w, 3), dtype=np.uint8)
            img[:] = BG_RGB
            return img
        t = float(self.data['times'][min(fi, len(self.data['times'])-1)])
        self.kymo_line.set_xdata([t, t])
        self._kymo_canvas.draw()
        buf = np.frombuffer(self._kymo_canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(self.fig_kymo.canvas.get_width_height()[::-1] + (3,)).copy()
        if img.shape[1] != self.right_w or img.shape[0] != self.kymo_h:
            import cv2
            img = cv2.resize(img, (self.right_w, self.kymo_h))
        return img

    # ── composite frame ───────────────────────────────────────────────────────

    def render_frame(self, fi):
        """
        Render frame fi and return a (H, W, 3) uint8 RGB array.
        """
        left   = self._render_vispy(fi)      # (H, left_w, 3)
        conn   = self._render_connectome(fi)  # (conn_h, right_w, 3)
        kymo   = self._render_kymograph(fi)   # (kymo_h, right_w, 3)

        # Ensure vispy panel height matches
        if left.shape[0] != self.H or left.shape[1] != self.left_w:
            import cv2
            left = cv2.resize(left, (self.left_w, self.H))

        right = np.zeros((self.H, self.right_w, 3), dtype=np.uint8)
        right[:self.conn_h, :]  = conn
        right[self.conn_h:, :]  = kymo

        frame = np.concatenate([left, right], axis=1)  # (H, W, 3)
        return frame

    def close(self):
        self.canvas.close()
        plt.close(self.fig_conn)
        plt.close(self.fig_kymo)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETED RUN TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

RENDER_FIELDNAMES = ['h5_path', 'video_path', 'status', 'notes', 'n_frames',
                     'wall_time_s', 'timestamp']

def load_render_completed():
    done = {}
    if os.path.exists(COMPLETED_CSV):
        with open(COMPLETED_CSV) as f:
            for row in csv.DictReader(f):
                done[row['h5_path']] = row
    return done

def write_render_completed(writer, f_out, h5_path, video_path,
                            status, notes, n_frames, wall_time):
    row = {
        'h5_path':    h5_path,
        'video_path': video_path or '',
        'status':     status,
        'notes':      notes,
        'n_frames':   n_frames,
        'wall_time_s': round(wall_time, 1),
        'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    writer.writerow(row)
    f_out.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-FILE RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

_abort = False

def _handle_sigint(sig, frame):
    global _abort
    print("\n[!] Ctrl+C caught — finishing current video then stopping...")
    _abort = True

signal.signal(signal.SIGINT, _handle_sigint)


def render_h5(h5_path, video_path, skip_if_exists=True):
    """
    Render one HDF5 to one MP4. Returns (status, notes, n_frames, wall_time).
    """
    if skip_if_exists and os.path.exists(video_path):
        size = os.path.getsize(video_path)
        if size > 10_000:
            print(f"  [skip] already exists: {video_path}")
            return 'skipped', 'already_exists', 0, 0.0

    t0 = time.time()

    # Partial output path — written to during render, moved on success
    tmp_path = video_path + ".tmp.mp4"

    try:
        data = load_data(h5_path)
    except Exception as e:
        return 'failed', f'load_error: {str(e)[:60]}', 0, time.time()-t0

    T = data['T']
    # Frame indices to render (downsampled by SIM_FPS)
    frame_indices = list(range(0, T, max(1, SIM_FPS)))
    n_frames = len(frame_indices)
    print(f"  Rendering {n_frames} frames ({T} body frames, step={SIM_FPS})...")

    try:
        renderer = OffscreenRenderer(data)
    except Exception as e:
        return 'failed', f'renderer_init: {str(e)[:60]}', 0, time.time()-t0

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, OUTPUT_FPS,
                             (VIDEO_WIDTH, VIDEO_HEIGHT))
    if not writer.isOpened():
        renderer.close()
        return 'failed', 'cv2_writer_failed_to_open', 0, time.time()-t0

    try:
        for idx, fi in enumerate(frame_indices):
            if _abort:
                writer.release()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return 'aborted', 'user_interrupt', idx, time.time()-t0
            frame_rgb = renderer.render_frame(fi)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            if (idx+1) % 50 == 0 or idx == n_frames-1:
                elapsed = time.time() - t0
                fps_so_far = (idx+1) / elapsed if elapsed > 0 else 0
                eta = (n_frames - idx - 1) / fps_so_far if fps_so_far > 0 else 0
                print(f"    {idx+1}/{n_frames}  {fps_so_far:.1f} fps  ETA {eta:.0f}s")
    except Exception as e:
        writer.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        renderer.close()
        return 'failed', f'render_error: {str(e)[:60]}', 0, time.time()-t0

    writer.release()
    renderer.close()

    # Move to final path on success
    os.replace(tmp_path, video_path)
    wall_time = time.time() - t0
    print(f"  Done: {video_path}  ({wall_time:.0f}s)")
    return 'completed', 'ok', n_frames, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# JOB DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_jobs(args):
    """
    Return list of (h5_path, video_path) tuples to render.
    If a specific h5 was given on the command line, just that one.
    Otherwise, scan genome_runs/ for completed genome_batch runs.
    """
    if args.h5:
        h5 = os.path.abspath(args.h5)
        genome_idx = 'manual'
        run_idx    = 0
        vid = os.path.join(VIDEO_DIR, f"manual_{os.path.basename(os.path.dirname(h5))}.mp4")
        return [(h5, vid)]

    jobs = []

    # Read genome_batch completed.csv if present
    if os.path.exists(GENOME_BATCH_CSV):
        with open(GENOME_BATCH_CSV) as f:
            for row in csv.DictReader(f):
                if row.get('status') != 'completed':
                    continue
                h5 = row.get('h5_path', '').strip()
                if not h5 or not os.path.exists(h5):
                    continue
                g = int(row['genome_idx'])
                r = int(row['run_idx'])
                vid = os.path.join(VIDEO_DIR, f"genome_{g:04d}", f"run_{r}.mp4")
                jobs.append((h5, vid))
    else:
        # Fallback: scan genome_runs directory directly
        pattern = os.path.join(GENOME_DIR, "genome_*", "run_*", "**", "kinematic_sim.h5")
        for h5 in glob.glob(pattern, recursive=True):
            parts = h5.split(os.sep)
            # Extract genome/run from path
            try:
                g_part = [p for p in parts if p.startswith("genome_")][-1]
                r_part = [p for p in parts if p.startswith("run_")][-1]
                g = int(g_part.split("_")[1])
                r = int(r_part.split("_")[1])
            except (IndexError, ValueError):
                g, r = 0, 0
            vid = os.path.join(VIDEO_DIR, f"genome_{g:04d}", f"run_{r}.mp4")
            jobs.append((h5, vid))

    print(f"Discovered {len(jobs)} completed genome runs to render")
    return jobs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global OUTPUT_FPS, SIM_FPS, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_QUALITY

    parser = argparse.ArgumentParser(description="SOIL genome video renderer")
    parser.add_argument('h5', nargs='?', default=None,
                        help="Optional: render a single HDF5 file")
    parser.add_argument('--fps', type=int, default=OUTPUT_FPS,
                        help=f"Output video FPS (default {OUTPUT_FPS})")
    parser.add_argument('--step', type=int, default=SIM_FPS,
                        help=f"Render every Nth simulation frame (default {SIM_FPS})")
    parser.add_argument('--width',  type=int, default=VIDEO_WIDTH)
    parser.add_argument('--height', type=int, default=VIDEO_HEIGHT)
    parser.add_argument('--crf',    type=int, default=VIDEO_QUALITY,
                        help="H.264 CRF (lower = better quality, default 22)")
    parser.add_argument('--force',  action='store_true',
                        help="Re-render even if output MP4 already exists")
    args = parser.parse_args()

    # Apply CLI overrides
    OUTPUT_FPS    = args.fps
    SIM_FPS       = args.step
    VIDEO_WIDTH   = args.width
    VIDEO_HEIGHT  = args.height
    VIDEO_QUALITY = args.crf

    os.makedirs(VIDEO_DIR, exist_ok=True)

    jobs = discover_jobs(args)
    if not jobs:
        print("No jobs found. Nothing to do.")
        return

    # Load already-rendered log
    done = load_render_completed()
    pending = [(h5, vid) for h5, vid in jobs
               if (h5 not in done or done[h5]['status'] not in ('completed','skipped'))
               and not (not args.force and os.path.exists(vid) and os.path.getsize(vid)>10_000)]

    print(f"{len(pending)} jobs to render  ({len(jobs)-len(pending)} already done)")

    write_header = not os.path.exists(COMPLETED_CSV)
    csv_f  = open(COMPLETED_CSV, 'a', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=RENDER_FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    n_ok = n_fail = n_skip = 0
    t_total = time.time()

    for i, (h5_path, video_path) in enumerate(pending):
        if _abort:
            break
        print(f"\n[{i+1}/{len(pending)}] {os.path.basename(h5_path)}")
        print(f"  → {video_path}")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        status, notes, n_frames, wall_time = render_h5(
            h5_path, video_path, skip_if_exists=not args.force)

        write_render_completed(writer, csv_f, h5_path, video_path,
                               status, notes, n_frames, wall_time)

        if status == 'completed':   n_ok   += 1
        elif status == 'skipped':   n_skip += 1
        elif status == 'aborted':   break
        else:                       n_fail += 1

        elapsed = time.time() - t_total
        remaining = len(pending) - i - 1
        avg = elapsed / (i+1)
        print(f"  [{status}]  total progress: {i+1}/{len(pending)}  "
              f"ETA {avg*remaining/60:.1f}min")

    csv_f.close()
    elapsed = time.time() - t_total
    print(f"\nDone.  {n_ok} rendered  {n_skip} skipped  {n_fail} failed  "
          f"in {elapsed/60:.1f}min")
    print(f"Videos: {VIDEO_DIR}/")
    print(f"Log:    {COMPLETED_CSV}")


if __name__ == '__main__':
    main()