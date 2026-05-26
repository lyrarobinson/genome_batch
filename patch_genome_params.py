#!/usr/bin/env python3
"""
Run this from ~/projects/worm/genome_batch/ to patch worm_kinematic_sim_graded.py
so all 8 genome parameters are overridable via environment variables.
"""

import os, sys, subprocess

TARGET = os.path.join(os.path.dirname(__file__), 'worm_kinematic_sim_graded.py')

with open(TARGET, 'r') as f:
    src = f.read()

errors = []

# ── 1. Replace module-level GC_AWA_SCALE / GC_SENSORY_SCALE / GC_ASH_SCALE / GC_SAAV_MAX ──
old = (
    "GC_AWA_SCALE        = 0.5    # nA per unit of awa_south/north difference (asymmetry amplifier)\n"
    "GC_SENSORY_SCALE    = 0.02   # nA per unit of sensory current (NEURON->graded scaling)\n"
    "GC_ASH_SCALE        = 65.0  # ASH boosted: osmolarity/pH signals weak in soil (~0.001nA raw)\n"
    "GC_SAAV_MAX         = 0.5   # nA: max SAAV injection at full satiation (worm.satiation=1.0)\n"
)
new = (
    "# ── Genome parameters: overridable via environment variables for batch runs ──\n"
    "import os as _os\n"
    "GC_AWA_SCALE        = float(_os.environ.get('GC_AWA_SCALE',      '0.5'))    # nA per unit of awa_south/north difference\n"
    "GC_SENSORY_SCALE    = float(_os.environ.get('GC_SENSORY_SCALE',  '0.02'))   # nA per unit of sensory current\n"
    "GC_ASH_SCALE        = float(_os.environ.get('GC_ASH_SCALE',      '65.0'))   # ASH boosted: osmolarity/pH signals weak in soil\n"
    "GC_SAAV_MAX         = float(_os.environ.get('GC_SAAV_MAX',       '0.5'))    # nA: max SAAV injection at full satiation\n"
)
if old in src:
    src = src.replace(old, new, 1)
    print("✓ Replaced GC_AWA_SCALE / GC_SENSORY_SCALE / GC_ASH_SCALE / GC_SAAV_MAX")
else:
    errors.append("✗ Could not find module-level GC_AWA_SCALE block")

# ── 2. Replace GC_REVERSAL_SCALE and add HEAD_CPG_AMP + K_PROPRIO as globals ──
old = "GC_REVERSAL_SCALE   = 0.0005 # reversal prob per mV AVA depolarisation above baseline"
new = (
    "GC_REVERSAL_SCALE   = float(_os.environ.get('GC_REVERSAL_SCALE', '0.0005'))  # reversal prob per mV AVA depolarisation above baseline\n"
    "HEAD_CPG_AMP        = float(_os.environ.get('HEAD_CPG_AMP',       '0.13'))   # CPG oscillation amplitude into DB/VB neurons\n"
    "K_PROPRIO           = float(_os.environ.get('K_PROPRIO',          '0.02'))   # proprioceptive curvature feedback gain (nA/rad)"
)
if old in src:
    src = src.replace(old, new, 1)
    print("✓ Replaced GC_REVERSAL_SCALE, added HEAD_CPG_AMP + K_PROPRIO as globals")
else:
    errors.append("✗ Could not find GC_REVERSAL_SCALE")

# ── 3. Replace DDVD_ICLAMP_MAX ──
old = "DDVD_ICLAMP_MAX = 1.0   # nA — max IClamp when drivers fully active"
new = "DDVD_ICLAMP_MAX = float(_os.environ.get('DDVD_ICLAMP_MAX', '1.0'))   # nA — max IClamp when drivers fully active"
if old in src:
    src = src.replace(old, new, 1)
    print("✓ Replaced DDVD_ICLAMP_MAX")
else:
    errors.append("✗ Could not find DDVD_ICLAMP_MAX")

# ── 4. Remove local HEAD_CPG_AMP definition (now reads from global) ──
old = "        HEAD_CPG_AMP  = 0.13\n"
new = "        # HEAD_CPG_AMP read from module-level global (env-overridable)\n"
if old in src:
    src = src.replace(old, new, 1)
    print("✓ Removed local HEAD_CPG_AMP definition")
else:
    errors.append("✗ Could not find local HEAD_CPG_AMP = 0.13")

# ── 5. Remove local K_PROPRIO definition (now reads from global) ──
old = "        K_PROPRIO = 0.02\n"
new = "        # K_PROPRIO read from module-level global (env-overridable)\n"
if old in src:
    src = src.replace(old, new, 1)
    print("✓ Removed local K_PROPRIO definition")
else:
    errors.append("✗ Could not find local K_PROPRIO = 0.02")

if errors:
    print("\nErrors — file NOT written:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

with open(TARGET, 'w') as f:
    f.write(src)

print("\nFile written. Verifying all 8 params are now env-overridable:")
result = subprocess.run(
    ['grep', '-n', 'environ.get', TARGET],
    capture_output=True, text=True
)
print(result.stdout)
