#!/usr/bin/env python3
"""
sweep_genome_ranges.py
----------------------
Tests min and max of each genome parameter range to find
boundaries of valid behaviour. 16 sims total (2 per parameter).

Run from ~/projects/worm/genome_batch/:
    python3 sweep_genome_ranges.py

Results saved to: sweep_results.csv
"""

import os, csv, subprocess, time, re
import numpy as np
import h5py

# ── Configuration ─────────────────────────────────────────────────────────────

SIM_DIR       = 'simulations/B_Full_2026-03-03_16-22-03'
ENV_CACHE     = 'env_warmup_cache.npz'
DURATION      = 10.0
N_PARALLEL    = 4
RESULTS_CSV   = 'sweep_results.csv'

BASELINE = {
    'HEAD_CPG_AMP':      0.13,
    'K_PROPRIO':         0.02,
    'DDVD_ICLAMP_MAX':   1.0,
    'GC_AWA_SCALE':      0.5,
    'GC_ASH_SCALE':      65.0,
    'GC_REVERSAL_SCALE': 0.0005,
    'GC_SAAV_MAX':       0.5,
    'GC_SENSORY_SCALE':  0.02,
}

# (min, max) — intentionally wide, including broken behaviour
SWEEP_RANGES = {
    'HEAD_CPG_AMP':      (0.01,    0.40),
    'K_PROPRIO':         (0.001,   0.08),
    'DDVD_ICLAMP_MAX':   (0.1,     3.0),
    'GC_AWA_SCALE':      (0.05,    2.0),
    'GC_ASH_SCALE':      (5.0,     200.0),
    'GC_REVERSAL_SCALE': (0.00005, 0.005),
    'GC_SAAV_MAX':       (0.05,    2.0),
    'GC_SENSORY_SCALE':  (0.002,   0.10),
}

START_X       = 680.0
START_Z       = 270.0
START_HEADING = 180.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_env(param_name, param_value):
    env = os.environ.copy()
    env['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    for k, v in BASELINE.items():
        env[k] = str(v)
    env[param_name] = str(param_value)
    return env


def run_sim(param_name, param_value, env):
    stamp    = f"{param_name}_{param_value:.8g}".replace('.','p').replace('-','m')
    log_path = f'/tmp/sweep_{stamp}.log'
    cmd = [
        'python3', '-u', 'worm_kinematic_sim_graded.py',
        '--sim_dir',          SIM_DIR,
        '--duration',         str(DURATION),
        '--env_step_ms',      '1.0',
        '--start_x',          str(START_X),
        '--start_z',          str(START_Z),
        '--start_heading',    str(START_HEADING),
        '--nacl_seed',        '42',
        '--colony_seed',      '7',
        '--env_cache',        ENV_CACHE,
        '--log_every',        '100',
        '--checkpoint_every', '0',
    ]
    log_f = open(log_path, 'w')
    proc  = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    return proc, log_path, log_f


def find_h5_from_log(log_path):
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Output:\s+(/\S+)', line)
                if m:
                    h5 = os.path.join(m.group(1), 'kinematic_sim.h5')
                    return h5 if os.path.exists(h5) else None
    except Exception:
        pass
    return None


def extract_metrics(h5_path, expected_duration):
    try:
        with h5py.File(h5_path, 'r') as f:
            times = f['environment/times'][:]
            if len(times) < 10:
                return None

            actual_duration = float(times[-1])
            completed       = actual_duration >= expected_duration * 0.9

            neuron_act = f['worm/neuron_activity'][:]
            mean_act   = float(np.mean(neuron_act))
            std_act    = float(np.std(neuron_act))
            max_act    = float(np.max(np.abs(neuron_act)))
            flatline   = std_act < 0.001
            saturated  = max_act > 500.0

            nose       = f['worm/nose_position'][:]
            deltas     = np.diff(nose[:, [0, 2]], axis=0)
            dist       = float(np.sum(np.linalg.norm(deltas, axis=1)))
            mean_speed = float(np.mean(f['worm/speed'][:]))

            reversing  = f['steering/reversing'][:]
            rev_count  = int(np.sum(np.diff(reversing.astype(int)) > 0))
            rev_frac   = float(np.mean(reversing))

            db_act     = f['steering/db_activity'][:]
            vb_act     = f['steering/vb_activity'][:]
            wave_amp   = float(np.mean(np.abs(db_act - vb_act)))

            quiescent   = f['steering/quiescent'][:]
            quiesc_frac = float(np.mean(quiescent))

            return {
                'completed':        int(completed),
                'actual_duration':  round(actual_duration, 2),
                'mean_neuron_act':  round(mean_act, 4),
                'std_neuron_act':   round(std_act, 4),
                'max_neuron_act':   round(max_act, 2),
                'flatline':         int(flatline),
                'saturated':        int(saturated),
                'dist_travelled':   round(dist, 2),
                'mean_speed':       round(mean_speed, 4),
                'rev_count':        rev_count,
                'rev_frac':         round(rev_frac, 4),
                'wave_amp':         round(wave_amp, 6),
                'quiesc_frac':      round(quiesc_frac, 4),
            }
    except Exception as e:
        print(f"    [WARN] Could not read {h5_path}: {e}")
        return None


# ── CSV setup ─────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'param_name', 'param_value', 'baseline_value', 'bound',
    'completed', 'actual_duration',
    'mean_neuron_act', 'std_neuron_act', 'max_neuron_act',
    'flatline', 'saturated',
    'dist_travelled', 'mean_speed',
    'rev_count', 'rev_frac',
    'wave_amp', 'quiesc_frac',
    'wall_time_s', 'notes',
]

completed_keys = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            completed_keys.add((row['param_name'], row['bound']))
    print(f"Resuming — {len(completed_keys)} points already done.")

results_f = open(RESULTS_CSV, 'a', newline='')
writer    = csv.DictWriter(results_f, fieldnames=FIELDNAMES)
if not completed_keys:
    writer.writeheader()
    results_f.flush()


# ── Build job list (min and max for each parameter) ───────────────────────────

jobs = []
for param_name, (lo, hi) in SWEEP_RANGES.items():
    for val, bound in [(lo, 'min'), (hi, 'max')]:
        if (param_name, bound) not in completed_keys:
            jobs.append((param_name, val, bound))
        else:
            print(f"  Skipping {param_name} {bound} (already done)")

print(f"\n{len(jobs)} sweep points to run")
print(f"Running {N_PARALLEL} in parallel\n")


# ── Run sweep ─────────────────────────────────────────────────────────────────

for batch_start in range(0, len(jobs), N_PARALLEL):
    batch = jobs[batch_start: batch_start + N_PARALLEL]

    active = []
    for param_name, param_value, bound in batch:
        env = build_env(param_name, param_value)
        print(f"  Launching: {param_name} = {param_value} [{bound}]  (baseline={BASELINE[param_name]})")
        proc, log_path, log_f = run_sim(param_name, param_value, env)
        active.append((param_name, param_value, bound, proc, log_path, log_f, time.time()))
        time.sleep(2)

    print(f"  Waiting for {len(active)} sims...")

    for param_name, param_value, bound, proc, log_path, log_f, t_start in active:
        proc.wait()
        log_f.close()
        wall_time = round(time.time() - t_start, 1)

        h5_path = find_h5_from_log(log_path)
        notes   = ''

        if proc.returncode != 0:
            notes   = 'sim_crashed'
            metrics = None
        elif h5_path is None:
            notes   = 'no_h5_found'
            metrics = None
        else:
            metrics = extract_metrics(h5_path, DURATION)
            if metrics is None:
                notes = 'h5_invalid'
            elif metrics['saturated']:
                notes = 'voltage_explosion'
            elif metrics['flatline']:
                notes = 'flatline'
            elif not metrics['completed']:
                notes = 'incomplete'

        row = {
            'param_name':     param_name,
            'param_value':    param_value,
            'baseline_value': BASELINE[param_name],
            'bound':          bound,
            'wall_time_s':    wall_time,
            'notes':          notes,
        }
        if metrics:
            row.update(metrics)
        else:
            for k in FIELDNAMES:
                if k not in row:
                    row[k] = ''

        writer.writerow(row)
        results_f.flush()

        status = notes if notes else 'ok'
        dist   = metrics['dist_travelled'] if metrics else '?'
        revs   = metrics['rev_count']      if metrics else '?'
        amp    = round(metrics['wave_amp'], 5) if metrics else '?'
        print(f"    {param_name} [{bound}]={param_value}  status={status}  "
              f"dist={dist}  revs={revs}  wave_amp={amp}  wall={wall_time}s")

    print()

results_f.close()
print(f"Done. Results in {RESULTS_CSV}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n=== Summary ===")
with open(RESULTS_CSV) as f:
    rows = list(csv.DictReader(f))

for param_name in SWEEP_RANGES:
    param_rows = [r for r in rows if r['param_name'] == param_name]
    print(f"\n  {param_name} (baseline={BASELINE[param_name]}):")
    for r in sorted(param_rows, key=lambda x: x['bound']):
        status = r.get('notes', '') or 'ok'
        dist   = r.get('dist_travelled', '?')
        revs   = r.get('rev_count', '?')
        amp    = r.get('wave_amp', '?')
        print(f"    [{r['bound']}] {r['param_value']:>10}  status={status:<20}  "
              f"dist={dist}  revs={revs}  wave_amp={amp}")
