#!/usr/bin/env python3
"""
run_genome_batch.py
-------------------
Batch simulation runner for SOIL genome parameter sweep.
Uses a deterministic Sobol sequence so genome indices are stable
across multiple runs — increasing N_GENOMES never resamples
previously computed genomes.

Run from ~/projects/worm/genome_batch/:
    python3 run_genome_batch.py

Configuration is at the top of this file.
Resume is automatic — completed runs are skipped.
"""

import os
import csv
import time
import subprocess
import numpy as np
import h5py
from scipy.stats import qmc

# ── Configuration ─────────────────────────────────────────────────────────────

# Hardware
N_PARALLEL        = 8       # simultaneous sims — adjust to hardware
                            # rule of thumb: 1 per CPU core, watch GPU memory

# Experiment
N_GENOMES         = 100     # total genome variants to sample
                            # increase later — already-completed genomes are skipped
N_RUNS_PER_GENOME = 3       # runs per genome with different start conditions
DURATION          = 30.0    # simulated seconds per run

# Simulation
ENV_STEP_MS       = 1.0
LOG_EVERY         = 100
NACL_SEED         = 42
COLONY_SEED       = 7
SIM_DIR           = 'simulations/B_Full_2026-03-03_16-22-03'

# Start position — outside colony zone (colony at x=540-780, z=150-390)
# Worms start either left (x=50-500) or right (x=800-950) of colony
START_Z_MIN       = 50
START_Z_MAX       = 490

# Genome parameter ranges
# Format: (min, max)
# Ranges are intentionally wide — includes degraded/broken behaviour
GENOME_PARAMS = {
    'HEAD_CPG_AMP':      (0.0001,  2.0),
    'K_PROPRIO':         (0.0001,  0.3),
    'DDVD_ICLAMP_MAX':   (0.001,   8.0),
    'GC_AWA_SCALE':      (0.001,   8.0),
    'GC_AWA_BASE':       (0.0001,  0.8),
    'GC_ASH_SCALE':      (0.1,     800.0),
    'EMA_ALPHA':         (0.00001, 0.1),
    'GC_SENSORY_SCALE':  (0.0001,  0.5),
}

# Paths
OUTPUT_DIR        = 'genome_runs'
COMPLETED_CSV     = 'completed.csv'
GENOME_TABLE_CSV  = 'genome_table.csv'
ENV_CACHE         = os.path.abspath('env_warmup_cache.npz')


# ── Sobol genome sampling ─────────────────────────────────────────────────────

def generate_genome_table(n_genomes, params):
    """
    Generate genome parameter sets using a scrambled Sobol sequence.
    Deterministic — same genome index always gives same parameters
    regardless of n_genomes. Safe to call multiple times with increasing
    n_genomes; indices 0..n-1 are always identical.
    """
    n_params = len(params)
    sampler  = qmc.Sobol(d=n_params, scramble=True, seed=42)
    samples  = sampler.random(n_genomes)  # (n_genomes, n_params) in [0,1]

    param_names = list(params.keys())
    rows = []
    for i, sample in enumerate(samples):
        row = {'genome_idx': i}
        for j, name in enumerate(param_names):
            lo, hi = params[name]
            # Log-scale sampling for parameters spanning orders of magnitude
            if hi / lo > 100:
                val = np.exp(np.log(lo) + sample[j] * (np.log(hi) - np.log(lo)))
            else:
                val = lo + sample[j] * (hi - lo)
            row[name] = round(float(val), 8)
        rows.append(row)
    return rows


def load_or_generate_genome_table(n_genomes, params):
    """
    Load existing genome table if present, extending it if n_genomes
    is larger than what was previously generated. Generates fresh if absent.
    """
    param_names = list(params.keys())
    fieldnames  = ['genome_idx'] + param_names

    if os.path.exists(GENOME_TABLE_CSV):
        with open(GENOME_TABLE_CSV) as f:
            existing = list(csv.DictReader(f))
        n_existing = len(existing)

        if n_existing >= n_genomes:
            print(f"Loaded {n_existing} genomes from {GENOME_TABLE_CSV} "
                  f"(using first {n_genomes})")
            return existing[:n_genomes]
        else:
            print(f"Extending genome table from {n_existing} to {n_genomes}...")
            all_rows = generate_genome_table(n_genomes, params)
            with open(GENOME_TABLE_CSV, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Genome table saved to {GENOME_TABLE_CSV}")
            return all_rows
    else:
        print(f"Generating {n_genomes} genome variants (Sobol sequence)...")
        all_rows = generate_genome_table(n_genomes, params)
        with open(GENOME_TABLE_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Genome table saved to {GENOME_TABLE_CSV}")
        return all_rows


# ── Start condition generation ────────────────────────────────────────────────

def get_start_conditions(genome_idx, run_idx):
    """
    Deterministic start position and heading from genome+run index.
    Worm starts outside colony zone (left or right region).
    """
    rng = np.random.default_rng(seed=genome_idx * 100 + run_idx)
    region   = rng.integers(0, 2)
    start_x  = float(rng.uniform(50, 500) if region == 0 else rng.uniform(800, 950))
    start_z  = float(rng.uniform(START_Z_MIN, START_Z_MAX))
    heading  = float(rng.uniform(0, 360))
    return start_x, start_z, heading


# ── HDF5 validation ───────────────────────────────────────────────────────────

def validate_h5(h5_path, expected_duration):
    """
    Check HDF5 is valid and sim reached expected duration.
    Returns (valid, notes).
    """
    if not os.path.exists(h5_path):
        return False, 'h5_missing'
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'environment/times' not in f:
                return False, 'h5_no_times'
            times = f['environment/times'][:]
            if len(times) < 10:
                return False, 'h5_too_short'
            actual = float(times[-1])
            if actual < expected_duration * 0.9:
                return False, f'incomplete_{actual:.1f}s'
            if 'worm/neuron_activity' not in f:
                return False, 'h5_no_activity'
            act = f['worm/neuron_activity'][:]
            if np.max(np.abs(act)) > 500.0:
                return False, 'voltage_explosion'
            if np.std(act) < 0.001:
                return False, 'flatline'
        return True, 'ok'
    except Exception as e:
        return False, f'h5_error_{str(e)[:30]}'


# ── Sim launcher ──────────────────────────────────────────────────────────────

def launch_sim(genome_idx, run_idx, genome, run_dir):
    """Launch a single sim process. Returns (proc, log_path, log_f)."""
    os.makedirs(run_dir, exist_ok=True)
    start_x, start_z, heading = get_start_conditions(genome_idx, run_idx)

    env = os.environ.copy()
    env['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    for k, v in genome.items():
        env[k] = str(v)

    log_path = os.path.join(run_dir, 'sim.log')
    cmd = [
        'python3', '-u', 'worm_kinematic_sim_graded.py',
        '--sim_dir',          SIM_DIR,
        '--duration',         str(DURATION),
        '--env_step_ms',      str(ENV_STEP_MS),
        '--start_x',          str(round(start_x, 2)),
        '--start_z',          str(round(start_z, 2)),
        '--start_heading',    str(round(heading, 2)),
        '--nacl_seed',        str(NACL_SEED),
        '--colony_seed',      str(COLONY_SEED),
        '--env_cache',        ENV_CACHE,
        '--log_every',        str(LOG_EVERY),
        '--checkpoint_every', '0',
    ]
    log_f = open(log_path, 'w')
    proc  = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    return proc, log_path, log_f


def find_h5_from_log(log_path):
    """Parse sim log to find output HDF5 path."""
    import re
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


# ── Completed run tracking ────────────────────────────────────────────────────

COMPLETED_FIELDNAMES = [
    'genome_idx', 'run_idx', 'status', 'notes',
    'start_x', 'start_z', 'start_heading',
    'h5_path', 'wall_time_s', 'timestamp',
]

def load_completed():
    completed = set()
    if os.path.exists(COMPLETED_CSV):
        with open(COMPLETED_CSV) as f:
            for row in csv.DictReader(f):
                completed.add((int(row['genome_idx']), int(row['run_idx'])))
    return completed


def write_completed(writer, f, genome_idx, run_idx, status, notes,
                    start_x, start_z, heading, h5_path, wall_time):
    row = {
        'genome_idx':    genome_idx,
        'run_idx':       run_idx,
        'status':        status,
        'notes':         notes,
        'start_x':       round(start_x, 2),
        'start_z':       round(start_z, 2),
        'start_heading': round(heading, 2),
        'h5_path':       h5_path or '',
        'wall_time_s':   round(wall_time, 1),
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    writer.writerow(row)
    f.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load or generate genome table
    genomes = load_or_generate_genome_table(N_GENOMES, GENOME_PARAMS)

    # Load completed runs
    completed = load_completed()
    print(f"{len(completed)} runs already completed")

    # Build job list
    jobs = []
    for genome_row in genomes:
        genome_idx = int(genome_row['genome_idx'])
        genome     = {k: genome_row[k] for k in GENOME_PARAMS}
        for run_idx in range(N_RUNS_PER_GENOME):
            if (genome_idx, run_idx) not in completed:
                jobs.append((genome_idx, run_idx, genome))

    total_runs = N_GENOMES * N_RUNS_PER_GENOME
    print(f"{len(jobs)} runs remaining of {total_runs} total "
          f"({N_GENOMES} genomes × {N_RUNS_PER_GENOME} runs)")
    print(f"Running {N_PARALLEL} in parallel, {DURATION}s simulated each\n")

    if not jobs:
        print("Nothing to do.")
        return

    # Open completed CSV for appending
    write_header = not os.path.exists(COMPLETED_CSV)
    completed_f  = open(COMPLETED_CSV, 'a', newline='')
    writer       = csv.DictWriter(completed_f, fieldnames=COMPLETED_FIELDNAMES)
    if write_header:
        writer.writeheader()
        completed_f.flush()

    # Run in parallel batches
    n_done = 0
    for batch_start in range(0, len(jobs), N_PARALLEL):
        batch = jobs[batch_start: batch_start + N_PARALLEL]

        active = []
        for genome_idx, run_idx, genome in batch:
            run_dir = os.path.join(OUTPUT_DIR,
                                   f'genome_{genome_idx:04d}',
                                   f'run_{run_idx}')
            start_x, start_z, heading = get_start_conditions(genome_idx, run_idx)
            proc, log_path, log_f = launch_sim(genome_idx, run_idx, genome, run_dir)
            active.append((genome_idx, run_idx, genome, run_dir,
                           start_x, start_z, heading, proc, log_path, log_f, time.time()))
            print(f"  Launched genome_{genome_idx:04d}/run_{run_idx} "
                  f"PID={proc.pid} "
                  f"start=({start_x:.0f},{start_z:.0f}) heading={heading:.0f}°")
            time.sleep(2)  # stagger to avoid HDF5 timestamp collision

        print(f"  Waiting for batch of {len(active)}...")

        for (genome_idx, run_idx, genome, run_dir,
             start_x, start_z, heading, proc, log_path, log_f, t_start) in active:

            proc.wait()
            log_f.close()
            wall_time = time.time() - t_start

            h5_path = find_h5_from_log(log_path)
            if proc.returncode != 0:
                status, notes = 'failed', 'sim_crashed'
            elif h5_path is None:
                status, notes = 'failed', 'no_h5_found'
            else:
                valid, notes = validate_h5(h5_path, DURATION)
                status = 'completed' if valid else 'failed'

            write_completed(writer, completed_f,
                           genome_idx, run_idx, status, notes,
                           start_x, start_z, heading, h5_path, wall_time)

            n_done += 1
            remaining = len(jobs) - n_done
            print(f"  genome_{genome_idx:04d}/run_{run_idx} → {status} "
                  f"[{notes}] {wall_time:.0f}s  "
                  f"({n_done}/{len(jobs)} done, {remaining} remaining)")

        print()

    completed_f.close()
    print(f"Batch complete. {n_done} runs processed.")
    print(f"Results in {COMPLETED_CSV}")
    print(f"HDF5 outputs in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()