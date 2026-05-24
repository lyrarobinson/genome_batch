# SOIL — genome batch simulation

C. elegans simulation stack for genome parameter sweep.

## Setup

### 1. Install conda environment
```bash
conda env create -f environment.yml
conda activate worm
```

### 2. Install NEURON
```bash
pip install neuron
```

### 3. Compile NEURON .mod files
```bash
cd simulations/B_Full_2026-03-03_16-22-03/
nrnivmodl .
cd ../..
```

### 4. Test single simulation
```bash
HDF5_USE_FILE_LOCKING=FALSE python3 -u worm_kinematic_sim_graded.py \
    --sim_dir simulations/B_Full_2026-03-03_16-22-03 \
    --duration 10 \
    --env_step_ms 1.0 \
    --start_x 680 --start_z 270 \
    --nacl_seed 42 --colony_seed 7 \
    --env_cache simulations/B_Full_2026-03-03_16-22-03/env_warmup_cache.npz \
    --log_every 100 \
    --checkpoint_every 0
```

## Notes
- `HDF5_USE_FILE_LOCKING=FALSE` is required for parallel runs
- The compiled x86_64/ directory is excluded from git — run nrnivmodl on each machine
- env_warmup_cache.npz is pre-generated and included — do not delete
- Never delete simulations/B_Full_2026-03-03_16-22-03/ — this is the compiled connectome
