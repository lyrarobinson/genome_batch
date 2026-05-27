# SOIL — genome batch simulation
C. elegans simulation stack for genome parameter sweep.

## Setup on a new machine (Pop!_OS)

### 0. System dependencies
```bash
sudo apt-get update
sudo apt-get install -y git python-is-python3 openjdk-19-jdk
```

### 1. Install miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, allow conda init, then restart terminal
```

### 2. Clone the repo
```bash
git clone https://github.com/lyrarobinson/genome_batch.git
cd genome_batch
```

### 3. Install conda environment
NEURON and all other dependencies are included in environment.yml.
```bash
conda env create -f environment.yml
conda activate worm
```

### 4. Compile NEURON .mod files
Must be run once on each new machine before any simulations.
```bash
cd simulations/B_Full_2026-03-03_16-22-03/
nrnivmodl .
cd ../..
```

### 5. Test single simulation
```bash
HDF5_USE_FILE_LOCKING=FALSE python3 -u worm_kinematic_sim_graded.py \
    --sim_dir simulations/B_Full_2026-03-03_16-22-03 \
    --duration 10 \
    --env_step_ms 1.0 \
    --start_x 680 --start_z 270 \
    --nacl_seed 42 --colony_seed 7 \
    --env_cache $(pwd)/env_warmup_cache.npz \
    --log_every 100 \
    --checkpoint_every 0
```

If you see `Done. 10.0s simulated in ...` the setup is complete.

## Notes
- `HDF5_USE_FILE_LOCKING=FALSE` is required for all runs, especially parallel ones
- The compiled `x86_64/` directory is excluded from git — run `nrnivmodl` on each machine
- `env_warmup_cache.npz` is pre-generated and included in the repo root — do not delete or move
- Never delete `simulations/B_Full_2026-03-03_16-22-03/` — this is the compiled connectome
