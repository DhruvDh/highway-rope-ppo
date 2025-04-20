#! /bin/bash

source cluster-setup.sh
cd "$SLURM_SUBMIT_DIR"
uv run main.py --generate-slurm --generate-slurm --slurm-gpus 4 --slurm-time 48:00:00
sbatch slurm_jobs/experiments_array.slurm