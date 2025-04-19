#! /bin/bash

source cluster-setup.sh
cd "$SLURM_SUBMIT_DIR"
uv run main.py --generate-slurm
sbatch slurm_jobs/experiments_array.slurm