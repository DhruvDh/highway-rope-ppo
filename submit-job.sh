#! /bin/bash

source cluster-setup.sh
cd "$SLURM_SUBMIT_DIR"

NUM_WORKERS_PER_NODE=32
GPUS_PER_NODE=4
NUM_ARRAY_TASKS=1
MAX_CONCURRENT_TASKS=1

uv run main.py --generate-slurm \
    --slurm-gpus $GPUS_PER_NODE \
    --slurm-cpus $NUM_WORKERS_PER_NODE \
    --slurm-time 48:00:00 \
    --slurm-num-tasks $NUM_ARRAY_TASKS \
    --slurm-max-concurrent $MAX_CONCURRENT_TASKS

sbatch slurm_jobs/experiments_array.slurm