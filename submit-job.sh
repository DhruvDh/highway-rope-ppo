#! /bin/bash

source cluster-setup.sh
cd "$SLURM_SUBMIT_DIR"

NUM_WORKERS_PER_NODE=64
GPUS_PER_NODE=4
# Maximum SLURM array tasks running concurrently
MAX_CONCURRENT_TASKS=10
# Compute total experiments and number of batches/tasks
TOTAL_EXPTS=$(uv run main.py --get-total-experiments | tail -n1)
NUM_ARRAY_TASKS=$(( (TOTAL_EXPTS + NUM_WORKERS_PER_NODE - 1) / NUM_WORKERS_PER_NODE ))
echo "Total experiments: $TOTAL_EXPTS, Array tasks: $NUM_ARRAY_TASKS"

uv run main.py --generate-slurm \
    --slurm-gpus $GPUS_PER_NODE \
    --slurm-cpus $NUM_WORKERS_PER_NODE \
    --slurm-time 48:00:00 \
    --slurm-num-tasks $NUM_ARRAY_TASKS \
    --slurm-max-concurrent $MAX_CONCURRENT_TASKS

sbatch slurm_jobs/experiments_array.slurm