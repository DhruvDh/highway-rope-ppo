#!/bin/bash
#SBATCH --job-name=HighwayHP
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --exclusive              # no other user can land on the node
#SBATCH --gres=gpu:4             # GPUs per task
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1                # one srun per array task
#SBATCH --mem-per-gpu=2G
#SBATCH --array=0-0%1
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/highway-ppo/logs/%x_%A_%a.out
#SBATCH --error=artifacts/highway-ppo/logs/%x_%A_%a.err
#SBATCH --hint=nomultithread     # pin 1 task per CPU core

module purge
module load cuda/12.4 cudnn/9.0.0-cuda12

# Oversubscribe: 16 workers per GPU for time-sharing
export OVERSUB=16
export OMP_NUM_THREADS=1

# Each SLURM array task runs one Python worker with parallel jobs per task
echo "Running SLURM Task ID: $SLURM_ARRAY_TASK_ID on Node: $SLURMD_NODENAME with $SLURM_CPUS_PER_TASK CPUs and $SLURM_GPUS_ON_NODE GPUs"
srun --cpu-bind=cores uv run main.py --array-task-id $SLURM_ARRAY_TASK_ID --num-cpus-per-task $SLURM_CPUS_PER_TASK
