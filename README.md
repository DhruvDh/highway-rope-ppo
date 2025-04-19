# highway-rope-ppo

A GPU‑parallel PPO runner for highway‑env experiments with positional embedding sweeps.

## Quickstart

1. Create and activate your Python venv (via `uv venv --seed && uv sync`).
2. Generate the SLURM array script for the full hyperparameter sweep (use `--slurm-gpus 4` on a 4‑GPU node):

```bash
uv run main.py --generate-slurm --slurm-gpus 4 --n-jobs-per-task 64
```

3. Submit the job array:

```bash
sbatch slurm_jobs/experiments_array.slurm
```

## SLURM Script Generation

Generate the SLURM array script with custom resources:

```bash
uv run main.py --generate-slurm \
  --slurm-gpus 4 \   # GPUs per task (maps to --gres=gpu)
  --slurm-partition GPU \  # SLURM partition name
  --slurm-time 48:00:00 \  # Walltime per task
  --n-jobs-per-task 64    # Number of internal PPO workers per task (time-shared via OVERSUB)
```

This command writes the `slurm_jobs/experiments_array.slurm` template, filling in:

- `--gres=gpu:{{ gpus }}` with the `--slurm-gpus` value.
- `--cpus-per-task={{ cpus_per_task }}` from the CLI (default 1).
- `--mem-per-gpu={{ mem_per_gpu }}` from the CLI (default "1G").
- `--array=0-{{ n_experiments - 1 }}` to cover all experiments.

Review the generated script before submitting with `sbatch slurm_jobs/experiments_array.slurm` to ensure it matches your cluster policies and resource limits.

## Flags

- `--n-jobs-per-task`: Number of parallel PPO workers each SLURM task should spawn (overrides `--n-jobs`).
- `OVERSUB` (env var): How many logical workers to time‑share per GPU (default `1`; set to `16` in the SLURM template).

## Notes

- The SLURM array runs one Python process per task (array index = experiment index), each of which will internally spawn multiple workers using Joblib and `DevicePool`.
- The `OVERSUB` environment variable controls the oversubscription factor for GPU time‑sharing in `DevicePool`.
