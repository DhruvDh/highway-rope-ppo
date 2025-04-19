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

## Flags

- `--n-jobs-per-task`: Number of parallel PPO workers each SLURM task should spawn (overrides `--n-jobs`).
- `OVERSUB` (env var): How many logical workers to time‑share per GPU (default `1`; set to `16` in the SLURM template).

## Notes

- The SLURM array runs one Python process per task (array index = experiment index), each of which will internally spawn multiple workers using Joblib and `DevicePool`.
- The `OVERSUB` environment variable controls the oversubscription factor for GPU time‑sharing in `DevicePool`.
