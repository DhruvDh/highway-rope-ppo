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
  --slurm-gpus 4 \    # GPUs per task (maps to --gres=gpu)
  --slurm-partition GPU \  # SLURM partition name
  --slurm-time 48:00:00 \  # Walltime per task
  --n-jobs-per-task 64    # Number of internal PPO workers per task (time-shared via OVERSUB)
```

This command writes the `slurm_jobs/experiments_array.slurm` template, filling in:

* `--gres=gpu:{{ gpus }}` with the `--slurm-gpus` value.
* `--cpus-per-task={{ cpus_per_task }}` from the CLI (default 1).
* `--mem-per-gpu={{ mem_per_gpu }}` from the CLI (default "1G").
* `--array=0-{{ n_experiments - 1 }}` to cover all experiments.

Review the generated script before submitting with `sbatch slurm_jobs/experiments_array.slurm` to ensure it matches your cluster policies and resource limits.

## Flags

* `--n-jobs-per-task`: Number of parallel PPO workers each SLURM task should spawn (overrides `--n-jobs`).
* `OVERSUB` (env var): How many logical workers to time‑share per GPU (default `1`; set to `16` in the SLURM template).

## Notes

* The SLURM array runs one Python process per task (array index = experiment index), each of which will internally spawn multiple workers using Joblib and `DevicePool`.
* The `OVERSUB` environment variable controls the oversubscription factor for GPU time‑sharing in `DevicePool`.

## Midterm Pilot Analysis

> Data from commit `09e104edb618e3d3035611bd2b2bd4cbb15062c7`, `artifacts/combined_validated_data.csv`

To refine the hyperparameter sweep for the final positional embedding experiments, the midterm dataset (162 runs) was re-analyzed, focusing only on runs matching the fixed hyperparameters intended for the final sweep:

* `features = 'x,y,vx,vy'`
* `learning_rate = 3e-4`
* `epochs = 8`

*(Note: `clip_eps=0.2` and `entropy_coef=0.005` from the pilot analysis could not be used for filtering as they were not encoded in the experiment names in the dataset).*

This subset contained 9 runs. Key insights from this filtered analysis:

* **Baseline Confirmation:** This configuration represents a strong starting point, with an average reward of **120.00 ± 14.10** across these 9 runs.
* **Hidden Dimension:** Performance increased with `hidden_dim` within this subset (`64`: 112.04, `128`: 122.08, `256`: **125.88**). This supports exploring dimensions ≥ 256 in the final sweep (`[256, 384, 512]`).
* **Batch Size:** Results were mixed compared to the full dataset analysis:
  * `batch_size=64` achieved the highest average reward (**125.15**) but had very high variance (± 25.93).
  * `batch_size=32` was more stable (120.10 ± 5.93).
  * The single best run in this subset used `hidden_dim=256` and `batch_size=64`, reaching a reward of **142.38**.
  * This justifies including both `[32, 64]` in the final sweep to investigate this trade-off further.

These findings confirm the selection of fixed parameters and validate the chosen sweep ranges for `hidden_dim` and `batch_size` in the `final-run` experiments.

### Analysis Script Run

```bash
uv run analysis.py
```

```text
❯ uv run analysis.py
Loading data from: artifacts/combined_validated_data-midterm-progress.csv
Successfully loaded and parsed 162 records.


=============== Analysis on Full Midterm Dataset ===============

=== Overall metrics (Full Dataset) ===
        final_reward  max_reward  training_steps
count         162.00      162.00          162.00
mean          114.38      118.37       243233.81
std            23.05       21.99        25498.00
min            32.06       32.06       175750.00
25%           103.64      107.92       220667.50
50%           116.99      122.42       247872.50
75%           132.08      136.25       265531.00
max           142.85      144.35       278528.00

=== By features (Full Dataset) ===
                         mean    std  count
features
presence,x,y,vx,vy     112.94  27.04     54
x,y,vx,vy              115.92  15.91     54
x,y,vx,vy,cos_h,sin_h  114.28  25.00     54

=== By learning rate (Full Dataset) ===
          mean    std  count
lr
0.0001  112.93  25.82     81
0.0003  115.82  19.96     81

=== By epochs/update (Full Dataset) ===
        mean    std  count
epochs
4     112.83  23.40     54
6     112.11  25.66     54
8     118.19  19.63     54

=== By hidden_dim (Full Dataset) ===
              mean    std  count
hidden_dim
64          101.47  25.87     54
128         119.00  20.53     54
256         122.67  16.18     54

=== By batch_size (Full Dataset) ===
              mean    std  count
batch_size
32          119.43  14.17     54
64          111.58  26.77     54
128         112.12  25.64     54


=============== Analysis for Fixed Final Config Subset ===============

Filtering criteria for this subset:
  - features = 'x,y,vx,vy'
  - lr       = 0.0003
  - epochs   = 8
(Note: clip_eps and entropy_coef are not available in the loaded data for filtering)

Found 9 runs matching the fixed criteria.

--- Overall metrics (Filtered Subset) ---
        final_reward  max_reward  training_steps
count           9.00        9.00            9.00
mean          120.00      121.40       253736.22
std            14.10       14.15        18617.38
min            95.33       95.72       214192.00
25%           113.27      115.74       247658.00
50%           116.94      119.38       260096.00
75%           123.84      124.54       268040.00
max           142.38      142.38       268688.00

--- By hidden_dim (within Fixed Config) ---
              mean    std  count
hidden_dim
64          112.04  14.87      3
128         122.08  13.60      3
256         125.88  15.33      3

--- By batch_size (within Fixed Config) ---
              mean    std  count
batch_size
32          120.10   5.93      3
64          125.15  25.93      3
128         114.75   2.47      3

--- Best Performing Run (within Fixed Config Subset) ---
  Experiment Name: feat=x,y,vx,vy_epochs=8_lr=0.0003_hidden_dim=256_batch_size=64
  Final Reward:    142.38
  Hidden Dim:      256
  Batch Size:      64
```

## Final Experiment Setup

The final set of experiments focuses on evaluating the impact of different positional embedding techniques (`RankPE`, `DistPE`, `RoPE`) compared to baseline conditions (`SORTED`, `SHUFFLED`).

Based on the midterm pilot analysis, the following hyperparameters are **fixed** across all final runs:

* **Features:** `["x", "y", "vx", "vy"]` (relative position & velocity)
* **Learning Rate:** `3e-4`
* **PPO Clip Epsilon:** `0.2`
* **Entropy Coefficient:** `0.005`
* **Epochs per Update:** `8`

The following hyperparameters are **swept** in the final run:

* **Hidden Dimension:** `[256, 384, 512]` (Network size)
* **Batch Size:** `[32, 64]` (Mini-batch size for PPO updates)
* **Embedding Dimension (`d_embed`):** `[4, 8, 16]` (Size of the positional embedding vector, applicable only to PE conditions)

This results in the following `sweep_dict` used in `main.py`:

```python
# Hyperparameter sweep setup
sweep_dict = {
    "lr": [3e-4],             # pilot winner
    "hidden_dim": [256, 384, 512], # coarse MLP size check
    "clip_eps": [0.2],           # default PPO
    "entropy_coef": [0.005],     # pilot winner
    "epochs": [8],             # pilot sweet-spot
    "batch_size": [32, 64],      # stable mid-range based on filtered analysis
    "d_embed": [4, 8, 16],       # novel axis for PE experiments
}
```

Each combination of swept hyperparameters is run for multiple random seeds across the following **Experimental Conditions** (`experiments/config.py:Condition`):

* `SORTED`: Baseline, observations sorted by distance, no PE.
* `SHUFFLED`: Baseline, observations shuffled randomly, no PE.
* `SHUFFLED_RANKPE`: Shuffled observations + learnable Rank Positional Embedding.
* `SHUFFLED_DISTPE`: Shuffled observations + fixed Sinusoidal Distance Positional Embedding.
* `SHUFFLED_ROPE`: Shuffled observations + fixed Rotary Positional Embedding.
