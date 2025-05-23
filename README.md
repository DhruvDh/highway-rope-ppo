# highway-rope-ppo

A GPU‑parallel PPO runner for highway‑env experiments with positional embedding sweeps.

## TODO

- [ ] Bigger set size: raise vehicles_count / vehicles_count observed to 30. Larger permutations are harder to ignore.

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

## Midterm Pilot Analysis

> Data from commit `09e104edb618e3d3035611bd2b2bd4cbb15062c7`, `artifacts/combined_validated_data.csv`

To refine the hyperparameter sweep for the final positional embedding experiments, the midterm dataset (162 runs) was re-analyzed, focusing only on runs matching the fixed hyperparameters intended for the final sweep:

- `features = 'x,y,vx,vy'`
- `learning_rate = 3e-4`
- `epochs = 8`

*(Note: `clip_eps=0.2` and `entropy_coef=0.005` from the pilot analysis could not be used for filtering as they were not encoded in the experiment names in the dataset).*

This subset contained 9 runs. Key insights from this filtered analysis:

- **Baseline Confirmation:** This configuration represents a strong starting point, with an average reward of **120.00 ± 14.10** across these 9 runs.
- **Hidden Dimension:** Performance increased with `hidden_dim` within this subset (`64`: 112.04, `128`: 122.08, `256`: **125.88**). This supports exploring dimensions ≥ 256 in the final sweep (`[256, 384, 512]`).
- **Batch Size:** Results were mixed compared to the full dataset analysis:
  - `batch_size=64` achieved the highest average reward (**125.15**) but had very high variance (± 25.93).
  - `batch_size=32` was more stable (120.10 ± 5.93).
  - The single best run in this subset used `hidden_dim=256` and `batch_size=64`, reaching a reward of **142.38**.
  - This justifies including both `[32, 64]` in the final sweep to investigate this trade-off further.

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

- **Features:** `["x", "y", "vx", "vy"]` (relative position & velocity)
- **Learning Rate:** `3e-4`
- **PPO Clip Epsilon:** `0.2`
- **Entropy Coefficient:** `0.005`
- **Epochs per Update:** `8`

The following hyperparameters are **swept** in the final run:

- **Hidden Dimension:** `[256, 384, 512]` (Network size)
- **Batch Size:** `[32, 64]` (Mini-batch size for PPO updates)
- **Embedding Dimension (`d_embed`):** `[4, 8, 16]` (Size of the positional embedding vector, applicable only to PE conditions)

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

Each hyper-parameter combination is repeated over several random seeds under the five **experimental conditions** (`experiments/config.py:Condition`):

- **`SORTED`** – baseline; rows already sorted by distance (nearest-first).  
  *No positional embedding added.*

- **`SHUFFLED`** – rows are reshuffled every timestep; identical feature set to `SORTED`.  
  *Pure permutation-noise baseline.*

- **`SHUFFLED_RANKPE`** – same shuffle as above **plus** a `d_embed`-wide **row-index tag** produced by a frozen lookup table (`tanh(Embedding[N,d])`).  
  *Acts as a dimensionality control: adds channels but conveys no geometric information.*

- **`SHUFFLED_DISTPE`** – shuffled rows **plus** a sinusoidal distance code:  
  `sin/cos(2 π · ‖x_i − x_ego‖ / MAX_DIST · freq_k)` for each vehicle.  
  *Injects explicit relative-distance signal.*

- **`SHUFFLED_ROPE`** – shuffled rows **plus** Rotary Positional Embedding: the first `rotate_dim` features of every row are rotated in 2-D planes by an angle proportional to distance.  
  *Mixes positional information into existing channels without widening the tensor.*

## Positional-Embedding Conditions

Below is a conceptual sketch of what happens to the **N × F** observation
matrix *before* it reaches the policy for each experimental condition.
(Variables: `obs` = current observation, `N` = rows/vehicles, `F` = base
feature width, `d` = chosen `d_embed`, `ego` = row 0 after any shuffle.)

### 1. `SORTED`  — baseline

```python
# highway-env already orders rows by nearest-first
return obs                     # shape: (N, F)
```

### 2. `SHUFFLED`  — permutation noise

```python
obs = obs[random_permutation(N)]   # shuffle rows every step
return obs                         # shape: (N, F)
```

### 3. `SHUFFLED_RANKPE`  — row-index tags (noise control)

```python
obs = obs[random_permutation(N)]           # same shuffle as above

rank_vec = tanh(rank_embedding_table[N, d])  # frozen at init, never trained
# -- table lives in the Gym wrapper, so PPO gradients do not reach it

return concat(obs, rank_vec, axis=1)        # shape: (N, F + d)
```

*Purpose: adds the same number of channels as DistPE while conveying no
distance information—tests whether "just wider input" helps.*

### 4. `SHUFFLED_DISTPE`  — sinusoidal distance codes

```python
obs = obs[random_permutation(N)]

dist = l2_norm(obs[:, :2] - obs[ego, :2])      # metres
norm = clip(dist / MAX_DIST, 0, 1)             # ∈ [0,1]

freqs = exp(-arange(0, d, 2) * ln(MAX_DIST)/d) # one freq per sin/cos pair
angle = 2π * norm[:, None] * freqs[None, :]    # radians

sincos = concat(sin(angle), cos(angle), axis=1)
return concat(obs, sincos, axis=1)             # shape: (N, F + d)
```

### 5. `SHUFFLED_ROPE`  — rotary positional embedding

```python
obs = obs[random_permutation(N)]

dist = l2_norm(obs[:, :2] - obs[ego, :2])      # metres
norm = clip(dist / MAX_DIST, 0, 1)

pair_count = rotate_dim // 2
inv_freq = 1 / MAX_DIST ** (arange(pair_count)/pair_count)
theta = 2π * norm[:, None] * inv_freq[None, :] # (N, pair_count)

obs[:, :rotate_dim] = rope_rotate(obs[:, :rotate_dim], theta)
return obs                                      # shape: (N, F)  (unchanged)
```

**Quick takeaway**

- RankPE adds dimensionality but no useful signal → acts as a noise control.  
- DistPE injects an explicit geometric signal (distance to ego).  
- RoPE mixes distance into the first features by rotation, keeping width fixed.  
- In our 15-vehicle runs, none of the embeddings surpassed the plain shuffled baseline.

## Final Run Summary (N = 15)

```
❯ uv run analysis.py
Loading data from: artifacts/combined_validated_data-final-run.csv
Successfully loaded and parsed 270 records.


=============== Analysis on Full Midterm Dataset ===============

=== Overall metrics (Full Dataset) ===
       final_reward  max_reward  training_steps
count        270.00      270.00          270.00
mean         115.72      121.10       255508.16
std           14.71       12.98        14747.18
min           70.29       73.15       206200.00
25%          107.84      114.14       244962.00
50%          116.66      122.92       260896.00
75%          127.77      130.91       268040.00
max          141.46      144.02       275632.00

=== By features (Full Dataset) ===
Column 'features' not found for grouping.

=== By learning rate (Full Dataset) ===
          mean    std  count
lr                          
0.0003  115.72  14.71    270

=== By epochs/update (Full Dataset) ===
          mean    std  count
epochs                      
8       115.72  14.71    270

=== By hidden_dim (Full Dataset) ===
              mean    std  count
hidden_dim                      
256         119.33  16.39     90
384         113.68  14.74     90
512         114.14  12.16     90

=== By batch_size (Full Dataset) ===
              mean    std  count
batch_size                      
32          113.06  14.74    135
64          118.37  14.23    135


=============== Analysis for Fixed Final Config Subset ===============

Filtering criteria for this subset:
  - lr       = 0.0003
  - epochs   = 8
(Note: clip_eps and entropy_coef are not available in the loaded data for filtering)

Found 270 runs matching the fixed criteria.

--- Overall metrics (Filtered Subset) ---
       final_reward  max_reward  training_steps
count        270.00      270.00          270.00
mean         115.72      121.10       255508.16
std           14.71       12.98        14747.18
min           70.29       73.15       206200.00
25%          107.84      114.14       244962.00
50%          116.66      122.92       260896.00
75%          127.77      130.91       268040.00
max          141.46      144.02       275632.00

--- By hidden_dim (within Fixed Config) ---
              mean    std  count
hidden_dim                      
256         119.33  16.39     90
384         113.68  14.74     90
512         114.14  12.16     90

--- By batch_size (within Fixed Config) ---
              mean    std  count
batch_size                      
32          113.06  14.74    135
64          118.37  14.23    135

--- Best Performing Run (within Fixed Config Subset) ---
  Experiment Name: shuffled_rope_lr0.0003_hidden_dim256_clip_eps0.2_entropy_coef0.005_epochs8_batch_size64_d_embed16_seed2042
  Final Reward:    141.46
  Hidden Dim:      256
  Batch Size:      64
```

## Results (15 observed vehicles)

![Final reward boxplot](./figures/box_final_reward.png)
![Episodes‑to‑120 boxplot](./figures/box_ep_to_thr.png)
![Hidden‑dim × PE heat‑map](./figures/heat_hidden_dim_vs_pe.png)
![Δ‑recovery vs ordering penalty](./figures/delta_recovery.png)
![AULC boxplot](./figures/box_auc.png)

### Key observations

- **Ordering penalty is small.**  
  Training with completely shuffled observations (no positional embedding) scored only ~2 points below the distance‑sorted baseline and reached the 120‑reward mark in roughly the same number of episodes.

- **Positional embeddings did not help — and sometimes hurt.**  
  *RankPE* and *RoPE* slowed learning by ≈100 episodes and finished at the same median reward as the plain shuffled baseline.  
  *DistPE* was consistently the worst performer in both speed and final score.

- **Model capacity dominated.**  
  A hidden dimension of **256** delivered the best final reward across *all* conditions. Increasing to 384 or 512 hid‑dim reduced performance unless RankPE was used – and even then, the gain was marginal.

- **Take‑away:** for \(N = 15\) vehicles our vanilla MLP appears robust to permutation noise; explicit positional signals are unnecessary.

- **No recovery of the (tiny) ordering gap.**  
  The dashed line in *delta_recovery.png* is the ~0 pt "gap" between sorted and shuffled.  All positional‑embedding bars are at or **below** zero – they add nothing, occasionally subtract.

- **Cumulative reward ranking mirrors speed ranking.**  
  The AULC plot shows `sorted ≈ shuffled ≫ shuffled_rankpe > shuffled_rope > shuffled_distpe`.  In other words, embeddings that slowed learning also harvested less total reward.

### Next step

The TODO list (above) already calls for doubling the observed set size.  
We will rerun the exact sweep with **30 vehicles** in the observation tensor to test whether larger permutations finally expose a benefit for learned or sinusoidal positional cues.
