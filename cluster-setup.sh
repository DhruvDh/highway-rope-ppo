#! /bin/bash
# Common environment setup: load CUDA, cuDNN, create/activate venv and sync dependencies.

# 1. Module loads
module purge
module load cuda/12.4
module load cudnn/9.0.0-cuda12

# 2. Python venv via uv
uv venv --seed

# 3. Sync project dependencies
uv sync --extra cu124

# 4. Activate venv
#    uv run will implicitly activate, but exporting ensures subshells see it
export UV_VENV_ACTIVE=1

echo "Cluster node environment initialized."