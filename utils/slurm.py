# utils/slurm.py
from pathlib import Path
from experiments.config import Experiment
import itertools  # placeholder to satisfy parser


def emit_slurm(
    exp: Experiment,
    partition="standard",
    gpus=1,
    cpus_per_task=4,
    mem="16G",
    time="04:00:00",
    conda_env="highway_rl",
    python_script="main.py",
    artifacts_dir="artifacts/highway-ppo",
):
    """Generates a SLURM batch script for a given experiment."""
    log_dir = Path(artifacts_dir) / "logs"
    slurm_dir = Path("slurm_jobs")
    log_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(exist_ok=True)

    output_log = log_dir / f"{exp.name}.out"
    error_log = log_dir / f"{exp.name}.err"
    script_path = slurm_dir / f"{exp.name}.slurm"

    content = f"""#!/bin/bash
#SBATCH --job-name={exp.name[:64]}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --output={output_log}
#SBATCH --error={error_log}

module purge
module load anaconda3
source activate {conda_env}
cd $SLURM_SUBMIT_DIR
srun python {python_script} --run-single-experiment "{exp.name}"
"""
    script_path.write_text(content)
    print(f"SLURM script generated: {script_path}")
    return script_path


def emit_slurm_array(
    n_experiments,
    partition="GPU",
    gpus=1,
    cpus_per_task=1,
    mem="48G",
    time="24:00:00",
    python_script="main.py",
    artifacts_dir="artifacts/highway-ppo",
):
    slurm_dir = Path("slurm_jobs"); slurm_dir.mkdir(exist_ok=True)
    log_dir   = Path(artifacts_dir) / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    script    = slurm_dir / "experiments_array.slurm"
    content = f"""#! /bin/bash
#SBATCH --job-name=HighwayPPO
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{n_experiments-1}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#
module purge
module load cuda/12.4
module load cudnn/9.0.0-cuda12
cd "$SLURM_SUBMIT_DIR"
uv venv --seed
uv sync
# Round‑robin GPU selection (oversubscribe OK)
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(( SLURM_ARRAY_TASK_ID % NUM_GPUS ))
fi
srun uv run {python_script} --exp-index $SLURM_ARRAY_TASK_ID
"""
    script.write_text(content)
    print(f"SLURM array script generated: {script}")
    return script
