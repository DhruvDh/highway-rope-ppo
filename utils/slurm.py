# utils/slurm.py
from pathlib import Path
import os
from experiments.config import Experiment


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
