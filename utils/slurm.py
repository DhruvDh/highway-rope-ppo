from pathlib import Path
from experiments.config import Experiment
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from jinja2.exceptions import UndefinedError


def emit_slurm(
    exp: Experiment,
    partition="GPU",
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
    n_experiments: int,
    partition: str = "GPU",
    gpus: int = 1,
    cpus_per_task: int = 1,
    mem_per_gpu: str = "1G",
    time: str = "24:00:00",
    python_script: str = "main.py",
    artifacts_dir: str = "artifacts/highway-ppo",
):
    """Render SLURM array script from template."""
    slurm_dir = Path("slurm_jobs")
    slurm_dir.mkdir(exist_ok=True)
    log_dir_path = Path(artifacts_dir) / "logs"
    log_dir_path.mkdir(parents=True, exist_ok=True)

    template_path = slurm_dir / "experiments_array.slurm.j2"
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    try:
        tmpl = env.get_template(template_path.name)
    except TemplateNotFound:
        raise RuntimeError(f"SLURM template not found at {template_path}")

    try:
        rendered = tmpl.render(
            n_experiments=n_experiments,
            partition=partition,
            gpus=gpus,
            cpus_per_task=cpus_per_task,
            mem_per_gpu=mem_per_gpu,
            time=time,
            python_script=python_script,
            log_dir=str(log_dir_path),
        )
    except UndefinedError as e:
        raise RuntimeError(f"Error rendering SLURM template: {e}")

    out_path = slurm_dir / "experiments_array.slurm"
    out_path.write_text(rendered)
    print(f"SLURM array script generated at {out_path}")
    return out_path
