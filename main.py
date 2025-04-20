import argparse
from joblib import Parallel, delayed

from config.base_config import HIGHWAY_CONFIG
from utils.reproducibility import SEED
from utils.logging_utils import setup_master_logger, ensure_artifacts_dir
from experiments.config import (
    Experiment,
    Condition,
    ConditionHP,
    CommonHP,
    expand_condition_hps,
)
from experiments.runner import ExperimentRunner
from utils.device_pool import DevicePool
from utils.slurm import emit_slurm_array
from collections import defaultdict
import warnings
import os
import math

warnings.filterwarnings("ignore", ".*Overriding environment .* already in registry.*")

# Shared pool and runner for experiment execution
pool = DevicePool()
runner = ExperimentRunner(HIGHWAY_CONFIG, pool)


def summarize(results):
    """Print the best average reward and configuration per condition."""
    best = defaultdict(lambda: (-1e9, ""))
    for r in results:
        cond = r["experiment_name"].split("_")[0]
        avg = r["avg_rewards"][-1]
        if avg > best[cond][0]:
            best[cond] = (avg, r["experiment_name"])
    print("\n=== BEST HP PER CONDITION ===")
    for c, (score, name) in best.items():
        print(f"{c:17} {score:7.2f}  {name}")


def define_experiments(base_seed=SEED, num_seeds=3):
    """Create a list of Experiment objects for each seed and hyperparameter combination."""
    experiments = []
    common_hps = CommonHP()
    # Baseline conditions (include in hyperparameter sweep as well)
    hp_sorted = ConditionHP(**vars(common_hps))
    hp_shuffled = ConditionHP(**vars(common_hps))
    # Hyperparameter sweep setup
    sweep_dict = {
        "lr": [3e-4],  # pilot winner
        "hidden_dim": [256, 384, 512],  # coarse MLP size check
        "clip_eps": [0.2],  # default PPO
        "entropy_coef": [0.005],  # explore lower & default
        "epochs": [8],  # pilot sweet‑spot
        "batch_size": [32, 64],  # stable mid‑range
        "d_embed": [4, 8, 16],  # novel axis
    }
    # Attach sweep to all conditions for full-grid runs
    hp_sorted.sweep = sweep_dict.copy()
    hp_shuffled.sweep = sweep_dict.copy()
    hp_rank = ConditionHP(**vars(common_hps))
    hp_rank.sweep = sweep_dict.copy()
    hp_dist = ConditionHP(**vars(common_hps))
    hp_dist.sweep = sweep_dict.copy()
    hp_rope = ConditionHP(**vars(common_hps))
    hp_rope.sweep = sweep_dict.copy()
    cond_to_hp = {
        Condition.SORTED: hp_sorted,
        Condition.SHUFFLED: hp_shuffled,
        Condition.SHUFFLED_RANKPE: hp_rank,
        Condition.SHUFFLED_DISTPE: hp_dist,
        Condition.SHUFFLED_ROPE: hp_rope,
    }
    for cond, hp_template in cond_to_hp.items():
        for hp in expand_condition_hps(hp_template):
            for i in range(num_seeds):
                seed = base_seed + i * 1000
                # Construct unique experiment name
                name_parts = [cond.name.lower()]
                for key in hp_template.sweep.keys():
                    val = getattr(hp, key)
                    name_parts.append(f"{key}{val}")
                name_parts.append(f"seed{seed}")
                exp_name = "_".join(name_parts)
                experiments.append(Experiment(exp_name, cond, hp, seed))
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Highway PPO Experiments")
    parser.add_argument(
        "--generate-slurm", action="store_true", help="Generate SLURM scripts."
    )
    parser.add_argument(
        "--run-single-experiment",
        type=str,
        default=None,
        help="Experiment name to run.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel jobs (-1 all devices)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="Num seeds per condition"
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="GPU", help="SLURM partition name"
    )
    parser.add_argument("--slurm-gpus", type=int, default=1, help="GPUs per node/task")
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=8,
        help="CPUs per node/task for parallel workers",
    )
    parser.add_argument(
        "--slurm-num-tasks",
        type=int,
        default=None,
        help="Number of SLURM array tasks (batches)",
    )
    parser.add_argument(
        "--slurm-max-concurrent",
        type=int,
        default=None,
        help="Max concurrent SLURM tasks",
    )
    parser.add_argument(
        "--slurm-mem-per-gpu", type=str, default="2G", help="Memory per GPU"
    )
    parser.add_argument(
        "--slurm-time", type=str, default="04:00:00", help="Walltime for SLURM jobs"
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="ID of the current SLURM array task (batch)",
    )
    parser.add_argument(
        "--get-total-experiments",
        action="store_true",
        help="Print total number of experiments and exit",
    )
    parser.add_argument(
        "--num-cpus-per-task",
        type=int,
        default=None,
        help="CPUs allocated to this task",
    )
    args = parser.parse_args()
    # Allow SLURM-provided CPU allocation to override job count
    ensure_artifacts_dir()
    master_logger = setup_master_logger()
    ALL_EXPTS = define_experiments(SEED, args.num_seeds)
    if args.get_total_experiments:
        print(len(ALL_EXPTS))
        exit(0)
    if args.generate_slurm:
        master_logger.info("Generating SLURM array script...")
        total_experiments = len(ALL_EXPTS)
        if args.slurm_num_tasks:
            num_tasks = args.slurm_num_tasks
        else:
            exp_per_task = args.slurm_cpus
            num_tasks = math.ceil(total_experiments / exp_per_task)
            master_logger.info(
                f"Calculated num_tasks={num_tasks} based on {total_experiments} experiments and CPUs per task {args.slurm_cpus}."
            )
        emit_slurm_array(
            n_tasks=num_tasks,
            partition=args.slurm_partition,
            gpus=args.slurm_gpus,
            cpus_per_task=args.slurm_cpus,
            mem_per_gpu=args.slurm_mem_per_gpu,
            time=args.slurm_time,
            python_script="main.py",
            max_concurrent_tasks=args.slurm_max_concurrent,
        )
        master_logger.info(f"SLURM array script generated for {num_tasks} tasks.")
        exit(0)
    else:
        # Determine jobs and experiment subset
        if args.array_task_id is not None:
            # SLURM array execution
            n_jobs = args.num_cpus_per_task or int(os.getenv("SLURM_CPUS_PER_TASK", 1))
            master_logger.info(f"SLURM run detected. Using n_jobs = {n_jobs}.")
            num_tasks = args.slurm_num_tasks or int(
                os.getenv("SLURM_ARRAY_TASK_COUNT", 1)
            )
            exps_per_task = math.ceil(len(ALL_EXPTS) / num_tasks)
            start = args.array_task_id * exps_per_task
            end = min(start + exps_per_task, len(ALL_EXPTS))
            if start >= len(ALL_EXPTS):
                master_logger.warning(
                    f"Task ID {args.array_task_id} has no experiments to run."
                )
                exps = []
            else:
                exps = ALL_EXPTS[start:end]
                master_logger.info(
                    f"SLURM Task {args.array_task_id}/{num_tasks - 1}: Running experiments {start} to {end - 1}."
                )
        elif args.run_single_experiment:
            # Local single experiment
            matches = [e for e in ALL_EXPTS if e.name == args.run_single_experiment]
            if not matches:
                matches = [
                    e
                    for e in ALL_EXPTS
                    if e.name.startswith(args.run_single_experiment)
                ]
            if len(matches) != 1:
                master_logger.error(
                    f"Experiment '{args.run_single_experiment}' selection ambiguous or not found."
                )
                exit(1)
            exps = matches
            n_jobs = 1
            master_logger.info(f"Running single experiment: {exps[0].name}")
        else:
            # Local full run
            exps = ALL_EXPTS
            n_devs = max(len(pool.devices), 1)
            n_jobs = n_devs
            master_logger.info(f"Local run. Using n_jobs = {n_jobs}.")
        master_logger.info(f"Launching {len(exps)} experiments with n_jobs={n_jobs}...")
        # Launch experiments
        launcher = runner.launch
        if n_jobs == 1:
            # Serial execution with shared runner
            results = [launcher(exp) for exp in exps]
        else:
            # Parallel execution with shared runner
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(launcher)(exp) for exp in exps
            )
        succ = sum(1 for r in results if r.get("status") == "COMPLETED")
        fail = len(results) - succ
        master_logger.info(f"Summary: {succ} succeeded, {fail} failed.")
        # Print best hyperparameter per condition
        summarize(results)
