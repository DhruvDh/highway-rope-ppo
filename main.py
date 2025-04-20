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
        "--n-jobs-per-task", type=int, default=None, help="Parallel jobs per SLURM task"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="Num seeds per condition"
    )
    parser.add_argument("--slurm-partition", type=str, default="GPU")
    parser.add_argument("--slurm-gpus", type=int, default=1)
    parser.add_argument("--slurm-time", type=str, default="04:00:00")
    parser.add_argument(
        "--exp-index",
        type=int,
        default=None,
        help="(internal) index in experiment list (for SLURM array)",
    )
    args = parser.parse_args()
    # Override n_jobs with per-task setting if provided
    if getattr(args, "n_jobs_per_task", None) is not None:
        args.n_jobs = args.n_jobs_per_task
    ensure_artifacts_dir()
    master_logger = setup_master_logger()
    ALL_EXPTS = define_experiments(SEED, args.num_seeds)
    if args.generate_slurm:
        master_logger.info(
            f"Generating SLURM array script for {len(ALL_EXPTS)} experiments..."
        )
        emit_slurm_array(
            n_experiments=len(ALL_EXPTS),
            partition=args.slurm_partition,
            gpus=args.slurm_gpus,
            cpus_per_task=1,
            mem_per_gpu="1G",
            time=args.slurm_time,
            python_script="main.py",
        )
        master_logger.info("SLURM array script generated.")
    else:
        # Select experiments
        # Prevent using both exp-index and run-single-experiment together
        if args.exp_index is not None and args.run_single_experiment:
            master_logger.error("Cannot specify both --exp-index and --run-single-experiment.")
            exit(1)
        if args.exp_index is not None:
            exps = [ALL_EXPTS[args.exp_index]]
            n_jobs = args.n_jobs

        elif args.run_single_experiment:
            # Support exact and prefix matching for single-experiment selection
            exact = [e for e in ALL_EXPTS if e.name == args.run_single_experiment]
            if exact:
                exps = exact
            else:
                pref = [e for e in ALL_EXPTS if e.name.startswith(args.run_single_experiment)]
                if len(pref) == 1:
                    exps = pref
                elif len(pref) > 1:
                    master_logger.error(f"Ambiguous experiment prefix '{args.run_single_experiment}' matches: {[e.name for e in pref]}")
                    exit(1)
                else:
                    master_logger.error(f"Experiment '{args.run_single_experiment}' not found.")
                    exit(1)
            n_jobs = 1

        else:
            exps = ALL_EXPTS
            n_devs = max(len(pool.devices), 1)
            n_jobs = n_devs if args.n_jobs == -1 else min(args.n_jobs, n_devs)
        master_logger.info(f"Running {len(exps)} experiments with n_jobs={n_jobs}...")
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
