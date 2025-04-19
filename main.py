import argparse
import copy
from joblib import Parallel, delayed
import torch

from config.base_config import HIGHWAY_CONFIG
from utils.reproducibility import SEED
from utils.logging_utils import setup_master_logger, ensure_artifacts_dir
from experiments.config import Experiment, Condition, ConditionHP, CommonHP
from experiments.runner import ExperimentRunner
from utils.device_pool import DevicePool
from utils.slurm import emit_slurm


def define_experiments(base_seed=SEED, num_seeds=3):
    """Create a list of Experiment objects for each seed and condition."""
    experiments = []
    common_hps = CommonHP()
    # Define base HPs for each condition
    hp_sorted = ConditionHP(**vars(common_hps))
    hp_shuffled = ConditionHP(**vars(common_hps), lr=1.5e-4, batch_size=128)
    hp_rank_pe = ConditionHP(**vars(common_hps), d_embed=8, hidden_dim=256, lr=1e-4)
    hp_dist_pe = ConditionHP(**vars(common_hps), d_embed=8, hidden_dim=256, lr=1e-4)
    for i in range(num_seeds):
        seed = base_seed + i * 1000
        experiments.append(
            Experiment(f"sorted_seed{seed}", Condition.SORTED, hp_sorted, seed)
        )
        experiments.append(
            Experiment(f"shuffled_seed{seed}", Condition.SHUFFLED, hp_shuffled, seed)
        )
        experiments.append(
            Experiment(
                f"shuffled_rankPE_d8_seed{seed}",
                Condition.SHUFFLED_RANKPE,
                hp_rank_pe,
                seed,
            )
        )
        experiments.append(
            Experiment(
                f"shuffled_distPE_d8_seed{seed}",
                Condition.SHUFFLED_DISTPE,
                hp_dist_pe,
                seed,
            )
        )
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
    parser.add_argument("--slurm-partition", type=str, default="standard")
    parser.add_argument("--slurm-gpus", type=int, default=1)
    parser.add_argument("--slurm-time", type=str, default="04:00:00")
    args = parser.parse_args()
    ensure_artifacts_dir()
    master_logger = setup_master_logger()
    # Initialize device pool for experiment execution
    pool = DevicePool()
    ALL_EXPERIMENTS = define_experiments(SEED, args.num_seeds)
    if args.generate_slurm:
        master_logger.info(
            f"Generating SLURM scripts for {len(ALL_EXPERIMENTS)} experiments..."
        )
        for exp in ALL_EXPERIMENTS:
            emit_slurm(
                exp,
                partition=args.slurm_partition,
                gpus=args.slurm_gpus,
                time=args.slurm_time,
            )
        master_logger.info("SLURM scripts generated.")
    else:
        # Select experiments
        if args.run_single_experiment:
            exps = [e for e in ALL_EXPERIMENTS if e.name == args.run_single_experiment]
            if not exps:
                master_logger.error(
                    f"Experiment '{args.run_single_experiment}' not found."
                )
                exit(1)
            n_jobs = 1
        else:
            exps = ALL_EXPERIMENTS
            # Use pre-initialized pool for device management
            n_devs = len(pool.devices)
            n_jobs = n_devs if args.n_jobs == -1 else min(args.n_jobs, n_devs)
        # Instantiate shared ExperimentRunner
        runner = ExperimentRunner(HIGHWAY_CONFIG, pool)
        master_logger.info(f"Running {len(exps)} experiments with n_jobs={n_jobs}...")
        # Launch experiments
        if n_jobs == 1:
            # Serial execution with shared runner
            results = [runner.launch(exp) for exp in exps]
        else:
            # Parallel execution with shared runner
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(runner.launch)(exp) for exp in exps
            )
        succ = sum(1 for r in results if r.get("status") == "COMPLETED")
        fail = len(results) - succ
        master_logger.info(f"Summary: {succ} succeeded, {fail} failed.")
