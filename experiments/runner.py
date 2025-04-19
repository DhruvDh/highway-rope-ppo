# experiments/runner.py
import numpy as np
import torch
import time
import traceback

import gymnasium as gym
from .config import Experiment
from .wrappers import make_env
from utils.device_pool import DevicePool
from ppo.agent import PPOAgent
from training.routine import train_with_experiment_name
from utils.logging_utils import setup_experiment_logger
from utils.reproducibility import set_random_seeds


class ExperimentRunner:
    def __init__(self, base_env_config: dict, device_pool: DevicePool):
        self.base_config = base_env_config
        self.pool = device_pool

    def _create_agent(
        self, state_dim: int, action_dim: int, hp, logger, device: torch.device
    ):
        """Instantiate PPOAgent with hyperparameters and move to device."""
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=hp.lr,
            gamma=hp.gamma,
            lam=hp.lam,
            eps_clip=hp.clip_eps,
            value_coef=hp.value_coef,
            entropy_coef=hp.entropy_coef,
            max_grad_norm=hp.max_grad_norm,
            epochs=hp.epochs,
            batch_size=hp.batch_size,
            hidden_dim=hp.hidden_dim,
            logger=logger,
            device=device,
        )
        return agent

    def launch(self, exp: Experiment):
        """Sets up and runs a single experiment."""
        run_results = {"experiment_name": exp.name, "status": "FAILED"}
        start_time = time.time()
        logger = None
        try:
            with self.pool.acquire() as device:
                # Set reproducibility
                set_random_seeds(exp.seed)

                # Experiment logger
                logger = setup_experiment_logger(exp.name)
                logger.info(
                    f"[{exp.name}] Acquired device: {device} | Seed: {exp.seed}"
                )
                logger.info(
                    f"[{exp.name}] Condition: {exp.condition.name} | HPs: {exp.hp}"
                )
                if exp.env_config_overrides:
                    logger.info(
                        f"[{exp.name}] Env Overrides: {exp.env_config_overrides}"
                    )

                env = None
                try:
                    # Create and wrap environment
                    env = make_env(
                        exp.condition,
                        self.base_config,
                        d_embed=exp.hp.d_embed,
                        env_overrides=exp.env_config_overrides,
                    )
                    # Move wrappers to device if supported
                    if hasattr(env, "to") and callable(env.to):
                        env = env.to(device)
                        logger.debug(
                            f"[{exp.name}] Moved environment wrapper to {device}"
                        )

                    # Compute dimensions
                    obs_space = env.observation_space
                    act_space = env.action_space
                    if isinstance(obs_space, gym.spaces.Box):
                        state_dim = int(np.prod(obs_space.shape))
                    else:
                        raise TypeError(
                            f"Unsupported observation space: {type(obs_space)}"
                        )
                    if isinstance(act_space, gym.spaces.Box):
                        action_dim = act_space.shape[0]
                    else:
                        raise TypeError(f"Unsupported action space: {type(act_space)}")

                    logger.info(
                        f"[{exp.name}] state_dim={state_dim}, action_dim={action_dim}"
                    )

                    # Create agent
                    agent = self._create_agent(
                        state_dim, action_dim, exp.hp, logger, device
                    )

                    # Run training
                    logger.info(f"[{exp.name}] Starting training...")
                    rewards, avg_rewards, metrics = train_with_experiment_name(
                        env=env,
                        agent=agent,
                        max_episodes=exp.max_episodes,
                        target_reward=exp.target_reward,
                        log_interval=exp.extra.get("log_interval", 20),
                        eval_interval=exp.extra.get("eval_interval", 50),
                        steps_per_update=exp.hp.steps_per_update,
                        experiment_name=exp.name,
                        exp_seed=exp.seed,
                        logger=logger,
                    )

                    run_results.update(
                        {
                            "status": "COMPLETED",
                            "rewards": rewards,
                            "avg_rewards": avg_rewards,
                            "metrics_history": metrics,
                        }
                    )
                    logger.info(f"[{exp.name}] Training completed successfully.")
                except Exception as e:
                    logger.error(
                        f"[{exp.name}] Experiment execution failed!", exc_info=True
                    )
                    run_results["error_message"] = str(e)
                    run_results["error_traceback"] = traceback.format_exc()
                finally:
                    if env is not None:
                        env.close()
                        logger.debug(f"[{exp.name}] Environment closed.")
        except Exception as e:
            print(f"FATAL: DevicePool error for {exp.name}: {e}")
            run_results["error_message"] = str(e)
            run_results["error_traceback"] = traceback.format_exc()
        end_time = time.time()
        run_results["duration_seconds"] = end_time - start_time
        if logger:
            logger.info(
                f"[{exp.name}] Run finished. Status: {run_results['status']}. Duration: {run_results['duration_seconds']:.2f}s"
            )
        return run_results
