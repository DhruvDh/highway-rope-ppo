# training/routine.py
import os
import time
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging
import torch

from utils.logging_utils import ensure_artifacts_dir, setup_experiment_logger
from utils.reproducibility import SEED


def evaluate(env, agent, num_episodes=10, render=False, exp_seed: int = 0):
    """Runs deterministic evaluation of the agent for a given number of episodes."""
    total_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=exp_seed + 1000 + ep)
        flat_state = state.reshape(-1)
        done = False
        episode_reward = 0.0
        while not done:
            action, _, _, _ = agent.select_action(flat_state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            flat_state = next_state.reshape(-1)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return float(np.mean(total_rewards))


def visualize_agent(env, agent, num_episodes=3, exp_seed: int = 0, logger=None):
    """Visualizes a trained agent in the environment (render_mode='human')."""
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("Visualizing agent...")
    # Ensure highway env registered
    if "highway-v0" not in gym.envs.registry:
        import highway_env

        highway_env._register_highway_envs()
    # Create a fresh env with human rendering
    viz_env = gym.make(env.spec.id, render_mode="human", config=env.config)
    try:
        for ep in range(num_episodes):
            state, _ = viz_env.reset(seed=exp_seed + 2000 + ep)
            flat_state = state.reshape(-1)
            done = False
            episode_reward = 0.0
            while not done:
                action, _, _, _ = agent.select_action(flat_state, deterministic=True)
                next_state, reward, terminated, truncated, _ = viz_env.step(action)
                done = terminated or truncated
                flat_state = next_state.reshape(-1)
                episode_reward += reward
            logger.info("viz_episode=%d reward=%.2f", ep + 1, episode_reward)
    finally:
        viz_env.close()


def train_with_experiment_name(
    env,
    agent,
    max_episodes=500,
    target_reward=0.0,
    log_interval=20,
    eval_interval=50,
    steps_per_update=2048,
    experiment_name="",
    exp_seed: int = 0,
    logger=None,
):
    """Train loop modified to include experiment_name for artifact naming."""
    # Setup logger
    if logger is None:
        logger = setup_experiment_logger(experiment_name)
    exp_prefix = f"[{experiment_name}]" if experiment_name else ""
    logger.info(f"{exp_prefix} Starting training for experiment: {experiment_name}")

    # Tracking variables
    rewards = []
    episode_rewards = []
    avg_rewards = []
    training_episodes = []
    eval_episodes = [0]
    best_avg_reward = -float("inf")

    metrics_history = {
        "experiment_name": experiment_name,
        "episode_rewards": [],
        "eval_rewards": [],
        "avg_eval_rewards": [],
        "policy_updates": [],
        "episode_numbers": [],
        "eval_episode_numbers": [],
        "timestamps": [],
    }

    solved = False
    start_time = time.time()
    total_steps = 0
    episode_num = 0

    artifacts_dir = ensure_artifacts_dir()

    # Initial evaluation
    logger.info(f"{exp_prefix} Performing initial evaluation...")
    init_reward = evaluate(env, agent, num_episodes=5, exp_seed=exp_seed)
    rewards.append(init_reward)
    avg_rewards.append(init_reward)
    metrics_history["eval_rewards"].append(init_reward)
    metrics_history["avg_eval_rewards"].append(init_reward)
    metrics_history["eval_episode_numbers"].append(0)
    metrics_history["timestamps"].append(0)
    logger.info(f"{exp_prefix} initial_eval reward={init_reward:.2f}")

    # Main training loop
    while episode_num < max_episodes:
        steps_collected = 0
        update_start = time.time()

        while steps_collected < steps_per_update and episode_num < max_episodes:
            episode_num += 1
            state, _ = env.reset(seed=exp_seed + episode_num)
            flat_state = state.reshape(-1)
            ep_reward = 0.0
            done = False

            while not done and steps_collected < steps_per_update:
                action, pre_tanh, log_prob, value = agent.select_action(flat_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                flat_next = next_state.reshape(-1)

                agent.memory.store(
                    flat_state,
                    action,
                    pre_tanh,
                    reward,
                    flat_next,
                    log_prob,
                    done,
                    value,
                )
                flat_state = flat_next
                ep_reward += reward
                steps_collected += 1
                total_steps += 1

            episode_rewards.append(ep_reward)
            training_episodes.append(episode_num)
            metrics_history["episode_rewards"].append(ep_reward)
            metrics_history["episode_numbers"].append(episode_num)

            # Logging
            if episode_num % log_interval == 0:
                recent_avg = np.mean(episode_rewards[-log_interval:])
                elapsed = time.time() - start_time
                logger.info(
                    "%s episode=%d reward=%.2f avg_reward=%.2f steps=%d time=%.2fs",
                    exp_prefix,
                    episode_num,
                    ep_reward,
                    recent_avg,
                    total_steps,
                    elapsed,
                )

            # Evaluation
            if episode_num % eval_interval == 0:
                logger.info(f"{exp_prefix} Evaluating at episode {episode_num}...")
                eval_reward = evaluate(env, agent, num_episodes=5, exp_seed=exp_seed)
                rewards.append(eval_reward)
                eval_episodes.append(episode_num)
                elapsed_eval = time.time() - start_time

                # Moving average
                avg_r = (
                    float(np.mean(rewards[-10:]))
                    if len(rewards) >= 10
                    else float(np.mean(rewards))
                )
                avg_rewards.append(avg_r)

                metrics_history["eval_rewards"].append(eval_reward)
                metrics_history["avg_eval_rewards"].append(avg_r)
                metrics_history["eval_episode_numbers"].append(episode_num)
                metrics_history["timestamps"].append(elapsed_eval)

                logger.info(
                    "%s eval episode=%d reward=%.2f avg_reward=%.2f time=%.2fs",
                    exp_prefix,
                    episode_num,
                    eval_reward,
                    avg_r,
                    elapsed_eval,
                )

                # Check solve
                if avg_r >= target_reward and not solved and len(rewards) >= 10:
                    logger.info(
                        f"{exp_prefix} Environment solved in {episode_num} episodes! avg reward={avg_r:.2f}"
                    )
                    agent.save(f"ppo_highway_solved_{experiment_name}.pth")
                    solved = True

                # Best model
                if avg_r > best_avg_reward:
                    best_avg_reward = avg_r
                    agent.save(f"ppo_highway_best_{experiment_name}.pth")
                    logger.info(
                        f"{exp_prefix} New best model saved, avg reward={best_avg_reward:.2f}"
                    )

        # Bootstrapping final value
        final_value = 0.0
        if not done:
            with torch.no_grad():
                _, _, final_value = agent.actor_critic.forward(flat_state)
                final_value = float(final_value.cpu().item())

        logger.debug(f"{exp_prefix} Updating policy after {steps_collected} steps...")
        update_metrics = agent.update(last_value=final_value)
        update_time = time.time() - update_start
        metrics_history["policy_updates"].append(
            {
                "episode": episode_num,
                "steps": steps_collected,
                "time": update_time,
                **update_metrics,
            }
        )
        logger.debug(f"{exp_prefix} Policy update done in {update_time:.2f}s")

    # Save metrics and artifacts
    metrics_path = os.path.join(
        artifacts_dir, f"training_metrics_{experiment_name}.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    logger.info(f"{exp_prefix} Metrics saved to {metrics_path}")

    # Plot rewards
    plt.figure(figsize=(12, 8))
    plt.plot(
        training_episodes,
        episode_rewards,
        alpha=0.3,
        label="Training Reward",
        color="gray",
    )
    # Smoothed
    if len(episode_rewards) > 20:
        smoothed = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")
        plt.plot(training_episodes[19:], smoothed, label="Training (Moving Avg)")
    plt.plot(eval_episodes, rewards, "ro-", label="Eval Reward")
    plt.plot(eval_episodes, avg_rewards, "go-", label="Eval Moving Avg")
    plt.axhline(y=target_reward, color="r", linestyle="--", label="Target Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Training Progress ({experiment_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(
        artifacts_dir, f"ppo_highway_rewards_{experiment_name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"{exp_prefix} Training plot saved to {plot_path}")

    # CSV summary
    csv_path = os.path.join(artifacts_dir, f"summary_{experiment_name}.csv")
    with open(csv_path, "w") as f:
        f.write("experiment,final_reward,max_reward,steps,best_model,plot\n")
        f.write(
            f"{experiment_name},{avg_rewards[-1]:.4f},{max(avg_rewards):.4f},{total_steps},"
        )
        f.write(
            f"ppo_highway_best_{experiment_name}.pth,{os.path.basename(plot_path)}\n"
        )
    logger.info(f"{exp_prefix} Summary CSV saved to {csv_path}")

    return rewards, avg_rewards, metrics_history
