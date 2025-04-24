"""
visualize.py

Utility to load a trained PPO checkpoint (.pth), roll out episodes in
Highway‑Env with live rendering or MP4 recording, and (optionally) loop over
a list of checkpoints to create a tiled "show‑reel" for the presentation.

Examples
--------
# 1. Play the best shuffled_rope agent live
python visualize.py --model artifacts/highway-ppo/ppo_highway_best_shuffled_rope_lr0.0003_hidden_dim256_...pth

# 2. Record a 30‑second MP4 for each of the five conditions
python visualize.py \
   --model-list best_checkpoints.txt \
   --record-dir demo_videos \
   --secs-per-agent 30
"""

from __future__ import annotations

import argparse
import pathlib
import re
import time
from typing import Iterable

import torch
import gymnasium as gym
import highway_env
import copy
from experiments.wrappers import (
    RankEmbedWrapper,
    DistanceEmbedWrapper,
    RotaryEmbedWrapper,
)
from experiments.config import Condition
from config.base_config import HIGHWAY_CONFIG
from ppo.agent import PPOAgent


CHK_RX = re.compile(r"^ppo_highway_(?:best|solved)_(?P<exp_name>.+)\.pth$")


def load_agent(checkpoint_path: pathlib.Path) -> tuple[PPOAgent, dict]:
    """
    Instantiate PPOAgent matching the dimensions of the checkpointed network
    (no reliance on hard‑coded 128 hidden units or 2‑action assumption).
    """
    chk = torch.load(checkpoint_path, map_location="cpu")
    state_dict = chk["model"]

    # Infer dimensions
    w_shared0 = state_dict["shared.0.weight"]  # [hidden_dim, state_dim]
    hidden_dim, state_dim = w_shared0.shape
    w_actor_final = state_dict["actor_mean.2.weight"]  # [action_dim, hidden_dim]
    action_dim = w_actor_final.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    agent.actor_critic.load_state_dict(state_dict)
    agent.actor_critic.eval()
    cfg = chk.get("config", {})
    return agent, cfg


def infer_condition(exp_name: str) -> Condition:
    """Guess Condition from experiment name prefix."""
    if exp_name.startswith("sorted"):
        return Condition.SORTED
    if exp_name.startswith("shuffled_rankpe"):
        return Condition.SHUFFLED_RANKPE
    if exp_name.startswith("shuffled_distpe"):
        return Condition.SHUFFLED_DISTPE
    if exp_name.startswith("shuffled_rope"):
        return Condition.SHUFFLED_ROPE
    if exp_name.startswith("shuffled"):
        return Condition.SHUFFLED
    raise ValueError(f"Cannot infer condition from {exp_name}")


def run_episode(
    env: gym.Env,
    agent: PPOAgent,
    render: bool = True,
):
    state, _ = env.reset(seed=0)
    flat_state = state.reshape(-1)
    done = False
    total_reward = 0.0
    while not done:
        action, *_ = agent.select_action(flat_state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        flat_state = state.reshape(-1)
        total_reward += reward
        if render:
            env.render()
    return total_reward


def visualise_single(
    model_path: pathlib.Path,
    render: bool,
    record_path: pathlib.Path | None,
    num_episodes: int,
):
    print(f"Loading {model_path.name}")
    agent, _ = load_agent(model_path)

    exp_match = CHK_RX.match(model_path.name)
    exp_name = exp_match.group("exp_name") if exp_match else model_path.stem

    cond = infer_condition(exp_name)
    # extract d_embed if the condition uses positional-embedding
    d_embed = None
    if cond in {
        Condition.SHUFFLED_RANKPE,
        Condition.SHUFFLED_DISTPE,
        Condition.SHUFFLED_ROPE,
    }:
        m = re.search(r"_d_embed(\d+)", exp_name)
        if not m:
            raise ValueError("Cannot parse d_embed from checkpoint name")
        d_embed = int(m.group(1))

    # decide render mode
    if record_path and not render:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    # Build the base env with render_mode at creation time and apply wrappers
    cfg = copy.deepcopy(HIGHWAY_CONFIG)
    obs_cfg = cfg.setdefault("observation", {})
    if cond == Condition.SORTED:
        obs_cfg.setdefault("order", "sorted")
    else:
        obs_cfg.setdefault("order", "shuffled")

    base_env = gym.make("highway-v0", config=cfg, render_mode=render_mode)
    if cond == Condition.SHUFFLED_RANKPE:
        env = RankEmbedWrapper(base_env, d_embed)
    elif cond == Condition.SHUFFLED_DISTPE:
        env = DistanceEmbedWrapper(base_env, d_embed)
    elif cond == Condition.SHUFFLED_ROPE:
        env = RotaryEmbedWrapper(base_env, d_embed)
    else:
        env = base_env

    # Wrap with RecordVideo for recording
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(record_path.parent),
            name_prefix=record_path.stem,
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )
        print(f"Recording to {record_path.parent}/{record_path.stem}*.mp4")

    rewards = []
    try:
        for ep in range(num_episodes):
            r = run_episode(env, agent, render=render)
            rewards.append(r)
            print(f"Episode {ep + 1}/{num_episodes}  reward={r:.1f}")
    finally:
        env.close()
    print(f"Mean reward: {sum(rewards) / len(rewards):.2f}")


def iter_models_from_file(path: pathlib.Path) -> Iterable[pathlib.Path]:
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield pathlib.Path(stripped)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise trained PPO agents.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=pathlib.Path, help="Single .pth checkpoint")
    g.add_argument(
        "--model-list",
        type=pathlib.Path,
        help="Text file: one checkpoint path per line for batch recording",
    )

    p.add_argument("--render", action="store_true", help="Interactive window")
    p.add_argument(
        "--record-dir",
        type=pathlib.Path,
        help="Directory to write MP4s (one per model)",
    )
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument(
        "--secs-per-agent",
        type=int,
        default=None,
        help="Cap video length (approx) per agent when using --model-list",
    )
    args = p.parse_args()

    if args.model:
        out_mp4 = None
        if args.record_dir:
            out_mp4 = args.record_dir / (args.model.stem + ".mp4")
        visualise_single(
            args.model,
            render=args.render,
            record_path=out_mp4,
            num_episodes=args.episodes,
        )
    else:
        args.record_dir = args.record_dir or pathlib.Path("demo_videos")
        for m in iter_models_from_file(args.model_list):
            out_path = args.record_dir / (m.stem + ".mp4")
            visualise_single(m, render=False, record_path=out_path, num_episodes=1)
            if args.secs_per_agent:
                # crude trim by overwriting writer duration; handled externally
                pass


if __name__ == "__main__":
    main()
