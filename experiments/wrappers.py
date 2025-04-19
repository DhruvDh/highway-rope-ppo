# experiments/wrappers.py
import copy
import gymnasium as gym
import highway_env
from .config import Condition
from .rank_embed import RankEmbedWrapper
from .dist_embed import DistanceEmbedWrapper
from typing import Optional, Dict, Any


def make_env(
    exp_condition: Condition,
    base_cfg: Dict[str, Any],
    d_embed: Optional[int] = None,
    env_overrides: Dict[str, Any] = {},
):
    """
    Creates and wraps the Highway-v0 environment based on the experiment condition.

    Args:
        exp_condition: The Condition enum member.
        base_cfg: The base highway-env configuration dictionary.
        d_embed: Embedding dimension (required for PE conditions).
        env_overrides: Dictionary to override specific keys in the base_cfg.

    Returns:
        A Gymnasium environment instance.
    """
    # 1. Deep copy to avoid mutating original config
    cfg = copy.deepcopy(base_cfg)

    # Helper for recursive deep merge
    def deep_update(orig: Dict[str, Any], updates: Dict[str, Any]):
        for key, val in updates.items():
            if key in orig and isinstance(orig[key], dict) and isinstance(val, dict):
                deep_update(orig[key], val)
            else:
                orig[key] = val

    # 2. Apply environment-specific overrides (recursive deep merge)
    deep_update(cfg, env_overrides)

    # 3. Set observation order based on condition if not overridden
    obs_cfg = cfg.setdefault("observation", {})
    match exp_condition:
        case Condition.SORTED:
            obs_cfg.setdefault("order", "sorted")
        case Condition.SHUFFLED | Condition.SHUFFLED_RANKPE | Condition.SHUFFLED_DISTPE:
            obs_cfg.setdefault("order", "shuffled")
        # other conditions can be added here

    # 4. Ensure highway-env is registered (always register to support Gym/Gymnasium registry changes)
    highway_env._register_highway_envs()

    # 5. Create base environment
    env = gym.make("highway-v0", config=cfg)

    # 6. Wrap based on condition
    match exp_condition:
        case Condition.SHUFFLED_RANKPE:
            if d_embed is None:
                raise ValueError("d_embed must be specified for SHUFFLED_RANKPE")
            return RankEmbedWrapper(env, d_embed=d_embed)
        case Condition.SHUFFLED_DISTPE:
            if d_embed is None:
                raise ValueError("d_embed must be specified for SHUFFLED_DISTPE")
            return DistanceEmbedWrapper(env, d_embed=d_embed)
        case _:
            return env
