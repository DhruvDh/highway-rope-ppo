# experiments/wrappers.py
import copy
import gymnasium as gym
import highway_env
from .config import Condition
from .rank_embed import RankEmbedWrapper
from .dist_embed import DistanceEmbedWrapper
from .rope_embed import RotaryEmbedWrapper
from typing import Optional, Dict, Any

_HIGHWAY_ENVS_REGISTERED = False


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
        case (
            Condition.SHUFFLED
            | Condition.SHUFFLED_RANKPE
            | Condition.SHUFFLED_DISTPE
            | Condition.SHUFFLED_ROPE
        ):
            obs_cfg.setdefault("order", "shuffled")
        # other conditions can be added here

    # Early validation for SHUFFLED_DISTPE d_embed
    if exp_condition is Condition.SHUFFLED_DISTPE and d_embed is not None:
        F = len(cfg["observation"].get("features", []))
        if d_embed % 2 != 0 or d_embed > F:
            raise ValueError("d_embed must be even and ≤ feature count for DistPE")

    # Early validation for SHUFFLED_ROPE d_embed
    if exp_condition is Condition.SHUFFLED_ROPE and d_embed is not None:
        # number of features per vehicle
        F = len(cfg["observation"].get("features", []))
        if d_embed % 2 != 0 or d_embed > F:
            raise ValueError("rotate_dim (d_embed) must be even and ≤ feature count")

    # 4. Ensure highway-env is registered (only once per session)
    global _HIGHWAY_ENVS_REGISTERED
    if not _HIGHWAY_ENVS_REGISTERED:
        highway_env._register_highway_envs()
        _HIGHWAY_ENVS_REGISTERED = True

    # 5. Create base environment
    env = gym.make("highway-v0", config=cfg)

    # ------------------------------------------------------------------ #
    #  Fast-fail for dimension mismatches after defaults have expanded   #
    # ------------------------------------------------------------------ #
    F = env.observation_space.shape[1]
    if exp_condition is Condition.SHUFFLED_ROPE and d_embed is not None:
        if d_embed % 2 or d_embed > F:
            raise ValueError(f"rotate_dim / d_embed must be even and ≤ {F}")

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
        case Condition.SHUFFLED_ROPE:
            # d_embed kept for CLI compatibility; becomes rotate_dim
            return RotaryEmbedWrapper(env, rotate_dim=d_embed)
        case _:
            return env
