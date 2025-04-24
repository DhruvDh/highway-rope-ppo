"""
utils.defaults
Central place to pull sensible default hyper-parameters from HIGHWAY_CONFIG
so that all wrappers stay in sync with the env definition.
"""

from config.base_config import HIGHWAY_CONFIG as _CFG


def max_dist() -> float:
    """Largest |x| or |y| the observation clip allows (metres)."""
    rng = _CFG["observation"]["features_range"]
    return max(abs(rng["x"][0]), abs(rng["x"][1]), abs(rng["y"][0]), abs(rng["y"][1]))


def max_rank() -> int:
    """Number of non-ego rows returned by the obs wrapper."""
    return _CFG["observation"]["vehicles_count"]


def feature_count() -> int:
    """Number of scalar features per vehicle row."""
    return len(_CFG["observation"]["features"])
