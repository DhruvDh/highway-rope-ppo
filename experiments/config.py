# experiments/config.py
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class Condition(Enum):
    SORTED = auto()
    SHUFFLED = auto()
    SHUFFLED_RANKPE = auto()
    SHUFFLED_DISTPE = auto()
    # Add more conditions like GRAPH_ATTENTION here later


@dataclass
class CommonHP:
    """Hyperparameters common across all conditions."""

    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    steps_per_update: int = 2048


@dataclass
class ConditionHP(CommonHP):
    """Hyperparameters that can be specialized per condition."""

    lr: float = 1e-4
    clip_eps: float = 0.2
    epochs: int = 6
    batch_size: int = 64
    hidden_dim: int = 128
    d_embed: Optional[int] = None  # Embedding dimension for PE variants


@dataclass
class Experiment:
    """Defines a single experiment run."""

    name: str  # Unique name for logging/artifacts
    condition: Condition
    hp: ConditionHP = field(default_factory=ConditionHP)
    seed: int = 42  # Specific seed for this run
    max_episodes: int = 1500  # Max training episodes for this run
    target_reward: float = 20.0  # Target reward for this run
    env_config_overrides: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(
        default_factory=dict
    )  # For runner-specific args or future use
