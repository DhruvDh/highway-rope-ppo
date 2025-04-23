# experiments/rank_embed.py
from gymnasium import ObservationWrapper, spaces
import numpy as np
import torch
import torch.nn as nn


class RankEmbedWrapper(ObservationWrapper):
    def __init__(self, env, d_embed=8):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("RankEmbedWrapper requires Box observation space.")
        if len(env.observation_space.shape) != 2:
            raise ValueError(
                "RankEmbedWrapper requires 2D Box observation space (N, F)."
            )

        N, F = env.observation_space.shape  # (vehicles, features)
        self.d_embed = d_embed
        self.table = nn.Embedding(N, d_embed)
        self.table.weight.data.uniform_(-0.05, 0.05)  # Small random init

        # Define new observation space shape
        new_shape = (N, F + d_embed)
        new_low = np.concatenate(
            [self.observation_space.low, -np.inf * np.ones((N, d_embed))], axis=1
        )
        new_high = np.concatenate(
            [self.observation_space.high, np.inf * np.ones((N, d_embed))], axis=1
        )

        self.observation_space = spaces.Box(
            low=new_low, high=new_high, shape=new_shape, dtype=np.float32
        )
        self._device = torch.device("cpu")  # Default device
        # Precompute static numpy embedding lookup to avoid repeated CPU transfers
        self._embed_cpu = self.table.weight.data.cpu().numpy()

    def to(self, device):
        """Moves the embedding table to the specified device."""
        self.table.to(device)
        self._device = device
        return self

    def observation(self, obs):
        # obs shape (N, F) - numpy array
        # Use precomputed static numpy embeddings and apply bounding
        embed = np.tanh(self._embed_cpu)

        # Ensure obs is float32 before concatenation
        obs_float = obs.astype(np.float32)

        return np.concatenate([obs_float, embed], axis=-1).astype(np.float32)
