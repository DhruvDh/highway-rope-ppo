import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np
import torch


class DistanceEmbedWrapper(ObservationWrapper):
    # Example using fixed sinusoidal encoding based on Euclidean distance
    def __init__(self, env, d_embed=8, max_dist=100.0, use_euclidean=True):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("DistanceEmbedWrapper requires Box observation space.")
        if len(env.observation_space.shape) != 2:
            raise ValueError(
                "DistanceEmbedWrapper requires 2D Box observation space (N, F)."
            )

        N, F = env.observation_space.shape
        self.d_embed = d_embed
        self.max_dist = float(max_dist)
        self.use_euclidean = use_euclidean

        # Check if features are sufficient
        required_feats = 2 if use_euclidean else 1
        if F < required_feats:
            raise ValueError(
                f"DistanceEmbedWrapper requires at least {required_feats} feature(s) for distance calculation (features available: {F})."
            )

        # Fixed sinusoidal frequencies (no learnable parameters needed for this type)
        self.freqs = torch.exp(
            -torch.arange(0, d_embed, 2, dtype=torch.float32)
            * (np.log(10000.0) / d_embed)
        )

        # Define new observation space
        new_shape = (N, F + d_embed)
        new_low = np.concatenate(
            [self.observation_space.low, -1.0 * np.ones((N, d_embed))], axis=1
        )
        new_high = np.concatenate(
            [self.observation_space.high, 1.0 * np.ones((N, d_embed))], axis=1
        )

        self.observation_space = spaces.Box(
            low=new_low, high=new_high, shape=new_shape, dtype=np.float32
        )
        self._device = torch.device("cpu")

    def to(self, device):
        """Moves frequency tensor to the specified device."""
        self.freqs = self.freqs.to(device)
        self._device = device
        return self

    def observation(self, obs):
        # obs shape (N, F) - numpy array
        if self.use_euclidean:
            dist = np.linalg.norm(obs[:, :2], axis=-1, keepdims=True)
        else:
            dist = np.abs(obs[:, :1])

        # Normalize distance
        norm_dist = np.clip(dist / self.max_dist, 0.0, 5.0)

        # Apply sinusoidal encoding
        freqs_np = self.freqs.cpu().numpy()
        angles = norm_dist * freqs_np
        embed = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

        obs_float = obs.astype(np.float32)
        return np.concatenate([obs_float, embed], axis=-1).astype(np.float32)
