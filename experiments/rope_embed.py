from gymnasium import ObservationWrapper, spaces
import numpy as np


class RotaryEmbedWrapper(ObservationWrapper):
    def __init__(self, env, d_embed=8, max_dist=100.0):
        super().__init__(env)
        # Ensure even embedding dimension for rotary embeddings
        if d_embed % 2 != 0:
            raise ValueError("d_embed must be even for RoPE")
        self.d_embed = d_embed
        self.max_dist = float(max_dist)
        # Precompute inverse frequencies for rotary encoding
        inv_freq = 1.0 / (10000 ** (np.arange(0, d_embed, 2) / d_embed))
        self.inv_freq = inv_freq.astype(np.float32)

        # Original observation space dimensions
        N, F = env.observation_space.shape
        # New observation space with appended rotary embeddings
        new_low = np.concatenate(
            [env.observation_space.low, -np.ones((N, d_embed), dtype=np.float32)],
            axis=1,
        )
        new_high = np.concatenate(
            [env.observation_space.high, np.ones((N, d_embed), dtype=np.float32)],
            axis=1,
        )
        self.observation_space = spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )

    def _rope(self, dist_norm: np.ndarray) -> np.ndarray:
        # dist_norm: shape (N,) normalized distances
        theta = np.outer(dist_norm, self.inv_freq)  # shape (N, d_embed/2)
        # Concatenate sin and cos components
        return np.concatenate([np.sin(theta), np.cos(theta)], axis=1)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (N, F)
        # Compute normalized distance of each vehicle
        dist = np.linalg.norm(obs[:, :2], axis=-1) / self.max_dist  # shape (N,)
        # Compute rotary positional embeddings
        rope_embed = self._rope(dist)
        # Concatenate and return full observation
        obs_float = obs.astype(np.float32)
        return np.concatenate([obs_float, rope_embed], axis=1).astype(np.float32)
