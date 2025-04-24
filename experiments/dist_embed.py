from gymnasium import ObservationWrapper, spaces
import numpy as np
from typing import Optional
from utils.defaults import max_dist as _max_dist
import torch


class DistanceEmbedWrapper(ObservationWrapper):
    # Example using fixed sinusoidal encoding based on Euclidean distance
    def __init__(
        self,
        env,
        d_embed: int = 8,
        max_dist: float = _max_dist(),
        base: float | None = None,
        use_euclidean: bool = True,
        ego_idx: int = 0,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("DistanceEmbedWrapper requires Box observation space.")
        if len(env.observation_space.shape) != 2:
            raise ValueError(
                "DistanceEmbedWrapper requires 2D Box observation space (N, F)."
            )

        N, F = env.observation_space.shape
        self.d_embed = d_embed
        # Ensure embedding dimension is even to match sinusoidal concatenation
        if self.d_embed % 2 != 0:
            raise ValueError(
                f"DistanceEmbedWrapper requires even d_embed; got {self.d_embed}"
            )
        self.max_dist = float(max_dist)
        base = base or self.max_dist  # sensible physical default for frequency base
        self.use_euclidean = use_euclidean
        # Index of ego vehicle for relative distance calculation
        self.ego_idx = ego_idx

        # Check if features are sufficient
        required_feats = 2 if use_euclidean else 1
        if F < required_feats:
            raise ValueError(
                f"DistanceEmbedWrapper requires at least {required_feats} feature(s) for distance calculation (features available: {F})."
            )

        # Use physical distance as base for sinusoidal frequencies
        self.freqs = torch.exp(
            -torch.arange(0, d_embed, 2, dtype=torch.float32) * (np.log(base) / d_embed)
        )
        # Cache numpy version of frequencies to avoid repeated .cpu().numpy()
        self._freqs_np: np.ndarray = self.freqs.cpu().numpy()

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
        # Move tensor and update cached numpy version
        self.freqs = self.freqs.to(device)
        self._freqs_np = self.freqs.cpu().numpy()
        self._device = device
        return self

    def observation(self, obs):
        # obs shape (N, F) - numpy array
        # Compute distance of each vehicle relative to ego and keep dims
        if self.use_euclidean:
            ego_xy = obs[self.ego_idx, :2]  # (2,) position of ego
            rel = obs[:, :2] - ego_xy[None, :]  # (N,2) relative positions
            dist = np.linalg.norm(rel, axis=-1, keepdims=True)
        else:
            ego_feat = obs[self.ego_idx, :1]  # (1,) feature of ego
            dist = np.abs(obs[:, :1] - ego_feat[None, :])  # (N,1) relative abs value

        # Normalize distance
        norm_dist = np.clip(dist / self.max_dist, 0.0, 1.0)

        # Apply sinusoidal encoding
        # DistPE uses angles in radians; multiply by 2π for full cycles over normalized distance
        angles = 2 * np.pi * norm_dist * self._freqs_np
        embed = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

        obs_float = obs.astype(np.float32)
        return np.concatenate([obs_float, embed], axis=-1).astype(np.float32)
