from gymnasium import ObservationWrapper, spaces
from utils.defaults import max_dist as _max_dist, feature_count as _F
import numpy as np


class RotaryEmbedWrapper(ObservationWrapper):
    """
    True Rotary Positional-Embedding wrapper
    ---------------------------------------
    * Rotates the first `rotate_dim` features of every vehicle (x,y,…) in 2-D planes whose angles grow with normalised distance.
    * Does **not** change the observation dimensionality.
    """

    def __init__(
        self,
        env,
        rotate_dim: int | None = None,
        max_dist: float = _max_dist(),
        base: float | None = None,
        ego_idx: int = 0,
    ):
        super().__init__(env)
        # Get shape and validate rotate_dim
        N, F = env.observation_space.shape
        self.rotate_dim = rotate_dim or (F - (F % 2))
        if self.rotate_dim % 2 != 0 or self.rotate_dim > F:
            raise ValueError(
                f"rotate_dim must be even and ≤ {F}; got {self.rotate_dim}"
            )
        self.max_dist = float(max_dist)
        base = base or self.max_dist  # sensible physical default for distance
        # Index of ego vehicle for relative distance calculation
        self.ego_idx = ego_idx

        # One inverse-frequency per 2-D pair
        pair_count = self.rotate_dim // 2
        self.inv_freq = 1.0 / (
            base ** (np.arange(pair_count, dtype=np.float32) / pair_count)
        )

        # Observation space unchanged
        self.observation_space = env.observation_space

    def _apply_rope(self, obs: np.ndarray, dist_norm: np.ndarray) -> np.ndarray:
        """
        Vectorised rotation of feature pairs.
        `obs`      : (N, F) float32
        `dist_norm`: (N,)   float32 in [0,1]
        """
        N = obs.shape[0]
        # Reshape first rotate_dim channels into pairs
        pair_obs = obs[:, : self.rotate_dim].reshape(N, -1, 2)  # (N, P, 2)
        # RoPE uses angles in radians; multiply by 2π so the slowest wave
        # completes one full rotation when dist_norm == 1
        theta = 2 * np.pi * dist_norm[:, None] * self.inv_freq[None, :]  # (N, P)
        sin, cos = np.sin(theta)[..., None], np.cos(theta)[..., None]
        x, y = pair_obs[..., 0:1], pair_obs[..., 1:2]
        # Apply rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
        pair_rot = np.concatenate([x * cos - y * sin, x * sin + y * cos], axis=-1)
        out = obs.copy()
        out[:, : self.rotate_dim] = pair_rot.reshape(N, self.rotate_dim)
        return out

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (N, F)
        # Compute normalized distance of each vehicle relative to ego and clip to [0,1]
        ego_xy = obs[self.ego_idx, :2]  # (2,) position of ego
        rel_xy = obs[:, :2] - ego_xy[None, :]  # (N,2) relative positions
        dist = (
            np.linalg.norm(rel_xy, axis=-1) / self.max_dist
        )  # (N,) normalized distances
        dist = np.clip(dist, 0.0, 1.0)
        obs_f = obs.astype(np.float32)
        return self._apply_rope(obs_f, dist)
