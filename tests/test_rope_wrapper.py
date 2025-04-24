import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from experiments.rope_embed import RotaryEmbedWrapper


class DummyEnv(Env):
    """
    Minimal gymnasium environment stub with observation_space and action_space,
    implementing abstract methods.
    """

    def __init__(self, shape):
        super().__init__()
        # Continuous Box space with given shape
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )
        # Simple dummy action space
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Return zero observation and empty info
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        # Return zero observation, zero reward, done=True, truncated=False, empty info
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


def test_shape_preserved():
    N, F = 5, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, rotate_dim=4, max_dist=10.0)
    obs = np.random.randn(N, F).astype(np.float32)
    out = wrapper.observation(obs)
    # Shape must be identical
    assert out.shape == obs.shape, f"Expected shape {obs.shape}, got {out.shape}"


def test_dtype_float32():
    N, F = 3, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, rotate_dim=4, max_dist=10.0)
    obs = np.random.randn(N, F).astype(np.float32)
    out = wrapper.observation(obs)
    assert out.dtype == np.float32, f"Expected dtype float32, got {out.dtype}"


def test_identity_for_zero_distance():
    N, F = 4, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, rotate_dim=4, max_dist=1.0)
    # obs with zero position (x,y)=0 -> dist_norm=0, rotation angle=0
    obs = np.zeros((N, F), dtype=np.float32)
    # Fill other channels to nonzero values to ensure full identity
    obs[:, 2:] = np.random.randn(N, F - 2).astype(np.float32)
    out = wrapper.observation(obs)
    assert np.allclose(out, obs, atol=1e-6), (
        "Wrapper should be identity when distance=0"
    )


def test_rotation_changes_values_for_nonzero_distance():
    N, F = 4, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, rotate_dim=4, max_dist=1.0)
    # obs with unit x component so rotation is evident
    obs = np.zeros((N, F), dtype=np.float32)
    obs[:, 0] = 1.0
    # Use full normalized distance = 1 for maximum rotation
    dist_norm = np.ones(N, dtype=np.float32)
    wrapped = wrapper._apply_rope(obs.copy(), dist_norm)
    # x,y channels should change because rotation angle != 0
    assert not np.allclose(wrapped[:, :2], obs[:, :2]), "Rotation did not alter values"


def test_invertibility():
    N, F = 6, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, rotate_dim=4, max_dist=1.0)
    obs = np.random.randn(N, F).astype(np.float32)
    dist_norm = np.random.rand(N).astype(np.float32)
    obs_f = obs.copy()
    wrapped = wrapper._apply_rope(obs_f, dist_norm)
    # Apply inverse rotation by negating distance
    inverted = wrapper._apply_rope(wrapped, -dist_norm)
    assert np.allclose(inverted, obs_f, atol=1e-6), (
        "Inverse rotation failed to recover original"
    )


def test_default_rotate_dim_uses_full_features():
    N, F = 4, 4
    dummy = DummyEnv((N, F))
    wrapper = RotaryEmbedWrapper(dummy, max_dist=1.0)
    # rotate_dim None should default to all feature dims
    assert wrapper.rotate_dim == F, (
        "Default rotate_dim should be full feature dimension"
    )


def test_invalid_rotate_dim_raises():
    dummy = DummyEnv((5, 4))
    # Odd rotate_dim
    with pytest.raises(ValueError):
        RotaryEmbedWrapper(dummy, rotate_dim=3)
    # Larger than F
    with pytest.raises(ValueError):
        RotaryEmbedWrapper(dummy, rotate_dim=6)
