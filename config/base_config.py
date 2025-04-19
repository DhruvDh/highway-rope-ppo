"""
Base configuration for the Highway Env experiments.
"""

HIGHWAY_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,  # Number of vehicles to observe
        "features": ["x", "y", "vx", "vy"],  # Features to include
        "normalize": True,  # Use built-in highway-env normalization
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-30, 30],
            "vy": [-30, 30],
            "presence": [0, 1],
            "cos_h": [-1, 1],
            "sin_h": [-1, 1],
        },
        "absolute": False,
        "order": "sorted",
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 40,
    "lanes_count": 3,
    "vehicles_count": 50,
    "vehicles_density": 2,
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.05,
    "reward_speed_range": [20, 30],
}
