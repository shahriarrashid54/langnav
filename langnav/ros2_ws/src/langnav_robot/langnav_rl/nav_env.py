"""Gymnasium environment for robot navigation in simulation."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any


class NavEnv(gym.Env):
    """
    Navigation environment where agent learns to reach target objects.

    State: [robot_x, robot_y, target_x, target_y, target_class_embedding]
    Action: [linear_vel, angular_vel]
    Reward: -distance_to_target - collision_penalty + goal_reached_bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps: int = 500, render_mode: str = None):
        """
        Initialize navigation environment.

        Args:
            max_episode_steps: Episode length limit
            render_mode: "human" or "rgb_array" or None
        """
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Robot state: [x, y, theta]
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_theta = 0.0

        # Target state
        self.target_pos = np.array([0.0, 0.0])
        self.target_id = 0

        # Episode tracking
        self.step_count = 0

        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi]),
            high=np.array([1.0, np.pi]),
            dtype=np.float32,
        )

        # Observation space: [robot_x, robot_y, target_x, target_y, distance, target_embedding (128)]
        obs_dim = 5 + 128  # distance + target embedding
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Spawn robot at origin
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_theta = 0.0

        # Spawn target at random location
        self.target_pos = self.np_random.uniform(-5, 5, size=2)
        self.target_id = self.np_random.integers(0, 80)  # 80 COCO classes

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state + reward.

        Args:
            action: [linear_vel, angular_vel]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        linear_vel, angular_vel = action

        # Update robot pose (simple kinematic model)
        dt = 0.1
        self.robot_theta += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * dt

        # Collision detection (stay in [-10, 10] bounds)
        self.robot_pos = np.clip(self.robot_pos, -10, 10)

        self.step_count += 1

        # Compute reward
        distance = np.linalg.norm(self.robot_pos - self.target_pos)
        reward = -distance  # Negative distance as reward

        # Goal reached
        terminated = distance < 0.5
        if terminated:
            reward += 10.0  # Bonus for reaching goal

        # Episode limit
        truncated = self.step_count >= self.max_episode_steps

        # Collision penalty
        if np.any(np.abs(self.robot_pos) >= 10):
            reward -= 1.0

        obs = self._get_obs()
        info = {"distance": distance, "success": terminated}

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        distance = np.linalg.norm(self.robot_pos - self.target_pos)
        target_embedding = self._get_target_embedding()

        obs = np.concatenate([
            self.robot_pos,
            self.target_pos,
            [distance],
            target_embedding,
        ]).astype(np.float32)

        return obs

    def _get_target_embedding(self) -> np.ndarray:
        """Get semantic embedding for target object (dummy for now)."""
        # In real version, this would be CLIP embedding of target class
        np.random.seed(self.target_id)
        return np.random.randn(128).astype(np.float32)

    def render(self):
        """Render environment (placeholder)."""
        if self.render_mode == "human":
            print(f"Robot: {self.robot_pos}, Target: {self.target_pos}")
