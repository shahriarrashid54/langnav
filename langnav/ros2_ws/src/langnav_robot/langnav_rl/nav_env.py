"""
2D navigation environment for fast RL training (no ROS2 required).

Observation (133 dims):
  [0]     distance          — meters to target (normalized by room size)
  [1]     cos(heading_err)  — cosine of angle between robot heading and target bearing
  [2]     sin(heading_err)  — sine of same
  [3]     dx_norm           — (target_x - robot_x) / room_size
  [4]     dy_norm           — (target_y - robot_y) / room_size
  [5:8]   nearest_obstacles — distance to 3 nearest obstacles (normalized)
  [8:]    target_embed      — 128-dim semantic embedding

Action (2 dims):
  [0]  linear_vel  ∈ [-1, 1]  m/s
  [1]  angular_vel ∈ [-π, π]  rad/s

Reward (potential-based + shaping):
  dense:     γ·Φ(s') - Φ(s),  Φ(s) = -distance   (guarantees policy improvement)
  heading:   +cos(heading_err) · 0.05               (face the target)
  goal:      +10.0 on arrival
  wall:      -1.0 on boundary hit
  obstacle:  -1.0 per collision
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, List


ROOM_SIZE       = 10.0
GOAL_RADIUS     = 0.5
WALL_PENALTY    = -1.0
OBS_PENALTY     = -1.0
GOAL_BONUS      = 10.0
GAMMA_POTENTIAL = 0.99
N_OBSTACLE_DIMS = 3
EMBED_DIM       = 128
OBS_DIM         = 8 + EMBED_DIM


class NavEnv(gym.Env):
    """Fast 2D navigation sim — matches GazeboEnv reward structure."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_episode_steps: int = 500,
        n_obstacles: int = 4,
        render_mode: str = None,
    ):
        """
        Args:
            max_episode_steps: Episode length cap
            n_obstacles: Static circular obstacles per episode
            render_mode: "human" for text debug output
        """
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.n_obstacles       = n_obstacles
        self.render_mode       = render_mode

        self.robot_pos    = np.zeros(2, dtype=np.float32)
        self.robot_theta  = 0.0
        self.target_pos   = np.zeros(2, dtype=np.float32)
        self.target_id    = 0
        self.obstacles: List[np.ndarray] = []  # List of (x, y, radius)
        self.step_count   = 0
        self._prev_dist   = 0.0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi], dtype=np.float32),
            high=np.array([1.0,  np.pi], dtype=np.float32),
        )

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        half = ROOM_SIZE / 2 - 0.5

        self.robot_pos   = np.zeros(2, dtype=np.float32)
        self.robot_theta = float(self.np_random.uniform(-np.pi, np.pi))

        # Target: at least 1.5 m from robot spawn
        while True:
            self.target_pos = self.np_random.uniform(-half, half, size=2).astype(np.float32)
            if np.linalg.norm(self.target_pos) >= 1.5:
                break

        self.target_id = int(self.np_random.integers(0, 80))

        # Random circular obstacles — not on robot, not on target
        self.obstacles = self._sample_obstacles(n=self.n_obstacles, half=half)

        self.step_count = 0
        self._prev_dist = float(np.linalg.norm(self.robot_pos - self.target_pos))

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        linear_vel  = float(action[0])
        angular_vel = float(action[1])

        dt = 0.1
        self.robot_theta += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * dt

        self.step_count += 1

        distance = float(np.linalg.norm(self.robot_pos - self.target_pos))
        reward, terminated = self._compute_reward(distance)
        truncated = self.step_count >= self.max_episode_steps

        self._prev_dist = distance

        info = {
            "distance":       distance,
            "success":        bool(terminated and distance < GOAL_RADIUS),
            "episode_length": self.step_count,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── Observation ─────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        distance = float(np.linalg.norm(self.robot_pos - self.target_pos))
        dx = (self.target_pos[0] - self.robot_pos[0]) / ROOM_SIZE
        dy = (self.target_pos[1] - self.robot_pos[1]) / ROOM_SIZE

        bearing = np.arctan2(
            self.target_pos[1] - self.robot_pos[1],
            self.target_pos[0] - self.robot_pos[0],
        )
        heading_err = bearing - self.robot_theta

        obs = np.array([
            distance / ROOM_SIZE,
            np.cos(heading_err),
            np.sin(heading_err),
            dx,
            dy,
            *self._nearest_obstacle_dists(),
        ], dtype=np.float32)

        obs = np.concatenate([obs, self._target_embedding()])
        return obs

    def _nearest_obstacle_dists(self) -> List[float]:
        """Distance to N_OBSTACLE_DIMS nearest obstacles, normalized."""
        dists = sorted([
            float(np.linalg.norm(self.robot_pos - obs[:2]) - obs[2])
            for obs in self.obstacles
        ])
        # Pad with max range if fewer obstacles than dims
        while len(dists) < N_OBSTACLE_DIMS:
            dists.append(ROOM_SIZE)
        return [d / ROOM_SIZE for d in dists[:N_OBSTACLE_DIMS]]

    def _target_embedding(self) -> np.ndarray:
        """Deterministic embedding per target_id (proxy for CLIP embed)."""
        rng = np.random.default_rng(seed=self.target_id)
        embed = rng.standard_normal(EMBED_DIM).astype(np.float32)
        return embed / (np.linalg.norm(embed) + 1e-8)

    # ── Reward ──────────────────────────────────────────────────────────────

    def _compute_reward(self, distance: float) -> Tuple[float, bool]:
        reward = 0.0
        terminated = False

        # Potential-based shaping (guarantees policy-invariance)
        reward += GAMMA_POTENTIAL * (-distance) - (-self._prev_dist)

        # Heading bonus
        bearing = np.arctan2(
            self.target_pos[1] - self.robot_pos[1],
            self.target_pos[0] - self.robot_pos[0],
        )
        heading_err = bearing - self.robot_theta
        reward += np.cos(heading_err) * 0.05

        # Goal
        if distance < GOAL_RADIUS:
            reward += GOAL_BONUS
            terminated = True
            return reward, terminated

        # Wall collision
        half = ROOM_SIZE / 2
        if np.any(np.abs(self.robot_pos) > half):
            self.robot_pos = np.clip(self.robot_pos, -half, half)
            reward += WALL_PENALTY

        # Obstacle collision
        for obs in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs[:2]) < obs[2] + 0.18:
                reward += OBS_PENALTY

        return reward, terminated

    # ── Obstacles ───────────────────────────────────────────────────────────

    def _sample_obstacles(self, n: int, half: float) -> List[np.ndarray]:
        obstacles = []
        for _ in range(n):
            for _attempt in range(100):
                pos = self.np_random.uniform(-half, half, size=2).astype(np.float32)
                radius = float(self.np_random.uniform(0.2, 0.5))
                if (
                    np.linalg.norm(pos - self.robot_pos) > radius + 1.0
                    and np.linalg.norm(pos - self.target_pos) > radius + 0.8
                ):
                    obstacles.append(np.array([pos[0], pos[1], radius], dtype=np.float32))
                    break
        return obstacles

    def render(self):
        if self.render_mode == "human":
            dist = np.linalg.norm(self.robot_pos - self.target_pos)
            print(
                f"step={self.step_count:4d}  "
                f"robot=({self.robot_pos[0]:5.2f}, {self.robot_pos[1]:5.2f})  "
                f"target=({self.target_pos[0]:5.2f}, {self.target_pos[1]:5.2f})  "
                f"dist={dist:.2f}"
            )
