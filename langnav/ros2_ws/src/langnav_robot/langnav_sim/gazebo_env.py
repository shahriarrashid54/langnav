"""
Gazebo-backed Gymnasium environment for RL training.
Extends NavEnv to drive real robot in sim via ROS2.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import threading
import time
from typing import Tuple, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langnav_rl.nav_env import NavEnv
from langnav_sim.worlds.world_generator import WorldGenerator


class GazeboEnv(NavEnv):
    """
    NavEnv backed by live Gazebo simulation.
    Uses ROS2 topics for state + action execution.
    Overrides reset() to randomize world via WorldGenerator.
    """

    def __init__(self, n_objects: int = 5, **kwargs):
        super().__init__(**kwargs)

        self.n_objects = n_objects
        self._ros_initialized = False
        self._lock = threading.Lock()

        # Latest sensor data
        self._odom = None
        self._scan = None
        self._world_objects = []

    def _init_ros(self):
        """Lazy ROS2 init (call once before first reset)."""
        if self._ros_initialized:
            return

        rclpy.init()
        self._node = Node("gazebo_env")
        self._world_gen = WorldGenerator()

        # Publishers
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)

        # Subscribers
        self._odom_sub = self._node.create_subscription(
            Odometry, "/odom", self._on_odom, 10
        )
        self._scan_sub = self._node.create_subscription(
            LaserScan, "/scan", self._on_scan, 10
        )

        # Spin in background thread
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

        self._ros_initialized = True

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset episode: randomize world, move robot to origin, sample target."""
        self._init_ros()
        super().reset(seed=seed)

        # Spawn random objects in Gazebo
        self._world_objects = self._world_gen.randomize(n_objects=self.n_objects)

        # Pick random object as navigation target
        if self._world_objects:
            target = self._world_gen.np_random.choice(self._world_objects)
            pos = target["position"]
            self.target_pos = np.array([pos[0], pos[1]], dtype=np.float32)
            self.target_id = hash(target["spec"]) % 80

        # Wait for first odometry
        timeout = 5.0
        start = time.time()
        while self._odom is None and time.time() - start < timeout:
            time.sleep(0.05)

        return self._get_obs(), {"objects": self._world_objects}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute velocity action, read new state from Gazebo."""
        linear_vel, angular_vel = float(action[0]), float(action[1])

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = float(np.clip(linear_vel, -1.0, 1.0))
        cmd.angular.z = float(np.clip(angular_vel, -np.pi, np.pi))
        self._cmd_pub.publish(cmd)

        # Wait one control step
        time.sleep(0.1)

        # Update robot pos from odometry
        if self._odom is not None:
            with self._lock:
                pos = self._odom.pose.pose.position
                self.robot_pos = np.array([pos.x, pos.y], dtype=np.float32)

        self.step_count += 1

        distance = float(np.linalg.norm(self.robot_pos - self.target_pos))
        reward = -distance

        # Goal condition
        terminated = distance < 0.5
        if terminated:
            reward += 10.0
            self._stop_robot()

        # Collision via LIDAR: any reading < 0.15 m
        if self._scan is not None:
            min_range = float(np.min(self._scan.ranges))
            if min_range < 0.15:
                reward -= 2.0
                terminated = True
                self._stop_robot()

        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "distance": distance,
            "success": terminated and distance < 0.5,
            "min_range": float(np.min(self._scan.ranges)) if self._scan else -1.0,
        }

        return obs, reward, terminated, truncated, info

    def _stop_robot(self):
        """Publish zero velocity."""
        self._cmd_pub.publish(Twist())

    def _on_odom(self, msg: Odometry):
        with self._lock:
            self._odom = msg

    def _on_scan(self, msg: LaserScan):
        with self._lock:
            # Replace inf readings with max range
            ranges = np.array(msg.ranges, dtype=np.float32)
            ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
            msg.ranges = ranges.tolist()
            self._scan = msg

    def close(self):
        """Shutdown ROS2."""
        if self._ros_initialized:
            self._stop_robot()
            rclpy.shutdown()
