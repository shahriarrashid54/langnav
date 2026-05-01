"""
Gazebo-backed Gymnasium environment for RL training.

Observation space (696 dims):
  [0:2]    pixel_center_norm — YOLO target bbox center, normalized [-1,1]
  [2]      semantic_match    — CLIP cosine similarity to command
  [3]      target_detected   — 1.0 when YOLO finds a match, 0.0 otherwise
  [4:184]  lidar_compressed  — 180 min-pooled LIDAR readings, range-normalized
  [184:]   text_embedding    — CLIP text embed of command (512 dims)

Action space: [linear_vel ∈ [-1,1], angular_vel ∈ [-π,π]]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time
from typing import Tuple, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langnav_sim.worlds.world_generator import WorldGenerator

# ROS2 / CV imports are lazy — loaded inside _init_ros() so this module
# can be imported in non-ROS2 environments without raising ModuleNotFoundError.
_rclpy = None
_Node = None
_Twist = None
_LaserScan = None
_Image = None
_Odometry = None
_CvBridge = None
_VisionObsBuilder = None
_OBS_DIM = None

# Default command when none is set — agent will not detect anything useful
_FALLBACK_COMMAND = "navigate to the target object"


class GazeboEnv(gym.Env):
    """
    Full-stack RL environment: Gazebo sim + ROS2 + YOLO/CLIP vision pipeline.

    Usage:
        env = GazeboEnv(n_objects=5)
        obs, info = env.reset(command="go to the red box")
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_objects: int = 5,
        max_episode_steps: int = 500,
        goal_radius: float = 0.5,
        collision_threshold: float = 0.15,
    ):
        """
        Args:
            n_objects: Objects to spawn per episode
            max_episode_steps: Episode length limit
            goal_radius: Distance (meters) that counts as goal reached
            collision_threshold: LIDAR reading (meters) that triggers collision
        """
        super().__init__()
        self.n_objects = n_objects
        self.max_episode_steps = max_episode_steps
        self.goal_radius = goal_radius
        self.collision_threshold = collision_threshold

        # Observation + action spaces (OBS_DIM resolved lazily on first reset)
        _obs_dim = 696  # matches VisionObsBuilder.OBS_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi], dtype=np.float32),
            high=np.array([1.0,  np.pi], dtype=np.float32),
        )

        # Runtime state
        self.robot_pos   = np.zeros(2, dtype=np.float32)
        self.target_pos  = np.zeros(2, dtype=np.float32)
        self.step_count  = 0
        self._command    = _FALLBACK_COMMAND
        self._world_objects: list = []

        # Thread-safe sensor buffers
        self._lock   = threading.Lock()
        self._odom   = None
        self._scan   = None
        self._image  = None

        # Lazy-initialized
        self._ros_initialized = False
        self._bridge = CvBridge()
        self._vis_builder: Optional[VisionObsBuilder] = None

    # ── ROS2 init ───────────────────────────────────────────────────────────

    def _init_ros(self):
        """Initialize ROS2 node and vision pipeline (called once on first reset)."""
        if self._ros_initialized:
            return

        # Lazy ROS2 imports — fail here (not at module load) if rclpy missing
        import rclpy as _rclpy_mod
        from rclpy.node import Node as _NodeCls
        from geometry_msgs.msg import Twist as _TwistCls
        from sensor_msgs.msg import LaserScan as _LaserScanCls, Image as _ImageCls
        from nav_msgs.msg import Odometry as _OdometryCls
        from cv_bridge import CvBridge as _CvBridgeCls
        from langnav_vision import VisionObsBuilder as _VisionObsBuilderCls

        self._bridge = _CvBridgeCls()

        _rclpy_mod.init()
        self._node = _NodeCls("gazebo_env")
        self._world_gen = WorldGenerator()
        self._vis_builder = _VisionObsBuilderCls(image_w=640, image_h=480)

        self._cmd_pub = self._node.create_publisher(_TwistCls, "/cmd_vel", 10)

        qos = _rclpy_mod.qos.QoSProfile(depth=1)
        self._node.create_subscription(_OdometryCls, "/odom",       self._on_odom,  qos)
        self._node.create_subscription(_LaserScanCls, "/scan",      self._on_scan,  qos)
        self._node.create_subscription(_ImageCls, "/camera/rgb",    self._on_image, qos)

        threading.Thread(target=_rclpy_mod.spin, args=(self._node,), daemon=True).start()

        self._rclpy = _rclpy_mod
        self._Twist = _TwistCls
        self._ros_initialized = True

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int = None,
        options: dict = None,
        command: str = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset episode.

        Args:
            seed: RNG seed
            options: Unused (Gymnasium compatibility)
            command: Natural language navigation target (e.g., "go to the red box").
                     Reuses previous command if None.

        Returns:
            (observation, info)
        """
        self._init_ros()

        if seed is not None:
            np.random.seed(seed)

        if command is not None:
            self._command = command

        self.step_count = 0
        self.robot_pos  = np.zeros(2, dtype=np.float32)

        # Randomize Gazebo world
        self._world_objects = self._world_gen.randomize(n_objects=self.n_objects)

        # Pick first object as target (command selects semantically at runtime)
        if self._world_objects:
            chosen = self._world_objects[0]
            pos = chosen["position"]
            self.target_pos = np.array([pos[0], pos[1]], dtype=np.float32)

        # Wait for all sensors to have at least one reading
        self._wait_for_sensors(timeout=5.0)

        obs, vis_info = self._build_obs()
        info = {**vis_info, "objects": self._world_objects, "command": self._command}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute velocity command, advance simulation one step, return state.

        Args:
            action: [linear_vel, angular_vel]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self._publish_cmd(action)
        time.sleep(0.1)  # Wait one control tick (10 Hz)

        # Update robot position from odometry
        with self._lock:
            if self._odom is not None:
                p = self._odom.pose.pose.position
                self.robot_pos = np.array([p.x, p.y], dtype=np.float32)

        self.step_count += 1

        # Build observation (runs YOLO+CLIP on latest camera frame)
        obs, vis_info = self._build_obs()

        # Reward shaping
        distance = float(np.linalg.norm(self.robot_pos - self.target_pos))
        reward, terminated = self._compute_reward(distance, vis_info)

        truncated = self.step_count >= self.max_episode_steps

        info = {
            **vis_info,
            "distance":  distance,
            "success":   terminated and distance < self.goal_radius,
            "step":      self.step_count,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._ros_initialized:
            self._stop_robot()
            self._rclpy.shutdown()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_obs(self) -> Tuple[np.ndarray, dict]:
        """Snapshot sensors, run vision pipeline, return (obs_vector, vis_info)."""
        with self._lock:
            image = self._image.copy() if self._image is not None else None
            scan  = np.array(self._scan.ranges, dtype=np.float32) if self._scan else None

        if image is None:
            # No camera frame yet — return zero obs
            return np.zeros(OBS_DIM, dtype=np.float32), {"detected": False, "semantic_match": 0.0}

        obs, vis_info = self._vis_builder.build(image, self._command, scan)
        return obs, vis_info

    def _compute_reward(self, distance: float, vis_info: dict) -> Tuple[float, bool]:
        """
        Reward function:
          - Dense: negative distance (agent learns to get closer)
          - Semantic bonus: scales with CLIP match score
          - Goal bonus: +10 on arrival
          - Collision penalty: -2 on proximity trigger
          - Direction bonus: pixel alignment to target center
        """
        reward = -distance * 0.1  # Scale down to keep reward in [-1, 1] range
        terminated = False

        # Semantic alignment bonus — encourage facing the right object
        sem_match = float(vis_info.get("semantic_match", 0.0))
        reward += sem_match * 0.2

        # Pixel centering bonus — encourage keeping target in frame center
        if vis_info.get("detected"):
            px = float(vis_info.get("pixel_x", 0.0)) if "pixel_x" in vis_info else 0.0
            center_bonus = max(0.0, 0.1 - abs(px) * 0.1)
            reward += center_bonus

        # Goal reached
        if distance < self.goal_radius:
            reward += 10.0
            terminated = True
            self._stop_robot()
            return reward, terminated

        # Collision via LIDAR
        with self._lock:
            scan = self._scan
        if scan is not None:
            min_range = float(np.min(scan.ranges))
            if min_range < self.collision_threshold:
                reward -= 2.0
                terminated = True
                self._stop_robot()

        return reward, terminated

    def _publish_cmd(self, action: np.ndarray):
        cmd = self._Twist()
        cmd.linear.x  = float(np.clip(action[0], -1.0, 1.0))
        cmd.angular.z = float(np.clip(action[1], -np.pi, np.pi))
        self._cmd_pub.publish(cmd)

    def _stop_robot(self):
        self._cmd_pub.publish(self._Twist())

    def _wait_for_sensors(self, timeout: float = 5.0):
        """Block until odom + scan + image all have at least one reading."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                ready = all([self._odom, self._scan, self._image is not None])
            if ready:
                return
            time.sleep(0.05)

    # ── ROS2 callbacks ───────────────────────────────────────────────────────

    def _on_odom(self, msg):
        with self._lock:
            self._odom = msg

    def _on_scan(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, 0.0, msg.range_max)
        with self._lock:
            self._scan = msg
            self._scan.ranges = ranges.tolist()

    def _on_image(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            with self._lock:
                self._image = frame
        except Exception:
            pass
