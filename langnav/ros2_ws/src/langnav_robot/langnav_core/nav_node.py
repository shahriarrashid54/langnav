"""
ROS2 orchestration node.

Subscriptions:
  /camera/rgb    (sensor_msgs/Image)   — RGB camera frames
  /scan          (sensor_msgs/LaserScan)
  /command       (std_msgs/String)     — natural language command

Publications:
  /cmd_vel       (geometry_msgs/Twist) — velocity commands
  /detected_target (std_msgs/String)   — JSON detection result
"""

import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langnav_vision import VisionObsBuilder


class NavNode(Node):
    """
    Core navigation node.

    Vision pipeline runs on every control tick (10 Hz).
    RL policy inference slot is prepared but requires a loaded PPO checkpoint.
    """

    def __init__(self):
        super().__init__("langnav_node")

        self._bridge   = CvBridge()
        self._lock     = threading.Lock()
        self._image    = None
        self._scan     = None
        self._command  = "navigate to the target object"

        # Vision pipeline (YOLO + CLIP)
        self.get_logger().info("Loading vision pipeline (YOLO + CLIP)...")
        self._vis_builder = VisionObsBuilder(image_w=640, image_h=480)
        self.get_logger().info("Vision pipeline ready.")

        # PPO model — loaded from checkpoint if path provided via param
        self._ppo_model = None
        self._load_ppo_if_available()

        qos = QoSProfile(depth=1)

        # Subscriptions
        self.create_subscription(Image,     "/camera/rgb", self._on_image,   qos)
        self.create_subscription(LaserScan, "/scan",       self._on_scan,    qos)
        self.create_subscription(String,    "/command",    self._on_command, qos)

        # Publications
        self._cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",         10)
        self._target_pub = self.create_publisher(String, "/detected_target", 10)

        # 10 Hz control loop
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info("NavNode initialized. Waiting for sensors...")

    # ── ROS2 callbacks ───────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            with self._lock:
                self._image = frame
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")

    def _on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, 0.0, msg.range_max)
        with self._lock:
            self._scan = msg
            self._scan.ranges = ranges.tolist()

    def _on_command(self, msg: String):
        with self._lock:
            self._command = msg.data
        self.get_logger().info(f"Command: {msg.data}")

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        10 Hz tick:
        1. Run YOLO + CLIP on latest frame
        2. Publish detection result
        3. Feed obs to PPO policy → get velocity action
        4. Publish /cmd_vel
        """
        with self._lock:
            image   = self._image.copy() if self._image is not None else None
            scan    = np.array(self._scan.ranges, dtype=np.float32) if self._scan else None
            command = self._command

        if image is None:
            return

        # Vision pipeline
        obs, vis_info = self._vis_builder.build(image, command, scan)

        # Publish detection JSON
        target_msg = String()
        target_msg.data = json.dumps({
            "detected":      vis_info.get("detected", False),
            "target_class":  vis_info.get("target_class"),
            "semantic_match": round(float(vis_info.get("semantic_match", 0.0)), 3),
            "command":       command,
        })
        self._target_pub.publish(target_msg)

        # PPO policy → velocity command
        vel = Twist()
        if self._ppo_model is not None:
            action, _ = self._ppo_model.predict(obs, deterministic=False)
            vel.linear.x  = float(np.clip(action[0], -1.0, 1.0))
            vel.angular.z = float(np.clip(action[1], -np.pi, np.pi))
        else:
            # No model loaded: rotate slowly to scan environment
            vel.angular.z = 0.3

        self._cmd_pub.publish(vel)

    # ── PPO loader ───────────────────────────────────────────────────────────

    def _load_ppo_if_available(self):
        """Load PPO checkpoint from ROS2 parameter 'model_path' if set."""
        self.declare_parameter("model_path", "")
        path = self.get_parameter("model_path").get_parameter_value().string_value

        if not path:
            self.get_logger().warn("No model_path param — running without PPO policy.")
            return

        try:
            from stable_baselines3 import PPO
            self._ppo_model = PPO.load(path)
            self.get_logger().info(f"PPO model loaded from {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load PPO model: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
