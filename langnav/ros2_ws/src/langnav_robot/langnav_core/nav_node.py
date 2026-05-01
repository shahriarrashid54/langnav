"""
ROS2 node: Listen to commands, process vision, execute navigation.

Subscribes to:
  - /camera/rgb (sensor_msgs/Image)
  - /command (std_msgs/String)

Publishes to:
  - /cmd_vel (geometry_msgs/Twist)
  - /detected_target (std_msgs/String)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# TODO: Import vision pipeline + RL model once ROS env setup


class NavNode(Node):
    """Main navigation orchestration node."""

    def __init__(self):
        super().__init__("langnav_node")

        self.bridge = CvBridge()
        self.last_image = None
        self.last_command = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/camera/rgb",
            self.on_image,
            qos_profile=rclpy.qos.QoSProfile(depth=1)
        )

        self.command_sub = self.create_subscription(
            String,
            "/command",
            self.on_command,
            qos_profile=rclpy.qos.QoSProfile(depth=1)
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.target_pub = self.create_publisher(String, "/detected_target", 10)

        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("NavNode initialized")

    def on_image(self, msg: Image):
        """Callback for camera images."""
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def on_command(self, msg: String):
        """Callback for navigation commands."""
        self.last_command = msg.data
        self.get_logger().info(f"Command received: {msg.data}")

    def control_loop(self):
        """
        Main control loop:
        1. Run vision pipeline (YOLO + CLIP)
        2. Get RL policy action
        3. Publish velocity command
        """
        if self.last_image is None or self.last_command is None:
            return

        try:
            # TODO: Run vision pipeline
            # vision_result = vision_pipeline.process(self.last_image, self.last_command)

            # TODO: Get RL policy action
            # obs = create_obs_from_vision(vision_result)
            # action, _ = ppo_model.predict(obs, deterministic=False)

            # Placeholder: just publish zero velocity
            vel = Twist()
            vel.linear.x = 0.0
            vel.angular.z = 0.0
            self.cmd_vel_pub.publish(vel)

            # Publish detected target
            target_msg = String()
            target_msg.data = "target_detected"  # TODO: Update with actual detection
            self.target_pub.publish(target_msg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
