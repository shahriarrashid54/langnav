"""Gazebo simulation: world generation, robot URDF, ROS2 launch, GazeboEnv."""

from .worlds.world_generator import WorldGenerator

# GazeboEnv requires rclpy (ROS2) — only available inside a ROS2 environment
try:
    from .gazebo_env import GazeboEnv
    __all__ = ["GazeboEnv", "WorldGenerator"]
except ImportError:
    __all__ = ["WorldGenerator"]
