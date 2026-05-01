"""
Randomize Gazebo world at episode start.
Spawns/deletes objects via ROS2 service calls to gazebo_ros.
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
import random
import math
from typing import List, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET


# Object palette: name, RGBA color, size (x, y, z)
@dataclass
class ObjectSpec:
    name: str
    color: Tuple[float, float, float, float]
    size: Tuple[float, float, float]


OBJECT_PALETTE = [
    ObjectSpec("red_box",    (0.8, 0.1, 0.1, 1.0), (0.5, 0.5, 0.5)),
    ObjectSpec("blue_box",   (0.1, 0.1, 0.8, 1.0), (0.5, 0.5, 0.5)),
    ObjectSpec("green_box",  (0.1, 0.7, 0.1, 1.0), (0.5, 0.5, 0.5)),
    ObjectSpec("yellow_box", (0.9, 0.8, 0.1, 1.0), (0.5, 0.5, 0.5)),
    ObjectSpec("tall_post",  (0.5, 0.3, 0.1, 1.0), (0.2, 0.2, 1.2)),
    ObjectSpec("flat_table", (0.6, 0.4, 0.2, 1.0), (1.0, 0.6, 0.1)),
    ObjectSpec("cylinder",   (0.2, 0.6, 0.8, 1.0), (0.4, 0.4, 0.8)),
]


class WorldGenerator(Node):
    """Spawn/delete objects in Gazebo world to randomize training scenes."""

    def __init__(self):
        super().__init__("world_generator")

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")

        self.spawned_objects: List[str] = []

    def randomize(
        self,
        n_objects: int = 5,
        room_bounds: Tuple[float, float] = (-4.0, 4.0),
        min_dist_from_origin: float = 1.0,
    ) -> List[dict]:
        """
        Delete all current objects and spawn N random ones.

        Args:
            n_objects: Number of objects to spawn
            room_bounds: (min, max) for X and Y placement
            min_dist_from_origin: Keep robot spawn area clear

        Returns:
            List of spawned object metadata
        """
        self._delete_all()
        positions = self._sample_positions(n_objects, room_bounds, min_dist_from_origin)

        spawned = []
        for i, pos in enumerate(positions):
            spec = random.choice(OBJECT_PALETTE)
            obj_name = f"{spec.name}_{i}"
            sdf = self._build_sdf(obj_name, spec, pos)
            self._spawn_object(obj_name, sdf, pos)
            self.spawned_objects.append(obj_name)

            spawned.append({
                "name": obj_name,
                "spec": spec.name,
                "color": spec.color,
                "position": pos,
            })
            self.get_logger().info(f"Spawned {obj_name} at {pos}")

        return spawned

    def _delete_all(self):
        """Delete all previously spawned objects."""
        for name in self.spawned_objects:
            req = DeleteEntity.Request()
            req.name = name
            self.delete_client.call(req)
        self.spawned_objects.clear()

    def _spawn_object(self, name: str, sdf: str, pos: Tuple[float, float]):
        """Call Gazebo spawn service."""
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.initial_pose = Pose()
        req.initial_pose.position.x = pos[0]
        req.initial_pose.position.y = pos[1]
        req.initial_pose.position.z = 0.0
        self.spawn_client.call(req)

    @staticmethod
    def _sample_positions(
        n: int,
        bounds: Tuple[float, float],
        min_dist: float,
        min_spacing: float = 1.2,
    ) -> List[Tuple[float, float]]:
        """Sample non-overlapping positions inside room bounds."""
        positions = []
        attempts = 0

        while len(positions) < n and attempts < 500:
            x = random.uniform(bounds[0], bounds[1])
            y = random.uniform(bounds[0], bounds[1])

            # Keep origin clear for robot spawn
            if math.sqrt(x**2 + y**2) < min_dist:
                attempts += 1
                continue

            # Maintain minimum spacing between objects
            too_close = any(
                math.sqrt((x - px)**2 + (y - py)**2) < min_spacing
                for px, py in positions
            )
            if too_close:
                attempts += 1
                continue

            positions.append((x, y))
            attempts = 0

        return positions

    @staticmethod
    def _build_sdf(name: str, spec: ObjectSpec, pos: Tuple[float, float]) -> str:
        """Generate SDF XML for a box object with given color and size."""
        r, g, b, a = spec.color
        sx, sy, sz = spec.size
        half_z = sz / 2.0

        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <pose>{pos[0]} {pos[1]} {half_z} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.083</ixx><iyy>0.083</iyy><izz>0.083</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
