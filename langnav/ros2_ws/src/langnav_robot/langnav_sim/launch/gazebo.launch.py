"""Launch Gazebo world + spawn TurtleBot3 + bring up NavNode."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory("langnav_robot")
    world_file = os.path.join(pkg_share, "worlds", "langnav_room.world")

    # ── Arguments ──────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz",         default_value="false"),
        DeclareLaunchArgument("world",        default_value=world_file),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz_enabled = LaunchConfiguration("rviz")
    world        = LaunchConfiguration("world")

    # ── Gazebo ─────────────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=["gazebo", "--verbose", world, "-s", "libgazebo_ros_factory.so"],
        output="screen",
    )

    # ── Robot description (URDF → /robot_description) ──────────────────────
    urdf_file = os.path.join(pkg_share, "urdf", "turtlebot3_langnav.urdf")
    with open(urdf_file) as f:
        robot_description_content = f.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "robot_description": robot_description_content,
        }],
    )

    # ── Spawn robot in Gazebo ──────────────────────────────────────────────
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "turtlebot3_langnav",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.01",
        ],
        output="screen",
    )

    # ── Nav2 stack ─────────────────────────────────────────────────────────
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"),
                "launch",
                "navigation_launch.py",
            ])
        ]),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # ── LangNav core node ──────────────────────────────────────────────────
    langnav_node = Node(
        package="langnav_robot",
        executable="nav_node",
        name="langnav_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ── RViz (optional) ────────────────────────────────────────────────────
    rviz_config = os.path.join(pkg_share, "config", "langnav.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(rviz_enabled),
        output="screen",
    )

    return LaunchDescription(
        args + [
            gazebo,
            robot_state_publisher,
            spawn_robot,
            nav2_launch,
            langnav_node,
            rviz_node,
        ]
    )
