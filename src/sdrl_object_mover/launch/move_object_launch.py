from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Arguments mirrored to node parameters
    args = {
        "entity_name": DeclareLaunchArgument("entity_name", default_value="red_ball"),
        "world_name": DeclareLaunchArgument("world_name", default_value="ground_plane_world"),
        "center_x": DeclareLaunchArgument("center_x", default_value="0.0"),
        "center_y": DeclareLaunchArgument("center_y", default_value="0.0"),
        "center_z": DeclareLaunchArgument("center_z", default_value="0.15"),
        "initial_x": DeclareLaunchArgument("initial_x", default_value="3.0"),
        "initial_y": DeclareLaunchArgument("initial_y", default_value="0.0"),
        "speed": DeclareLaunchArgument("speed", default_value="0.1"),
        "trajectory": DeclareLaunchArgument(
            "trajectory", default_value="circle", choices=["circle", "random"]
        ),
    }

    params = {name: LaunchConfiguration(name) for name in args.keys()}

    mover_node = Node(
        package="sdrl_object_mover",
        executable="move_object",
        name="object_mover",
        output="screen",
        parameters=[params],
    )

    return LaunchDescription(list(args.values()) + [mover_node])
