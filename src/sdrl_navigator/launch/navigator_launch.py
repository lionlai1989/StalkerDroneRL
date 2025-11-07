from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    navigator_node = Node(
        package="sdrl_navigator",
        executable="navigator",
        name="navigator",
        output="screen",
        parameters=[{"control_mode": LaunchConfiguration("control_mode")}],
    )

    return LaunchDescription([navigator_node])
