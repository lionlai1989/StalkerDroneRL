from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    navigator_node = Node(
        package="sdrl_navigator",
        executable="navigator",
        name="navigator",
        output="screen",
        parameters=[
            {
                "control_mode": LaunchConfiguration("control_mode"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
    )

    return LaunchDescription([navigator_node])
