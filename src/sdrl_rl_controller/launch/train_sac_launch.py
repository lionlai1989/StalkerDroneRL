from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time",
    )

    # RL training node
    rl_training_node = Node(
        package="sdrl_rl_controller",
        executable="train_sac",
        name="sac_trainer",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )
    shutdown_on_rl_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=rl_training_node,
            on_exit=[EmitEvent(event=Shutdown(reason="RL training finished"))],
        )
    )
    return LaunchDescription([use_sim_time_arg, rl_training_node, shutdown_on_rl_exit])
