from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # RL training node
    rl_training_node = Node(
        package="sdrl_rl_controller",
        executable="train_sac",
        name="sac_trainer",
        output="screen",
    )
    shutdown_on_rl_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=rl_training_node,
            on_exit=[EmitEvent(event=Shutdown(reason="RL training finished"))],
        )
    )
    return LaunchDescription([rl_training_node, shutdown_on_rl_exit])
