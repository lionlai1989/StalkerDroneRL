"""
The main launch file launches the whole simulation environment and the SAC training script in the
deterministic order.
"""

from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the directories of the ROS packages
    bringup_path = get_package_share_directory("sdrl_bringup")
    lionquadcopter_path = get_package_share_directory("sdrl_lionquadcopter")
    navigator_path = get_package_share_directory("sdrl_navigator")
    object_mover_path = get_package_share_directory("sdrl_object_mover")
    rl_controller_path = get_package_share_directory("sdrl_rl_controller")

    # Named arguments that can be passed to the launch file from the command line
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        choices=["true", "false"],
        description="Whether to execute gzclient (GUI)",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz for visualization",
    )
    fixed_frame_arg = DeclareLaunchArgument(
        "fixed_frame",
        default_value="world",
        description="Fixed frame for RViz",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use simulated clock (ROS time)",
    )
    ball_speed_arg = DeclareLaunchArgument(
        "ball_speed",
        default_value="0.5",
        description="Linear speed (m/s) for ball along path",
    )
    ball_trajectory_arg = DeclareLaunchArgument(
        "ball_trajectory",
        default_value="circle",
        choices=["circle", "random"],
        description="Trajectory for ball",
    )
    world_arg = DeclareLaunchArgument(
        name="world",
        default_value=str(Path(bringup_path) / "worlds" / "ground_plane.world"),
        description="Path to the Gazebo world file to load",
    )
    control_mode_arg = DeclareLaunchArgument(
        "control_mode",
        choices=["geometric", "rl"],
        description="Navigator control mode (geometric or rl)",
    )

    # Ensure Gazebo can find our system plugin and model resources
    plugin_dir = Path(get_package_prefix("sdrl_lionquadcopter")) / "lib"
    lion_quadcopter_models_dir = Path(lionquadcopter_path) / "models"
    object_mover_models_dir = Path(object_mover_path) / "models"

    # Gazebo environment variables
    gazebo_env_vars = [
        SetEnvironmentVariable(name="GZ_SIM_SYSTEM_PLUGIN_PATH", value=f"{plugin_dir}"),
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=f"{lion_quadcopter_models_dir}:{object_mover_models_dir}",
        ),
        SetEnvironmentVariable(
            name="GZ_FILE_PATH",
            value=f"{lion_quadcopter_models_dir}:{object_mover_models_dir}",
        ),
    ]

    # Gazebo simulation with GUI and run it unpaused immediately
    gazebo_gui = ExecuteProcess(
        cmd=["gz", "sim", "-r", LaunchConfiguration("world")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_gui")),
    )
    # Gazebo simulation without GUI and run it unpaused immediately
    gazebo_headless = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-s", LaunchConfiguration("world")],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("use_gui")),
    )

    # Spawn red_ball model dynamically. 0.15 is the ball radius.
    red_ball_sdf = object_mover_models_dir / "red_ball.sdf"
    red_ball_pose = (
        "pose: { position: { x: 0, y: -3, z: 0.15 }, orientation: { w: 1, x: 0, y: 0, z: 0 } }"
    )
    spawn_ball = ExecuteProcess(
        cmd=[
            "gz",
            "service",
            "-s",
            "/world/ground_plane_world/create",
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            f'sdf_filename: "{red_ball_sdf}" name: "red_ball" {red_ball_pose}',
        ],
        output="screen",
    )

    # Include object mover launch
    mover_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{Path(object_mover_path) / 'launch' / 'move_object_launch.py'}",
        ),
        launch_arguments={
            "speed": LaunchConfiguration("ball_speed"),
            "trajectory": LaunchConfiguration("ball_trajectory"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    model_ns = "/X3"
    print("namespace: ", model_ns)
    # TF static transform publisher (anchor odom to world for visualization/tools)
    tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "world",
            "--child-frame-id",
            f"{model_ns}/odom",
        ],
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            f"{Path(bringup_path) / 'rviz' / 'stalker_drone.rviz'}",
            "--fixed-frame",
            LaunchConfiguration("fixed_frame"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    navigator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{Path(navigator_path) / 'launch' / 'navigator_launch.py'}",
        ),
        launch_arguments={
            "control_mode": LaunchConfiguration("control_mode"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # Call a Gazebo service to spawn an entity in the world dynamically after a delay to make sure
    # the world is fully loaded before trying to insert the drone (avoid race conditions).
    # Spawning at (0, 0, 0) sits the drone on the ground because the nested X3 model has a
    # model-level +Z pose of 0.053302 m (see src/sdrl_lionquadcopter/models/x3_uav/model.sdf). The
    # base_link collision box is 0.11 m tall, so its origin is near mid-height. The model-level
    # offset lifts it so the bottom touches z=0. If that offset were removed, we'd need to spawn at
    # z=0.053302 for perfect ground contact.
    lion_sdf_path = Path(lionquadcopter_path) / "models" / "lion_quadcopter.sdf"
    lion_sdf_pose = (
        "pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }"
    )
    spawn_drone = ExecuteProcess(
        cmd=[
            "gz",
            "service",
            "-s",
            "/world/ground_plane_world/create",
            "--reqtype",
            "gz.msgs.EntityFactory",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            f'sdf_filename: "{lion_sdf_path}" name: "lion_quadcopter" {lion_sdf_pose}',
        ],
        output="screen",
    )

    rl_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{Path(rl_controller_path) / 'launch' / 'train_sac_launch.py'}",
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # Auto-start simulation by unpausing the gz sim (applies to both GUI and headless)
    # NOTE: No longer needed since we start Gazebo with the -r flag (unpaused)
    unpause_world = ExecuteProcess(
        cmd=[
            "gz",
            "service",
            "-s",
            "/world/ground_plane_world/control",
            "--reqtype",
            "gz.msgs.WorldControl",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            "pause: false",
        ],
        output="screen",
    )

    # Bridge Gazebo simulation clock to ROS 2 /clock.
    # Syntax for parameter_bridge argument:
    # - /TOPIC_NAME@ROS_MSG_TYPE[GZ_MSG_TYPE
    # - The first '@' separates the topic name from the type specification. Immediately after '@'
    #   comes the ROS message type.
    # - The ROS type is followed by one of '@', '[', or ']' to indicate direction:
    #     '@'  -> bidirectional bridge (ROS <-> Gazebo)
    #     '['  -> Gazebo -> ROS (subscribe in Gazebo, publish to ROS)
    #     ']'  -> ROS -> Gazebo (subscribe in ROS, publish to Gazebo)
    # Remap the Gazebo clock topic name to the standard ROS 2 /clock topic.
    # ROS 2's use_sim_time mechanism always listens on /clock, so this makes
    # all nodes that set use_sim_time:=true automatically follow Gazebo time
    # without hard-coding the Gazebo world-specific topic name.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/world/ground_plane_world/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        remappings=[("/world/ground_plane_world/clock", "/clock")],
        output="screen",
    )

    # Deterministic startup ordering via timers. It does not ensure the readiness of the services.
    # TimerAction is part of the ROS 2 launch system, not a ROS node. Thus, it uses wall-clock time.
    delayed_spawn_ball = TimerAction(period=1.5, actions=[spawn_ball])
    delayed_mover = TimerAction(period=2.0, actions=[mover_launch])
    delayed_tf_publisher = TimerAction(period=2.5, actions=[tf_publisher])
    delayed_rviz_node = TimerAction(period=3.0, actions=[rviz_node])
    delayed_navigator_launch = TimerAction(period=3.5, actions=[navigator_launch])
    delayed_spawn_drone = TimerAction(period=4.0, actions=[spawn_drone])
    delayed_unpause_world = TimerAction(period=5.0, actions=[unpause_world])
    delayed_rl_training = TimerAction(period=5.5, actions=[rl_launch])

    return LaunchDescription(
        [
            # Launch arguments
            use_sim_time_arg,
            use_gui_arg,
            use_rviz_arg,
            fixed_frame_arg,
            world_arg,
            control_mode_arg,
            ball_speed_arg,
            ball_trajectory_arg,
            # Gazebo setup
            *gazebo_env_vars,
            # Gazebo simulation
            gazebo_gui,
            gazebo_headless,
            # Deterministic sequencing
            delayed_spawn_ball,
            delayed_mover,
            delayed_tf_publisher,
            delayed_rviz_node,
            delayed_navigator_launch,
            delayed_spawn_drone,
            delayed_rl_training,
            clock_bridge,
        ]
    )
