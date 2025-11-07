# Container-based Development

This repo is designed for container-first development with ROS 2 Humble and Gazebo Harmonic. You can
develop entirely inside a dev container with VS Code. 

---

## Prerequisites

- Docker and Docker Compose
- VS Code with the Dev Containers extension

---

## Setup the development environment

1. Open the project folder in VS Code
2. Run "Dev Containers: Rebuild and Reopen in Container"
3. Have fun!

---

## Build and run the simulation

Run these commands inside the container shell:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source ./install/setup.bash
```

Launch StalkerDrone (Gazebo + RViz + Navigator):

```bash
ros2 launch sdrl_bringup stalker_drone_launch.py \
  use_rviz:=true use_gui:=true ball_speed:=1.0 ball_trajectory:=circle control_mode:=geometric
```

Key launch arguments (`sdrl_bringup/launch/stalker_drone_launch.py`):
- `use_gui` (true|false): start Gazebo client GUI
- `use_rviz` (true|false): start RViz with preset config
- `ball_speed`: (m/s)
- `ball_trajectory` (circle|random)
- `control_mode` (geometric|rl)

---

## Common topics and services

ROS 2 topics:
- `/X3/gt_odom` (nav_msgs/Odometry): drone ground-truth odometry
- `/X3/cmd_odom` (nav_msgs/Odometry): desired pose/twist from Navigator
- `/X3/ros/motor_speed` (std_msgs/Float32MultiArray): motor command from controller
- `/X3/navi_state` (std_msgs/String): Navigator state machine
- `/X3/ros_bottom_cam/image_raw` (sensor_msgs/Image): bottom camera image
- `/X3/ros_bottom_cam/pose` (geometry_msgs/PoseStamped): camera pose

Service:
- `/X3/reset_drone_initial_pose` (std_srvs/Trigger): resets the drone to its initial pose

Gazebo services (examples):
- `/world/ground_plane_world/create` (spawn entities)
- `/world/ground_plane_world/control` (pause/unpause)

---

## Debug

```bash
ros2 topic list | grep X3
ros2 topic echo /X3/gt_odom --once
ros2 topic echo /X3/navi_state --once

gz topic -l | grep motor_speed
gz topic -i -t /model/X3/gazebo/command/motor_speed
gz topic -t /X3/gazebo/command/motor_speed --msgtype gz.msgs.Actuators -p 'velocity:[200,200,200,200]'
gz service -l | head
gz service -i -s /world/ground_plane_world/control

ros2 service call /X3/reset_drone_initial_pose std_srvs/srv/Trigger {}
```

---

<!-- ## Reinforcement Learning
```bash
source /opt/ros/humble/setup.bash \
&& colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
&& source ./install/setup.bash \
&& ros2 launch sdrl_bringup train_sac_launch.py \
  use_rviz:=true use_gui:=true control_mode:=rl

source /opt/ros/humble/setup.bash \
&& colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
&& source ./install/setup.bash \
&& ros2 launch sdrl_bringup train_sac_launch.py \
  use_rviz:=false use_gui:=false control_mode:=rl

tensorboard --logdir ./tb_logs/SAC_0 --port 6006 --bind_all
``` -->

## Notes on environment variables

Set automatically by the launch/Docker image:
- `GZ_SIM_RESOURCE_PATH`, `GZ_FILE_PATH`: Gazebo model/asset lookup paths
- `GZ_SIM_SYSTEM_PLUGIN_PATH`: where the quadcopter system plugin `.so` is discovered

---
