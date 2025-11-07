# StalkerDroneRL - ROS 2 Humble & Gazebo Harmonic Drone Simulation

**StalkerDroneRL** is a fully containerized ROS 2 + Gazebo project that simulates a quadcopter
*stalking a red ball*.

<video width="640" controls>
  <source src="https://github.com/user-attachments/assets/a0ed7a53-b3c4-4faa-8ca0-1d236d85ee02" type="video/webm">
</video>

---

## Highlights

- **ROS 2 Humble + Gazebo Harmonic** — uses recent, long-term support versions.

- **Completely containerized** — I went through the *installation hell* of ROS 2 and Gazebo so you
  don't have to. One `docker compose up` and everything just works.

- **Everything built from scratch** — all algorithms here are built from scratch, ranging from the geometric
  controller (output low-level motor speed commands) to object detection. This is not a "just tune
  some PID gains" project.

- **Generic and modular** — the drone, camera, and sensor components come directly from [Open
  Robotics](https://www.openrobotics.org/). You can tweak parameters like mass, inertia, rotor
  speeds, or the camera's FOV for your hardware platform.

---


## Architecture

There are seven ROS packages in `src/`:
```
src/
  sdrl_bringup/
  sdrl_geometric_controller/
  sdrl_lionquadcopter/
  sdrl_navigator/
  sdrl_object_mover/
  sdrl_perception/
  sdrl_rl_controller/
```
- **sdrl_bringup**: Launches the entire simulation environment and orchestrates all ROS processes.

- **sdrl_geometric_controller**: Implements a low-level geometric PD controller.

- **sdrl_lionquadcopter**: Gazebo plugin for the `lion_quadcopter.sdf` model.

- **sdrl_navigator**: High‑level navigator that controls the drone's overall structure and state
  machine.

- **sdrl_object_mover**: Controls the red ball.

- **sdrl_perception**: A simple object detection package for the red ball.

- **sdrl_rl_controller**: Reinforcement learning package that trains an RL‑based controller. This is
  a work in progress.

---

## How to run

1. Build the Docker image
```
docker compose build dev --no-cache
```

2. Spawn the container
```
docker compose up -d dev
```

3. Log in to the container
```
docker exec -it stalkerdronerl_dev bash
```

4. Launch the StalkerDrone with the geometric controller
```bash
source /opt/ros/humble/setup.bash \
&& colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
&& source "./install/setup.bash" \
&& ros2 launch sdrl_bringup stalker_drone_launch.py \
use_rviz:=true use_gui:=true ball_speed:=1.0 ball_trajectory:=circle control_mode:=geometric
```

If everything goes well, you should see Gazebo and RViz launch, as shown at the beginning of this
README.

---

## How to develop

All development details are in `DEVELOP.md`.

## Additional Resources

All materials used during development are recorded here.

### ROS 2 Humble and Gazebo Harmonic
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Harmonic Documentation](https://gazebosim.org/docs/harmonic/getstarted/)

### Quadcopter Dynamics and Control
- [Visual Navigation for Autonomous Vehicles (VNAV)](https://vnav.mit.edu/lectures.html)

- Multicopter Flight Control — Johannes Stephan

- Quad Rotorcraft Control: Vision-Based Hovering and Navigation — Luis Rodolfo García Carrillo et al.

- T. Lee, M. Leok, and N. H. McClamroch,
“Geometric tracking control of a quadrotor UAV on SE(3),” IEEE CDC, 2010.

- G. Tang, W. Sun, and K. Hauser,
“Learning Trajectories for Real-Time Optimal Control of Quadrotors,” IROS, 2018.

- E. Tal and S. Karaman,
“Accurate Tracking of Aggressive Quadrotor Trajectories Using Incremental Nonlinear Dynamic Inversion and Differential Flatness,” IEEE T-CST, 2021.

## License and third‑party assets

1. This repository's source code is licensed under Apache License 2.0 (see `LICENSE`).

2. Third‑party model assets live under `src/sdrl_lionquadcopter/models/` and remain under their
original licenses. In particular, the nested `x3_uav` model originates from Open Robotics Gazebo
Fuel (X3 UAV) and is typically provided under Creative Commons Attribution 4.0 (CC‑BY‑4.0).

3. The geometric controller implementation follows the method described in `T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3),"  IEEE CDC, 2010.`
