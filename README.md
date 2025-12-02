# StalkerDroneRL - ROS 2 + Gazebo + Gymnasium + Stable Baselines3 Drone Simulation

**StalkerDroneRL** is a fully containerized **ROS 2** + **Gazebo** project that simulates a quadcopter
*stalking a red ball*, utilizing a deep reinforcement learning controller powered by **Gymnasium**
and **Stable Baselines3**.

[Watch demo](https://github.com/user-attachments/assets/a0ed7a53-b3c4-4faa-8ca0-1d236d85ee02)

---

## Highlights

- **ROS 2 Humble + Gazebo Harmonic + Gymnasium + Stable Baselines3** — The entire software stack uses
  recent, long-term support versions. It is the perfect demo project illustrating how to integrate
  these four open-source frameworks.

- **Containerized environment** — I went through the *installation hell* of ROS 2 and
  Gazebo so you don't have to. During development, all you need is VS Code. For deployment, just run
  `docker compose up` and everything just works.

- **Everything built from scratch** — all algorithms here are implemented from scratch, ranging from the geometric
  controller, perception, tracking, and navigation, to the training and deployment of the RL-based controller.

- **Generic and modular** — the drone, camera, and sensor components come directly from [Open
  Robotics](https://www.openrobotics.org/). You can tweak parameters like mass, inertia, rotor
  speeds, camera FOV, etc., for your hardware platform.

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

- **sdrl_rl_controller**: A Reinforcement Learning package that implements the Gymnasium environment
  and trains the drone controller using Stable Baselines3. 

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

3. Exec into the container
```
docker exec -it stalkerdronerl_dev bash
```

4. Launch the StalkerDroneRL with the geometric controller or Soft Actor-Critic (SAC) controller.

The default SAC model can be downloaded from the
[link](https://drive.google.com/file/d/1wHCf7Fkoc3zCc1ft2Urj3eVZOVITnhos/view?usp=sharing). The SAC
model needs to be placed at the root folder of this project `StalkerDroneRL`.

```bash
# Geometric controller
source /opt/ros/humble/setup.bash \
&& colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
&& source "./install/setup.bash" \
&& ros2 launch sdrl_bringup stalker_drone_launch.py \
  use_rviz:=true use_gui:=true ball_speed:=1.0 ball_trajectory:=circle control_mode:=geometric

# SAC controller
source /opt/ros/humble/setup.bash \
&& colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
&& source "./install/setup.bash" \
&& ros2 launch sdrl_bringup stalker_drone_launch.py \
 use_rviz:=true use_gui:=true ball_speed:=0.5 ball_trajectory:=circle control_mode:=rl
```

If everything goes well, you should see Gazebo and RViz launch, as shown at the beginning of this
README.

---

## How to develop

All development details are in [DEVELOP.md](DEVELOP.md).

---

## References

Materials used during development and recommended reading.

### ROS 2 Humble and Gazebo Harmonic
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Harmonic Documentation](https://gazebosim.org/docs/harmonic/getstarted/)

### Quadcopter Navigation and Control
- [Visual Navigation for Autonomous Vehicles (VNAV)](https://vnav.mit.edu/lectures.html)
- [AA 203: Optimal and Learning-Based Control Course Notes — James Harrison](https://github.com/StanfordASL/AA203-Notes)
- [Quadcopter Dynamics and Simulation — Andrew Gibiansky](https://andrew.gibiansky.com/blog/physics/quadcopter-dynamics/)
- Multicopter Flight Control — Johannes Stephan
- Small Unmanned Aircraft: Theory and Practice — Randal W. Beard and Timothy W. McLain
- Quad Rotorcraft Control: Vision-Based Hovering and Navigation — Luis Rodolfo García Carrillo et al.
- Quadrotor Dynamics and Control Rev 0.1 — Randal Beard
- R. Mahony, V. Kumar and P. Corke, "Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor," in IEEE Robotics & Automation Magazine, vol. 19, no. 3, pp. 20-32, Sept. 2012, doi: 10.1109/MRA.2012.2206474.
- T. Lee, M. Leok and N. H. McClamroch, "Geometric tracking control of a quadrotor UAV on SE(3)," 49th IEEE Conference on Decision and Control (CDC), Atlanta, GA, USA, 2010, pp. 5420-5425, doi: 10.1109/CDC.2010.5717652.
- E. Tal and S. Karaman, "Accurate Tracking of Aggressive Quadrotor Trajectories Using Incremental Nonlinear Dynamic Inversion and Differential Flatness," in IEEE Transactions on Control Systems Technology, vol. 29, no. 3, pp. 1203-1218, May 2021, doi: 10.1109/TCST.2020.3001117.

### Deep Reinforcement Learning and Control
- [CS 285: Deep Reinforcement Learning (UC Berkeley)](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)
- Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
- Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." International conference on machine learning. Pmlr, 2018.
- Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
- Hwangbo, Jemin, et al. "Control of a quadrotor with reinforcement learning." IEEE Robotics and Automation Letters 2.4 (2017): 2096-2103.
- Johannink, Tobias, et al. "Residual reinforcement learning for robot control." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.
- Lambert, Nathan O., et al. "Low-level control of a quadrotor with deep model-based reinforcement learning." IEEE Robotics and Automation Letters 4.4 (2019): 4224-4230.
- Molchanov, Artem, et al. "Sim-to-(multi)-real: Transfer of low-level robust control policies to multiple quadrotors." 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.
- Kaufmann, Elia, Leonard Bauersfeld, and Davide Scaramuzza. "A benchmark comparison of learned control policies for agile quadrotor flight." 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.

---

## How to cite

Use GitHub's “Cite this repository” button on the repository page to get BibTeX, APA, or EndNote entries generated from `CITATION.cff`.

For quick copy, an example one‑line BibTeX is:
```bibtex
@software{StalkerDroneRL, author={Lai, Chih-An Lion}, title={The StalkerDroneRL Project}, year={2025}, version={1.0.0}, url={https://github.com/lionlai1989/StalkerDroneRL}}
```

---

## License and third‑party assets

1. This repository's source code is licensed under Apache License 2.0 (see `LICENSE`).

2. Third‑party model assets live under `src/sdrl_lionquadcopter/models/` remain under their
original licenses. In particular, the nested `x3_uav` model originates from Open Robotics Gazebo
Fuel (X3 UAV) and is typically provided under Creative Commons Attribution 4.0 (CC‑BY‑4.0).
