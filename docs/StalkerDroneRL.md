### Purpose
This project simulates a drone in a physical environment using ROS 2 Humble and Gazebo Harmonic.

The deep RL framework I choose is Gymnasium and Stable-Baselines3.

In this project, a drone autonomously tracks a moving red ball. It first detects the 3D position of
the red ball, compute desired pose and twist, and then send the odometry command to the geometric
controller. The geometric controller computes and returns the rotor speeds.

The ultimate goal of this project is to implement a reinforcement learning-based controller to
control the drone. This RL controller will replace the geometric controller.

I implement a RL training script `src/sdrl_rl_controller/sdrl_rl_controller/train_sac.py` which can
train a SAC controller succesfully. However, I think the overall project can be polished and
modularized better.

### Known issues

#### SAC algorithm
When training the SAC controller with "train_sac.py", the reward is slowly increasing and reach the
max value around 450K steps (1 million steps in total). Based on the total reward (return) value, i
can see that the drone learns how to get closer to the command positions.
Once it reaches command positions, it gets a big reward.

As soon as the drone reaches the command positions, it gets a big reward. Then the command positions
change based on the position of the moving red ball. Then the reward of the current time step drops
significantly (because the current position is no longer the command position). I think this behaviour
confuses the SAC agent. Do you agree?

I use curriculum learning on the random initial pose. the initial random ranges starts from at the initial
learning phase and gradually increases to 3.0. Do you agree this is a good idea?

Analyze my entire codebase and suggest improvement.

#### Use simulation time

The entire codebase is designed to use simulation time. In theory I believe this means the python module
`import time` shall not be used becuase it uses the wall-clock time from the operating system.

But currently it's used in `src/sdrl_rl_controller/sdrl_rl_controller/train_sac.py`.

Do you agree that `import time` should be avoided entirely in  `src/sdrl_rl_controller/sdrl_rl_controller/train_sac.py`?

Analyze my entire codebase and suggest improvement.

#### Reset drone navigator and dynamics

Currently, the drone navigator and dynamics are reset in the `reset` function in `src/sdrl_rl_controller/sdrl_rl_controller/train_sac.py`.
But I notice that the gz sim shows the drone has a large initial velocity after the reset.
The drone's position is reset, but the drone will move abruptly to a random direction. This behaviour looks
like there is left-over velocity from the previous episode.

Check the `reset()` function and all the related and subsequent code to see if my reset procedure is
correctly implemented.


### Design Philosophy
- We want to decouple simulation time and wall clock time.

Normally, the ROS client libraries will use your computer's system clock as a time source, also
known as the "wall-clock" or "wall-time" (like the clock on the wall of your lab). When you are
running a simulation or playing back logged data, however, it is often desirable to instead have the
system use a simulated clock so that you can have accelerated, slowed, or stepped control over your
system's perceived time. For example, if you are playing back sensor data into your system, you may
wish to have your time correspond to the timestamps of the sensor data.

To support this, the ROS client libraries can listen to the /clock topic that is used to publish
"simulation time".

In order for your code to take advantage of the ROS simulation time, it is important that all code
use the appropriate ROS client library Time API for accessing time and sleeping instead of using the
language-native routines. This will allow your system to have a consistent time measurement whether
it is using the wall-clock or a simulated clock. These APIs are described briefly below, though you
should familiarize yourself with your client library of choice for more details.

In order for a ROS node to use simulation time according to the /clock topic, the /use_sim_time
parameter must be set to true before the node is initialized. This can be done in a launchfile or
from the command line.

If the /use_sim_time parameter is set, the ROS Time API will return time=0 until it has received a
value from the /clock topic. Then, the time will only be updated on receipt of a message from the
/clock topic, and will stay constant between updates.

For calculations of time durations when using simulation time, clients should always wait until the
first non-zero time value has been received before starting, because the first simulation time value
from /clock topic may be a high value.

A Clock Server is any node that publishes to the /clock topic, and there should never be more than
one running in a single ROS network. In most cases, the Clock Server is either a simulator or a log
playback tool.

In order to resolve any issues with startup order, it is important that the /use_sim_time Parameter
is set to true in any launch files using a Clock Server. If you are playing back a bag file with
rosbag play, using the --clock option will run a Clock Server while the bag file is being played.

### Guide
- Firstly, always read `README.md` and `DEVELOP.md` to understand the project's overall structure.
- Secondly, always read the docstring of every module to understand the details.

### Coding conventions (C++)
- Always use ROS 2 Humble and Gazebo Harmonic.
- Follow the existing style in this repository.

### Coding conventions (Python)
- Always use ROS 2 Humble and Gazebo Harmonic.
- Follow the existing style in this repository.
