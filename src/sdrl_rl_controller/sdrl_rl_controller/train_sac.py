"""
SAC Training Script for Quadcopter RL Control.
"""

import argparse
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger

try:
    # Preferred (unversioned) imports if available
    import gz.transport as gz_transport
    from gz.msgs.boolean_pb2 import Boolean
    from gz.msgs.world_control_pb2 import WorldControl
except ModuleNotFoundError:
    # Harmonic Debian packages expose versioned subpackages
    import gz.transport13 as gz_transport
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.world_control_pb2 import WorldControl

from sdrl_geometric_controller.quadcopter_params import QuadcopterParams
from sdrl_geometric_controller.transform import quat_to_rotmat, quat_to_euler, roll_pitch_to_tilt
from sdrl_geometric_controller.motor_mixing import wrench_to_motor_speeds


WORLD_NAME = "ground_plane_world"
_GZ_NODE = gz_transport.Node()


def _gz_request(service: str, req, timeout_ms: int) -> bool:
    """Send a service request using whichever binding signature is available."""
    # Preferred signature (gz.transport13): pass message types
    res = _GZ_NODE.request(service, req, req.__class__, Boolean, int(timeout_ms))
    # Some bindings return (ok, resp), others just resp (Boolean)
    if isinstance(res, tuple):
        ok, resp = res
        return bool(ok and getattr(resp, "data", False))
    return bool(getattr(res, "data", False))


def world_control(
    world: str, *, pause: bool | None = None, step_multi: int | None = None, timeout_ms: int = 10000
) -> None:
    req = WorldControl()
    if pause is not None:
        req.pause = bool(pause)
    if step_multi is not None:
        req.step = True
        req.multi_step = int(step_multi)
    if not _gz_request(f"/world/{world}/control", req, timeout_ms):
        # raise RuntimeError("world_control failed")
        # Do not raise error, just print warning because sometimes it fails but it actually works
        print("world_control failed")


GRAVITY = 9.81
MAX_POS_DIST = 9.0  # 9**2 ~= 5**2 + 5**2 + 5**2
MAX_POS_ERR = 5.0
MAX_VEL_DIST = 5.0  # 5**2 ~= 3**2 + 3**2 + 3**2
MAX_VEL_ERR = 3.0
MAX_ANG_VEL = 6.28  # 2π rad/s ~= 360 deg/s


class QuadcopterTrackingEnv(gym.Env):
    """Enhanced Gymnasium environment for quadcopter RL training."""

    def __init__(
        self,
        control_freq: int = 100,
        max_episode_steps: int = 2000,
        node_name: str = "quadcopter_rl_env",
        use_sim_time: bool = True,
    ):
        super().__init__()

        # Initialize ROS2 node with use_sim_time parameter
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(
            node_name,
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, use_sim_time
                )
            ],
        )

        # Curriculum learning configuration
        self.curriculum_config = {
            "enabled": True,
            "initial_success_radius": 3.0,  # Start with easier target 2
            "final_success_radius": 0.38,  # Final tight tolerance 0.38
            "initial_difficulty_steps": 300_000,  # Steps to start increasing difficulty
            "final_difficulty_steps": 900_000,  # Steps to reach final difficulty
        }
        self.global_step_count = 0

        # Environment configuration
        self.control_freq = control_freq
        self.step_period = 1.0 / control_freq
        self.max_steps = max_episode_steps
        self.step_count = 0
        self.use_sim_time = use_sim_time

        # The reward function is designed to prevent 2 main problems:
        # 1. "Lazy Policy" (Positive Living Reward Loophole): If the base reward for survival is too
        #    high relative to the task reward, the agent may choose to simply hover safely to
        #    accumulate the living reward (e.g., +0.1 * 2000 steps = +200) rather than risking a
        #    crash to pursue the objective.
        # 2. "Suicide Policy" (Negative Living Reward Loophole): If the penalty for existence (or
        #    error) is too strict without a sufficient positive signal, the agent may intentionally
        #    crash immediately to minimize the accumulation of negative rewards (e.g., -0.1 * 2000
        #    steps = -200)
        # Strategy: In the initial state, the net reward should be a tiny negative value, e.g.
        # -0.001, so that if the agent does not take any action, the total reward will be roughly
        # -0.001 * 2000 steps = -2, which is a small negative value compared to the crash penalty.
        # The step reward range is [-10.0, +10.0].
        # Positive value as reward, negative value as penalty.
        self.reward_config = {
            "base_reward": 0.2,
            "pos_dist_penalty": -0.0388,
            "vel_dist_penalty": -0.007,
            "progress_reward": 1.0,
            "danger_zone_tilt": math.pi / 6,  # 6, 7, 8, 9
            "tilt_penalty": -0.5,
            "success_reward": 5.0,
            "crash_penalty": -10.0,
        }

        # Safety configuration
        self.safety_config = {
            "max_tilt_angle": math.pi / 4,  # 45 degrees
            "max_altitude": 10.0,
            "min_altitude": 0.5,
            "success_pos_tol": 0.38,
            "success_tilt_tol": math.radians(15),
        }

        # Initialize state variables
        self.gt_odom = None
        self.cmd_odom = None
        self.navi_state = None
        self._last_pos_dist = None

        # Threading locks for thread-safe data access
        self.odom_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # Setup ROS2 communication
        self._setup_ros_communication()

        # QuadcopterParams defined in `sdrl_geometric_controller/quadcopter_params.py`.
        self.drone_params = QuadcopterParams()
        self.wrench_limits = {
            "force_z": self.drone_params.force_z_limit,
            "torque_x": self.drone_params.torque_x_limit,
            "torque_y": self.drone_params.torque_y_limit,
            "torque_z": self.drone_params.torque_z_limit,
        }
        self.max_motor_speed = self.drone_params.motor_max_rot_velocity

        # Action space: fz, tx, ty, tz
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: normalized roll and pitch ranges in [-1, 1]
        # x_err_n, y_err_n, z_err_n (Body), vx_err_n, vy_err_n, vz_err_n (Body),
        # roll_n, pitch_n, p_n, q_n, r_n
        self.observation_space = gym.spaces.Box(
            low=-np.ones(11, dtype=np.float32),
            high=np.ones(11, dtype=np.float32),
            dtype=np.float32,
        )

        # Performance tracking
        self.episode_stats = {
            "total_reward": 0.0,
            "crash": False,
        }

        self._crash_count = 0

    def _setup_ros_communication(self):
        """Setup ROS2 publishers and subscribers."""
        # QoS profiles
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.node.create_subscription(Odometry, "/X3/gt_odom", self._gt_odom_cb, qos_best_effort)
        self.node.create_subscription(Odometry, "/X3/cmd_odom", self._cmd_odom_cb, qos_reliable)
        self.node.create_subscription(String, "/X3/navi_state", self._navi_state_cb, qos_reliable)

        # Publisher
        self.motor_pub = self.node.create_publisher(
            Float32MultiArray, "/X3/ros/motor_speed", qos_reliable
        )

        # Reset service client
        self.reset_cli = self.node.create_client(Trigger, "/X3/reset_drone_initial_pose")

        # Start ROS2 spinning in separate thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

    def _gt_odom_cb(self, msg: Odometry):
        with self.odom_lock:
            self.gt_odom = msg

    def _cmd_odom_cb(self, msg: Odometry):
        with self.odom_lock:
            self.cmd_odom = msg

    def _navi_state_cb(self, msg: String):
        with self.state_lock:
            self.navi_state = msg.data

    def _wait_for_odom(self, min_time_ns: int, timeout_sec: float = 2.0) -> None:
        """Block until odometry timestamp >= min_time_ns."""
        start_wait = time.monotonic()
        while True:
            with self.odom_lock:
                if self.gt_odom is not None:
                    # Convert msg time to nanoseconds manually or via rclpy
                    msg_time = self.gt_odom.header.stamp
                    # seconds * 1e9 + nanoseconds
                    odom_ns = msg_time.sec * 1_000_000_000 + msg_time.nanosec
                    if odom_ns >= min_time_ns:
                        return

            if time.monotonic() - start_wait > timeout_sec:
                self.node.get_logger().warn(
                    f"Timeout waiting for odom. Min: {min_time_ns}, Last: {odom_ns if 'odom_ns' in locals() else 'None'}"
                )
                return

            time.sleep(0.0005)  # 0.5ms wall-clock sleep to yield

    def _compute_error(self) -> np.ndarray:
        """Compute observation error vector without normalization."""
        with self.odom_lock:
            # Snapshot the references to ensure consistency
            gt_odom = self.gt_odom
            cmd_odom = self.cmd_odom

        assert gt_odom is not None and cmd_odom is not None
        gt_pos = gt_odom.pose.pose.position
        cmd_pos = cmd_odom.pose.pose.position
        gt_wxyz = gt_odom.pose.pose.orientation

        # Position error (World Frame)
        dx_w = cmd_pos.x - gt_pos.x
        dy_w = cmd_pos.y - gt_pos.y
        dz_w = cmd_pos.z - gt_pos.z

        # Velocity error (World Frame)
        gt_v = gt_odom.twist.twist.linear
        cmd_v = cmd_odom.twist.twist.linear
        dvx_w = cmd_v.x - gt_v.x
        dvy_w = cmd_v.y - gt_v.y
        dvz_w = cmd_v.z - gt_v.z

        # Drone's rotating motion wrt world frame. R is Rotation of Body w.r.t World
        rotmat = quat_to_rotmat(gt_wxyz.w, gt_wxyz.x, gt_wxyz.y, gt_wxyz.z)

        # Rotate errors from world frame to body frame.
        # err_body = R.T @ err_world.
        pos_err_w = np.array([dx_w, dy_w, dz_w])
        vel_err_w = np.array([dvx_w, dvy_w, dvz_w])
        pos_err_b = rotmat.T @ pos_err_w
        vel_err_b = rotmat.T @ vel_err_w

        dx, dy, dz = pos_err_b
        dvx, dvy, dvz = vel_err_b

        # Since the desired angular velocity is always (0, 0, 0) (check `navigator.py`),
        # the angular velocity error is effectively (0 - p, 0 - q, 0 - r).
        # We use the raw angular velocity in Body Frame as it contains the same information.
        ang_vel = gt_odom.twist.twist.angular

        # The desired roll and pitch are 0 (level flight), so these act as errors.
        # Yaw is calculated but not used in the state vector.
        roll, pitch, yaw = quat_to_euler(
            gt_wxyz.w,
            gt_wxyz.x,
            gt_wxyz.y,
            gt_wxyz.z,
        )

        return dx, dy, dz, dvx, dvy, dvz, roll, pitch, yaw, ang_vel.x, ang_vel.y, ang_vel.z

    def _compute_normalized_state(
        self, dx, dy, dz, dvx, dvy, dvz, roll, pitch, p, q, r
    ) -> np.ndarray:
        """Compute normalized state vector."""

        # Normalize position error
        x_err_n = np.clip(dx / MAX_POS_ERR, -1.0, 1.0)
        y_err_n = np.clip(dy / MAX_POS_ERR, -1.0, 1.0)
        z_err_n = np.clip(dz / MAX_POS_ERR, -1.0, 1.0)

        # Normalize velocity error
        vx_err_n = np.clip(dvx / MAX_VEL_ERR, -1.0, 1.0)
        vy_err_n = np.clip(dvy / MAX_VEL_ERR, -1.0, 1.0)
        vz_err_n = np.clip(dvz / MAX_VEL_ERR, -1.0, 1.0)

        # NOTE: it may have the singularity problem
        # Roll is [-pi, pi] -> [-1, 1]
        roll_n = np.clip(roll / math.pi, -1.0, 1.0)
        # Pitch is [-pi/2, pi/2] -> [-1, 1]
        pitch_n = np.clip(pitch / (math.pi / 2.0), -1.0, 1.0)

        # Angular velocity
        p_n = np.clip(p / MAX_ANG_VEL, -1.0, 1.0)
        q_n = np.clip(q / MAX_ANG_VEL, -1.0, 1.0)
        r_n = np.clip(r / MAX_ANG_VEL, -1.0, 1.0)

        state = np.array(
            [
                x_err_n,
                y_err_n,
                z_err_n,
                vx_err_n,
                vy_err_n,
                vz_err_n,
                roll_n,
                pitch_n,
                p_n,
                q_n,
                r_n,
            ],
            dtype=np.float32,
        )

        return state

    def compute_reward(self, pos_dist, vel_dist, tilt, success, crash) -> float:
        """Compute reward."""

        reward = self.reward_config["base_reward"]

        reward += self.reward_config["pos_dist_penalty"] * pos_dist

        reward += self.reward_config["vel_dist_penalty"] * vel_dist

        if self._last_pos_dist is not None:
            progress = self._last_pos_dist - pos_dist
            reward += self.reward_config["progress_reward"] * progress

        tilt_excess = abs(tilt) - self.reward_config["danger_zone_tilt"]
        if tilt_excess > 0:
            reward += self.reward_config["tilt_penalty"] * tilt_excess

        if success:
            reward += self.reward_config["success_reward"]

        if crash:
            reward += self.reward_config["crash_penalty"]

        reward = np.clip(reward, -10.0, 10.0)

        return reward

    def _action_to_motors(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized actions to motor speeds via wrench.
        In this implementation, we do not use expert control signal for residual learning. If we use
        expert control signal, the initial reward/return will be very high, and no matter what
        the agent does, it will never be higher than the initial reward/return which results in
        the agent never learning to explore.

        The problem with residual learning is that it will give high reward and return when using expert
        control signal in the beginning of the training. By doing this, SAC will never learn to explore
        because no matter what it does, it will only get the worse reward and return. It will think the
        initial state (random state) is the better state. By doing this, SAC will never learn to explore.
        """

        force_rl, torque_rl = self.action_to_wrench(action)

        force_total = force_rl
        torque_total = torque_rl

        motor_speeds = wrench_to_motor_speeds(
            force_total,
            torque_total,
            self.drone_params.rotor_positions,
            self.drone_params.rotor_cf,
            self.drone_params.rotor_cd,
            self.drone_params.yaw_signs,
            self.drone_params.motor_max_rot_velocity,
        )
        return motor_speeds

    def action_to_wrench(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Map normalized action [-1,1]^4 to wrench based on the wrench limits."""
        assert action.shape == (4,), f"Invalid action shape: {action.shape}"
        assert np.all((action >= -1.0) & (action <= 1.0)), f"Invalid action values: {action}"

        fz_min, fz_max = self.wrench_limits["force_z"]
        tx_min, tx_max = self.wrench_limits["torque_x"]
        ty_min, ty_max = self.wrench_limits["torque_y"]
        tz_min, tz_max = self.wrench_limits["torque_z"]

        F_hover = self.drone_params.hover_thrust

        # Calculate scaling factors for the asymmetric map:
        S_pos = fz_max - F_hover  # Max upward reserve (e.g., 21.88 - 15.115 = 6.765 N)
        S_neg = F_hover - fz_min  # Max downward reserve (e.g., 15.115 - 0.0 = 15.115 N)

        # Calculate Force Z (Fz) using piecewise mapping:
        if action[0] >= 0:
            # Positive Deviation (a_z in  maps to Fz in [F_hover, F_max])
            # Fz = F_hover + (Upward Reserve) * a_z
            force_z = F_hover + S_pos * action[0]
        else:
            # Negative Deviation (a_z in [-1, 0] maps to Fz in [F_min, F_hover])
            # Fz = F_hover + (Downward Reserve) * a_z
            # Since S_neg is positive and a_z is negative, this correctly subtracts force.
            force_z = F_hover + S_neg * action[0]

        # Apply final numerical clipping as a safety measure
        force_z = np.clip(force_z, fz_min, fz_max)

        # Torques are typically naturally centered at 0, so symmetric mapping is fine.
        def map_to_range_symmetric(a: float, v_min: float, v_max: float) -> float:
            """Map a in [-1, 1] to [v_min, v_max] linearly, centered at (v_min+v_max)/2."""
            a_clipped = np.clip(a, -1.0, 1.0)
            center = 0.5 * (v_max + v_min)
            half_range = 0.5 * (v_max - v_min)
            return center + half_range * a_clipped

        # Note: We use action[1], action[2], action[3] for torques
        torque_x = map_to_range_symmetric(action[1], tx_min, tx_max)
        torque_y = map_to_range_symmetric(action[2], ty_min, ty_max)
        torque_z = map_to_range_symmetric(action[3], tz_min, tz_max)

        # Compile results
        torque = np.array([torque_x, torque_y, torque_z], dtype=np.float32)

        return force_z, torque

    def get_success_radius(self) -> float:
        """Calculate success radius based on curriculum progress."""
        if not self.curriculum_config["enabled"]:
            return self.safety_config["success_pos_tol"]

        if self.global_step_count < self.curriculum_config["initial_difficulty_steps"]:
            progress_ratio = 0.0
        else:
            # Calculate progress relative to the difficulty ramp-up phase
            steps_since_start = (
                self.global_step_count - self.curriculum_config["initial_difficulty_steps"]
            )
            ramp_duration = (
                self.curriculum_config["final_difficulty_steps"]
                - self.curriculum_config["initial_difficulty_steps"]
            )
            progress_ratio = min(1.0, steps_since_start / ramp_duration)

        assert 0.0 <= progress_ratio <= 1.0, (
            f"progress_ratio {progress_ratio} is out of bounds [0, 1]"
        )

        current_success_radius = (
            self.curriculum_config["initial_success_radius"] * (1 - progress_ratio)
            + self.curriculum_config["final_success_radius"] * progress_ratio
        )
        return current_success_radius

    def step(self, action):
        """Execute one environment step.

        During training, when the action is sampled, is it sampled from the SAC policy?
        If it is sampled from the SAC policy, does it depend on the current state?
        If it does not depend on the current state, in my opinion, the training can never succeed,
        because the agent can never control the drone to the desired state.
        """
        # Validate state - this should never happen if reset() succeeded
        with self.odom_lock:
            valid_state = self.gt_odom is not None and self.cmd_odom is not None

        assert valid_state, (
            "gt_odom or cmd_odom is None in step(); reset() must ensure valid state before stepping"
        )

        # Capture current time before stepping
        start_time_ns = self.node.get_clock().now().nanoseconds
        # We expect time to advance by step_period
        target_time_ns = start_time_ns + int(self.step_period * 1e9)

        # Apply action
        motor_speeds = self._action_to_motors(action).tolist()
        msg = Float32MultiArray()
        msg.data = motor_speeds
        self.motor_pub.publish(msg)

        # Step the simulation
        # The physics runs at 1000 Hz. Control runs at 100 Hz.
        # We need to step 1000 / 100 = 10 physics steps.
        steps_to_take = 10
        world_control(WORLD_NAME, step_multi=steps_to_take)  # non-blocking call

        # Wait for physics update using explicit odom check instead of rate.sleep()
        # This ensures we don't compute reward on stale data.
        self._wait_for_odom(target_time_ns)

        # Compute error without normalization
        dx, dy, dz, dvx, dvy, dvz, roll, pitch, _, p, q, r = self._compute_error()
        pos_dist = min(math.sqrt(dx * dx + dy * dy + dz * dz), MAX_POS_DIST)
        vel_dist = min(math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz), MAX_VEL_DIST)
        tilt = roll_pitch_to_tilt(roll, pitch)

        # Apply curriculum learning for success criteria
        current_success_radius = self.get_success_radius()
        success = (
            pos_dist < current_success_radius and tilt < self.safety_config["success_tilt_tol"]
        )

        with self.state_lock:
            current_navi_state = self.navi_state

        if current_navi_state == "CRASHED":
            self._crash_count += 1
        else:
            self._crash_count = 0

        crash = (
            current_navi_state == "CRASHED" and self._crash_count >= 4
            # The crash state will be updated by the navigator.
            # or tilt > self.safety_config["max_tilt_angle"]
            # the ground truth height below min_altitude does not mean crash. it could be that it's
            # still in the process of taking off.
            # or self.gt_odom.pose.pose.position.z < self.safety_config["min_altitude"]
        )

        # Compute reward
        reward = self.compute_reward(
            pos_dist,
            vel_dist,
            tilt,
            success,
            current_navi_state == "CRASHED",
        )

        # Enhanced logging for debugging reward function
        if self.step_count == 0:
            self.node.get_logger().info(
                f"Step 0: Reward={reward:.4f}, Dist={pos_dist:.2f}m, "
                f"VelErr={vel_dist:.2f}m/s, Tilt={math.degrees(tilt):.1f}°"
            )
        if crash:
            self.node.get_logger().info(f"Crash! Reward: {reward:.4f}")
        if self.step_count > 0 and self.step_count % 200 == 0:
            self.node.get_logger().info(
                f"Step {self.step_count}: Reward={reward:.4f}, Dist={pos_dist:.2f}m"
            )

        # Update episode stats
        self.episode_stats["total_reward"] += reward

        # Update housekeeping variables
        self.step_count += 1
        self.global_step_count += 1
        self._last_pos_dist = pos_dist

        terminated = crash
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            self.node.get_logger().info(
                f"Episode end: steps={self.step_count}, "
                f"crash={crash}, truncated={truncated}, "
                f"total_reward={self.episode_stats['total_reward']:.2f}, "
            )
            self.episode_stats["crash"] = crash

        # compute state with normalization
        state = self._compute_normalized_state(dx, dy, dz, dvx, dvy, dvz, roll, pitch, p, q, r)

        return state, reward, terminated, truncated, {"success": success, "crash": crash}

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Request drone reset
        self._request_reset(timeout_sec=30.0)

        # Reset internal state
        self.step_count = 0
        self._last_pos_dist = None
        self._crash_count = 0

        with self.odom_lock:
            self.gt_odom = None
            self.cmd_odom = None

        with self.state_lock:
            self.navi_state = None

        # Reset episode stats
        self.episode_stats = {
            "total_reward": 0.0,
            "success": False,
            "crash": False,
        }

        # Wait for fresh data using a temporary higher-frequency rate
        wait_rate = self.node.create_rate(10)  # 10 Hz for waiting
        timeout_start = self.node.get_clock().now()
        timeout_duration = 5.0  # seconds

        while rclpy.ok():
            with self.odom_lock:
                got_data = self.gt_odom is not None and self.cmd_odom is not None

            if got_data:
                break

            elapsed = (self.node.get_clock().now() - timeout_start).nanoseconds / 1e9
            if elapsed > timeout_duration:
                self.node.get_logger().error("Timeout waiting for odometry data after reset.")
                raise RuntimeError("Failed to receive odometry data after reset")
            wait_rate.sleep()

        # Pause physics
        world_control(WORLD_NAME, pause=True)

        dx, dy, dz, dvx, dvy, dvz, roll, pitch, _, p, q, r = self._compute_error()
        state = self._compute_normalized_state(dx, dy, dz, dvx, dvy, dvz, roll, pitch, p, q, r)

        return state, {}

    def _request_reset(self, timeout_sec: float) -> bool:
        """Request navigator to reset drone position."""
        try:
            if not self.reset_cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("Reset service not available")

            req = Trigger.Request()
            future = self.reset_cli.call_async(req)

            # Wait for response using ROS time
            start_time = self.node.get_clock().now()
            wait_rate = self.node.create_rate(10)  # 10 Hz for waiting
            while rclpy.ok() and not future.done():
                elapsed = (self.node.get_clock().now() - start_time).nanoseconds / 1e9
                if elapsed > timeout_sec:
                    raise RuntimeError("Reset timeout")
                wait_rate.sleep()

            resp = future.result()
            if resp is None or not resp.success:
                raise RuntimeError("Reset failed")
            return True

        except Exception as e:
            self.node.get_logger().error(f"Reset error: {e}")
            raise e

    def close(self):
        """Clean up ROS2 resources."""
        self.spin_thread.join()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def create_training_parser():
    """Create argument parser for training configuration."""
    parser = argparse.ArgumentParser(description="Train RL agent for quadcopter control")

    # Algorithm selection (only SAC is supported)
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac"],
        help="RL algorithm to use (only 'sac' is supported)",
    )

    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint save frequency",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help=(
            "Number of parallel environments. "
            "Must be 1 for the ROS2/Gazebo-based simulation (single drone instance)."
        ),
    )

    # Environment parameters
    parser.add_argument(
        "--control-freq",
        type=int,
        default=100,  # slightly faster than 100 Hz to make the wait time in step() shorter.
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode",
    )

    # Paths
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="~/StalkerDroneRL/checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./tb_logs",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, auto)",
    )

    return parser


def setup_algorithm(algo_name: str, env, args):
    """Setup RL algorithm with appropriate hyperparameters."""

    common_kwargs = {
        "env": env,
        "verbose": 1,
        "tensorboard_log": args.log_dir,
        "device": args.device,
    }

    if algo_name != "sac":
        raise ValueError(f"Unknown or unsupported algorithm: {algo_name}. Only 'sac' is supported.")

    model = SAC(
        policy="MlpPolicy",
        **common_kwargs,
        learning_rate=3e-4,
        buffer_size=200_000,  # 200k, 600k, 1,000k
        learning_starts=20_000,  # 20k, 60k, 100k
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={
            "net_arch": [256, 256],
            "log_std_init": -2.0,  # -3.0, -2.0, -1.0
        },
        use_sde=False,  # Could try True for more structured exploration
        ent_coef="auto",  # Automatic entropy tuning
        target_entropy="auto",  # Let SAC determine optimal entropy
    )

    return model


def main(argv=None):
    """Main training function.

    Note:
        When launched via ROS 2 (e.g. from a launch file), additional ROS-specific CLI
        arguments like ``--ros-args`` and ``-r __node:=...`` are passed in. We strip
        those out using :func:`rclpy.utilities.remove_ros_args` before giving the
        remaining arguments to :mod:`argparse`, so this script can be used both with
        ``ros2 launch`` / ``ros2 run`` and as a plain Python CLI.
    """
    parser = create_training_parser()

    # Strip ROS 2 arguments (e.g. --ros-args -r __node:=sac_trainer) so argparse
    # does not complain about unrecognized options when launched as a ROS node.
    if argv is None:
        cleaned_argv = remove_ros_args(sys.argv)
        argv = cleaned_argv[1:]  # drop program name

    args = parser.parse_args(argv)

    # NOTE: The Gazebo/ROS2 simulation exposes a single physical drone instance.
    # Running multiple vectorized environments would cause all envs to share the
    # same topics and motor commands, which is fundamentally incorrect. We
    # therefore *forbid* n_envs != 1 here.
    if args.n_envs != 1:
        raise ValueError(
            "Parallel environments (--n-envs != 1) are not supported with the "
            "current ROS2/Gazebo setup. Please set --n-envs=1."
        )

    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        set_random_seed(args.seed)

    # Create environment factory
    def make_env():
        env = QuadcopterTrackingEnv(
            control_freq=args.control_freq,
            max_episode_steps=args.max_episode_steps,
            use_sim_time=True,  # Always use sim time for training
        )
        env = Monitor(env)
        return env

    # Create vectorized environment (single-env DummyVecEnv; multi-env is forbidden above)
    env = DummyVecEnv([lambda: make_env()])

    # Setup algorithm
    if args.resume:
        print(f"Loading model from: {args.resume}")
        if args.algo != "sac":
            raise ValueError(
                f"Unknown or unsupported algorithm: {args.algo}. Only 'sac' is supported."
            )
        model = SAC.load(args.resume, env=env, device=args.device)
    else:
        model = setup_algorithm(args.algo, env, args)
        # Set the model's environment explicitly (already passed into the constructor,
        # but this keeps the pattern explicit and works if we later swap envs).
        model.set_env(env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algo}_quadcopter_sac",
        save_replay_buffer=True,
    )

    # Single callback: pure SAC training with checkpointing
    callbacks = [checkpoint_callback]

    # Configure logger
    logger = configure(
        str(log_dir / f"{args.algo}_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        ["stdout", "tensorboard", "csv"],
    )
    model.set_logger(logger)

    # Train
    try:
        print(f"Starting SAC training with {args.algo}")
        print(f"Total timesteps: {args.total_timesteps}")

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False if args.resume else True,
        )

        # Save final model
        final_path = checkpoint_dir / f"{args.algo}_quadcopter_final"
        model.save(str(final_path))
        print(f"Saved final model to: {final_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
