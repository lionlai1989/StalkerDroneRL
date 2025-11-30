"""
Observation, State, and Action transformations for RL-based drone control.

Terminology:
- Observation: Raw sensor measurements (e.g., odometry from /X3/gt_odom). Or the error between the
  observed pose of the drone and the desired pose.
- State: The true/estimated state of the drone (position, velocity, orientation, angular velocity)
- Action: The RL agent's output, mapped to physical commands (wrench, motor speeds)

In this project, we use ground truth odometry, so observation ≈ state (no estimation needed).

"""

import math
import numpy as np
from sdrl_geometric_controller.transform import quat_to_rotmat, quat_to_euler, roll_pitch_to_tilt
from sdrl_geometric_controller.motor_mixing import wrench_to_motor_speeds

# Constants
MAX_POS_ERR = 5.0
MAX_VEL_ERR = 3.0
MAX_ANG_VEL = 6.28  # 2π rad/s ~= 360 deg/s


def compute_tracking_error(gt_pose, gt_twist, cmd_pose, cmd_twist):
    """
    Compute tracking error between ground truth and commanded pose/twist without normalization.
    Args:
        gt_pose: geometry_msgs.msg.Pose
        gt_twist: geometry_msgs.msg.Twist
        cmd_pose: geometry_msgs.msg.Pose
        cmd_twist: geometry_msgs.msg.Twist
    Returns:
        tuple: (dx, dy, dz, dvx, dvy, dvz, roll, pitch, yaw, p, q, r)
    """
    gt_pos = gt_pose.position
    cmd_pos = cmd_pose.position

    # Position error (World Frame)
    dx_w = cmd_pos.x - gt_pos.x
    dy_w = cmd_pos.y - gt_pos.y
    dz_w = cmd_pos.z - gt_pos.z

    # Velocity error (World Frame)
    gt_v = gt_twist.linear
    cmd_v = cmd_twist.linear
    dvx_w = cmd_v.x - gt_v.x
    dvy_w = cmd_v.y - gt_v.y
    dvz_w = cmd_v.z - gt_v.z

    # Drone's rotating motion wrt world frame. R is Rotation of Body w.r.t World
    rotmat = quat_to_rotmat(
        gt_pose.orientation.w,
        gt_pose.orientation.x,
        gt_pose.orientation.y,
        gt_pose.orientation.z,
    )

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
    ang_vel = gt_twist.angular
    p, q, r = ang_vel.x, ang_vel.y, ang_vel.z

    # The desired roll and pitch are 0 (level flight), so these act as errors.
    # Yaw is calculated but not used in the state vector.
    roll, pitch, yaw = quat_to_euler(
        gt_pose.orientation.w,
        gt_pose.orientation.x,
        gt_pose.orientation.y,
        gt_pose.orientation.z,
    )

    return dx, dy, dz, dvx, dvy, dvz, roll, pitch, yaw, p, q, r


def compute_normalized_observation(dx, dy, dz, dvx, dvy, dvz, roll, pitch, p, q, r):
    """
    Compute normalized observation vector.
    Returns:
        np.ndarray: shape (11,)
    """
    x_err_n = np.clip(dx / MAX_POS_ERR, -1.0, 1.0)
    y_err_n = np.clip(dy / MAX_POS_ERR, -1.0, 1.0)
    z_err_n = np.clip(dz / MAX_POS_ERR, -1.0, 1.0)

    vx_err_n = np.clip(dvx / MAX_VEL_ERR, -1.0, 1.0)
    vy_err_n = np.clip(dvy / MAX_VEL_ERR, -1.0, 1.0)
    vz_err_n = np.clip(dvz / MAX_VEL_ERR, -1.0, 1.0)

    # NOTE: it may have singularity problem
    # Roll is [-pi, pi] -> [-1, 1]
    roll_n = np.clip(roll / math.pi, -1.0, 1.0)
    # Pitch is [-pi/2, pi/2] -> [-1, 1]
    pitch_n = np.clip(pitch / (math.pi / 2.0), -1.0, 1.0)

    p_n = np.clip(p / MAX_ANG_VEL, -1.0, 1.0)
    q_n = np.clip(q / MAX_ANG_VEL, -1.0, 1.0)
    r_n = np.clip(r / MAX_ANG_VEL, -1.0, 1.0)

    obs = np.array(
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
    return obs


def action_to_wrench(action, wrench_limits, hover_thrust):
    """
    Map normalized action [-1,1]^4 to wrench based on the wrench limits.
    """
    assert action.shape == (4,), f"Invalid action shape: {action.shape}"
    assert np.all((action >= -1.0) & (action <= 1.0)), f"Invalid action values: {action}"

    fz_min, fz_max = wrench_limits["force_z"]
    tx_min, tx_max = wrench_limits["torque_x"]
    ty_min, ty_max = wrench_limits["torque_y"]
    tz_min, tz_max = wrench_limits["torque_z"]

    F_hover = hover_thrust

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
