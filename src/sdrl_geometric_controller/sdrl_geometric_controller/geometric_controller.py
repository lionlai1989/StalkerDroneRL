"""
Geometric Controller

T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a quadrotor
UAV on SE(3)," Proceedings of the 49th IEEE Conference on Decision and Control (CDC), 2010.
"""

import math

import numpy as np
from geometry_msgs.msg import Pose, Twist

GRAVITY = 9.81


def rotation_error(rot_current, rot_desired):
    rot_err = rot_desired.T @ rot_current - rot_current.T @ rot_desired
    return np.array(
        [
            0.5 * (rot_err[2, 1] - rot_err[1, 2]),
            0.5 * (rot_err[0, 2] - rot_err[2, 0]),
            0.5 * (rot_err[1, 0] - rot_err[0, 1]),
        ]
    )


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-9:
        return np.eye(3)
    s = 2.0 / n
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )
    return R


def quaternion_to_euler_yaw(q):
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class GeometricController:
    def __init__(self):
        # Proportional gain on position error for position control
        self.kp_position = 3.0
        # Derivative gain on linear velocity error for linear velocity control
        self.kv_linvel = 5.0
        # Proportional gain on rotation matrix error for rotation matrix control
        self.kr_rotmat = 6.0
        # Derivative gain on angular velocity error for angular velocity control
        self.kw_angvel = 3.0

        # NOTE: Physical parameters defined here MUST match the values defined in
        # "src/sdrl_lionquadcopter/models/lion_quadcopter.sdf"
        # "src/sdrl_lionquadcopter/models/x3_uav/model.sdf"
        # Mass of the drone (kg)
        self.mass = 1.5
        # Diagonal inertia [Ixx, Iyy, Izz]
        self.inertia = np.array([0.0347563, 0.07, 0.0977], dtype=float)
        # Thrust coefficient cf or motorConstant (N / (rad/s)^2)
        self.rotor_cf = 8.54858e-06
        # Drag coefficient cd or momentConstant (N*m / (rad/s)^2)
        self.rotor_cd = 0.016
        # Maximum rotor angular velocity (rad/s)
        self.motor_max_rot_velocity = 800.0
        # Rotor positions (x, y, z) in body frame. All rotations are identity matrix.
        self.rotor_positions = np.array(
            [
                [0.13, -0.22, 0.023],
                [-0.13, 0.2, 0.023],
                [0.13, 0.22, 0.023],
                [-0.13, -0.2, 0.023],
            ]
        )
        # Rotor yaw torque directions (+1 for ccw, -1 for cw)
        self.yaw_signs = np.array([+1, +1, -1, -1])

        # max tilt angle for the drone (rad)
        self.max_tilt_angle = math.pi / 12

        # max acceleration for the drone (m/s^2)
        self.max_accel = self.compute_max_accel()

        (
            self.force_z_limit,
            self.torque_x_limit,
            self.torque_y_limit,
            self.torque_z_limit,
        ) = self.calculate_wrench_limits()

        assert self.max_accel > 0.0, "Max acceleration must be positive"
        assert self.rotor_cf > 0.0, "rotor_cf must be positive"
        assert self.rotor_cd > 0.0, "rotor_cd must be positive"
        assert self.motor_max_rot_velocity > 0.0, "motor_max_rot_velocity must be positive"

    def compute_max_accel(self):
        """Compute thrust-limited max acceleration magnitude (m/s^2).

        F_max = 4 * rotor_cf * motor_max_rot_velocity^2
        """
        return 4.0 * self.rotor_cf * (self.motor_max_rot_velocity**2) / self.mass

    def calculate_wrench_limits(self):
        """Calculate force and torque limits based on motor capabilities.

        Returns:
            Tuple of (force_z_limit, torque_x_limit, torque_y_limit, torque_z_limit)
            where each limit is a (min, max) tuple
        """
        # Maximum thrust force (all motors at max speed)
        max_thrust_per_motor = self.rotor_cf * (self.motor_max_rot_velocity**2)
        max_total_thrust = 4.0 * max_thrust_per_motor

        force_z_limit = (0.0, max_total_thrust)

        self.hover_thrust = self.mass * GRAVITY + 0.5  # 1.5 * 9.81 = 14.715 N, +0.5 or +1.0?

        # TODO: To prevent the drone from crashing, limit rpy torque. Loosen them later.
        torque_x_limit = (-0.1, 0.1)
        torque_y_limit = (-0.1, 0.1)
        torque_z_limit = (-0.05, 0.05)
        return force_z_limit, torque_x_limit, torque_y_limit, torque_z_limit

    def compute_motor_speeds(
        self, curr_pose: Pose, curr_twist: Twist, desired_pose: Pose, desired_twist: Twist
    ) -> np.ndarray:
        """Compute 4 motor speeds (rad/s) from current and desired states.

        Inputs:
        - curr_pose: geometry_msgs/Pose (current world pose)
        - curr_twist: geometry_msgs/Twist (current world linear vel, body angular vel)
        - desired_pose: geometry_msgs/Pose (desired world pose)
        - desired_twist: geometry_msgs/Twist (desired world linear vel, body angular vel)

        Pose:
        - position: body frame movement in world frame
        - orientation: body frame rotation in world frame

        Twist:
        - linear: linear velocities in world frame
        - angular: angular velocities in body frame.
        """
        force, torque = self.compute_wrench(curr_pose, curr_twist, desired_pose, desired_twist)
        return self.wrench_to_motor_speeds(force, torque)

    def compute_wrench(
        self, curr_pose: Pose, curr_twist: Twist, desired_pose: Pose, desired_twist: Twist
    ) -> tuple[float, np.ndarray]:
        curr_pos = np.array(
            [curr_pose.position.x, curr_pose.position.y, curr_pose.position.z], dtype=float
        )
        curr_wxyz = np.array(
            [
                curr_pose.orientation.w,
                curr_pose.orientation.x,
                curr_pose.orientation.y,
                curr_pose.orientation.z,
            ],
            dtype=float,
        )
        curr_linvel = np.array(
            [curr_twist.linear.x, curr_twist.linear.y, curr_twist.linear.z], dtype=float
        )
        curr_angvel = np.array(
            [curr_twist.angular.x, curr_twist.angular.y, curr_twist.angular.z], dtype=float
        )

        des_pos = np.array(
            [desired_pose.position.x, desired_pose.position.y, desired_pose.position.z], dtype=float
        )
        des_wxyz = np.array(
            [
                desired_pose.orientation.w,
                desired_pose.orientation.x,
                desired_pose.orientation.y,
                desired_pose.orientation.z,
            ],
            dtype=float,
        )
        des_yaw = float(quaternion_to_euler_yaw(des_wxyz))
        des_lin_vel = np.array(
            [desired_twist.linear.x, desired_twist.linear.y, desired_twist.linear.z], dtype=float
        )
        des_angvel = np.array(
            [desired_twist.angular.x, desired_twist.angular.y, desired_twist.angular.z], dtype=float
        )

        e_pos = curr_pos - des_pos
        e_linvel = curr_linvel - des_lin_vel

        # Compute force
        curr_rot = quaternion_to_rotation_matrix(curr_wxyz)
        acc_ctrl = (
            -self.kp_position * e_pos
            - self.kv_linvel * e_linvel
            + GRAVITY * np.array([0.0, 0.0, 1.0])
        )  # Control acceleration from PD control and gravity. Discard the desired acc.
        body_z = curr_rot[:, 2]
        force = self.mass * float(np.dot(acc_ctrl, body_z))
        if force < 0.0:  # NOTE: what would happen if force is negative?
            force = 0.0

        # Compute torque
        acc_mag = np.linalg.norm(acc_ctrl)
        if acc_mag > self.max_accel:
            acc_ctrl = acc_ctrl * (self.max_accel / acc_mag)
        desired_rot = self.compute_desired_orientation(acc_ctrl, des_yaw)
        e_rot = rotation_error(curr_rot, desired_rot)
        e_angvel = curr_angvel - curr_rot.T.dot(desired_rot.dot(des_angvel))
        torque = (
            -self.kr_rotmat * e_rot
            - self.kw_angvel * e_angvel
            + np.cross(curr_angvel, self.inertia * curr_angvel)
        )
        return force, torque

    def compute_desired_orientation(self, acc, yaw):
        a = acc.copy()
        if a[2] < 1e-6:
            a[2] = 1e-6
        horiz = math.hypot(a[0], a[1])
        max_horiz = math.tan(self.max_tilt_angle) * abs(a[2])
        if horiz > max_horiz:
            scale = max_horiz / (horiz + 1e-9)
            a[0] *= scale
            a[1] *= scale
        norm_a = float(np.linalg.norm(a))
        if norm_a > 1e-6:
            z_w_des = a / norm_a
        else:
            z_w_des = np.array([0.0, 0.0, 1.0])
        # Desired heading direction in world xy-plane
        x_c_des = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        # Check for near-singularity
        if abs(float(np.dot(z_w_des, x_c_des))) > 0.999:
            x_c_des = np.array([math.cos(yaw + 0.01), math.sin(yaw + 0.01), 0.0])
        # Compute orthonormal basis
        y_w_des = np.cross(z_w_des, x_c_des)
        y_w_des /= float(np.linalg.norm(y_w_des))
        x_w_des = np.cross(y_w_des, z_w_des)
        x_w_des /= float(np.linalg.norm(x_w_des))
        # Rotation matrix columns are body axes in world frame
        R_d = np.column_stack((x_w_des, y_w_des, z_w_des))
        return R_d

    def wrench_to_motor_speeds(self, force: float, torque: np.ndarray) -> np.ndarray:
        """
        Map a desired wrench (force + torque) to rotor speeds using the
        sign-based lever-arm mixing approach.
        """

        # Initialize thrusts equally (baseline total force)
        thrusts = np.full(4, 0.25 * force, dtype=float)

        # Roll and pitch distribution using sign-based lever arms
        x = self.rotor_positions[:, 0]
        y = self.rotor_positions[:, 1]
        sx = np.where(x >= 0.0, 1.0, -1.0)
        sy = np.where(y >= 0.0, 1.0, -1.0)
        # lever arm length from geometry
        arm_len = np.mean(np.hypot(x, y))
        assert arm_len > 0.0, "Lever arm length must be positive"

        # Roll: delta_f_i = (torque_x / (4 * lever_arm)) * sy_i
        ax = torque[0] / (4.0 * arm_len)
        thrusts += ax * sy

        # Pitch: delta_f_i = (-torque_y / (4 * lever_arm)) * sx_i
        ay = -torque[1] / (4.0 * arm_len)
        thrusts += ay * sx

        # Yaw: delta_f_i = (torque_z/(4c)) * yaw_sign_i
        c = self.rotor_cd / self.rotor_cf
        assert c > 0.0, "c must be positive"
        az = torque[2] / (4.0 * c)
        thrusts += az * self.yaw_signs

        # Ensure non-negative thrust
        thrusts = np.clip(thrusts, 0.0, None)

        # Convert thrust to motor speeds
        speeds = np.sqrt(thrusts / self.rotor_cf)
        return np.clip(speeds, 0.0, self.motor_max_rot_velocity)

    def wrench_to_motor_speeds_problematic(self, force: float, torque: np.ndarray) -> np.ndarray:
        """
        Keep this function for educational purposes.
        The pseudoinverse solution `w = np.linalg.pinv(mat) @ wrench` gives a least-squares solution
        to the linear system relating the desired force and torques [Fz, τx, τy, τz] to the squared
        motor speeds `w = ω²`. Physically, a motor cannot spin at a negative speed, so `w_i = ω_i^2
        ≥ 0`. If the pseudoinverse returns negative entries, clipping them to zero introduces a
        deviation from the requested wrench.

        Map a desired wrench (force + torque) directly to motor angular velocities (rad/s).

        We form a 4x4 allocation matrix `mat` such that:
          [Fz, τx, τy, τz] = mat @ w
        with w = [ω1*|ω1|, ω2*|ω2|, ω3*|ω3|, ω4*|ω4|]. In this system ω_i ≥ 0,
        hence w_i = ω_i^2.
        """
        x = self.rotor_positions[:, 0]
        y = self.rotor_positions[:, 1]

        mat = np.vstack(
            [
                self.rotor_cf * np.ones(4),
                self.rotor_cf * y,
                -self.rotor_cf * x,
                self.rotor_cd * self.yaw_signs,
            ]
        )

        wrench = np.array([force, torque[0], torque[1], torque[2]], dtype=float)
        w = np.linalg.pinv(mat) @ wrench
        w = np.clip(w, 0.0, None)
        speeds = np.sqrt(w)
        return np.clip(speeds, 0.0, self.motor_max_rot_velocity)
