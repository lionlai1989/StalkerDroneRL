"""
Geometric Controller

Pure algorithmic SE(3) controller inspired by:

T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a quadrotor
UAV on SE(3)," Proceedings of the 49th IEEE Conference on Decision and Control (CDC), 2010.
"""

import math

import numpy as np
from geometry_msgs.msg import Pose, Twist


class GeometricController:
    def __init__(self):
        # Control gains and limits (aligned with project defaults)
        self.kp_position = 3.0
        self.kv_linvel = 5.0
        self.kr_rotmat = 6.0
        self.kw_angvel = 3.0
        self.max_accel = 12.0
        self.max_tilt_angle = math.pi / 12  # ~15 degrees

        # Physical and actuator parameters (must match SDF motor model)
        self.mass = 1.5
        # Diagonal inertia [Ixx, Iyy, Izz]
        self.inertia = np.array([0.0347563, 0.07, 0.0977], dtype=float)
        self.rotor_cf = 8.54858e-06  # thrust coefficient
        self.rotor_cd = 0.016  # drag (moment) coefficient
        self.motor_max_rot_velocity = 800.0

        # Geometry (lever arm = max rotor radius in XY plane)
        self.lever_arm = 0.255539
        # Rotor positions (x,y,z) in body frame on diagonals, order:
        #   rotor_0: +x, -y; rotor_1: -x, +y; rotor_2: +x, +y; rotor_3: -x, -y
        L = self.lever_arm / math.sqrt(2.0)
        self.rotor_positions = [
            np.array([+L, -L, 0.0], dtype=float),
            np.array([-L, +L, 0.0], dtype=float),
            np.array([+L, +L, 0.0], dtype=float),
            np.array([-L, -L, 0.0], dtype=float),
        ]
        # Rotor yaw spin directions (+1 for ccw, -1 for cw) in same order
        self.yaw_signs = [1.0, 1.0, -1.0, -1.0]

    def compute_motor_speeds(
        self, curr_pose: Pose, curr_twist: Twist, desired_pose: Pose, desired_twist: Twist
    ) -> list[float]:
        """Compute 4 motor speeds (rad/s) from current and desired states.

        Inputs:
        - curr_pose: geometry_msgs/Pose (current world pose)
        - curr_twist: geometry_msgs/Twist (current world linear vel, body angular vel)
        - desired_pose: geometry_msgs/Pose (desired world pose)
        - desired_twist: geometry_msgs/Twist (desired world linear vel, body angular vel)

        All vectors are numpy arrays of shape (3,), quaternion is (w,x,y,z).
        Linear velocities are in world frame; angular velocities are in body frame.
        """
        # Current state
        p = curr_pose.position
        q = curr_pose.orientation
        v = curr_twist.linear
        w = curr_twist.angular
        curr_pos = np.array([p.x, p.y, p.z], dtype=float)
        curr_quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=float)
        curr_lin_vel = np.array([v.x, v.y, v.z], dtype=float)
        curr_ang_vel_body = np.array([w.x, w.y, w.z], dtype=float)

        # Desired state
        dp = desired_pose.position
        dq = desired_pose.orientation
        des_pos = np.array([dp.x, dp.y, dp.z], dtype=float)
        des_quat_wxyz = np.array([dq.w, dq.x, dq.y, dq.z], dtype=float)
        des_yaw = float(self.quaternion_to_euler_yaw(des_quat_wxyz))
        dv = desired_twist.linear
        dw = desired_twist.angular
        des_lin_vel = np.array([dv.x, dv.y, dv.z], dtype=float)
        des_ang_vel_body = np.array([dw.x, dw.y, dw.z], dtype=float)

        # --- Errors ---
        e_p = curr_pos - des_pos
        e_v = curr_lin_vel - des_lin_vel

        gravity = np.array([0.0, 0.0, -9.81])

        # Desired acceleration with PD control + gravity
        a_des = -self.kp_position * e_p - self.kv_linvel * e_v - gravity
        a_mag = float(np.linalg.norm(a_des))
        if a_mag > self.max_accel and a_mag > 1e-6:
            a_des = a_des * (self.max_accel / a_mag)

        # Rotations
        R_d = self.compute_desired_orientation(a_des, des_yaw)
        R = self.quaternion_to_rotation_matrix(curr_quat_wxyz)

        # Attitude error and angular rate error
        e_R = self.compute_rotation_error(R, R_d)
        des_omega_body = R.T.dot(R_d.dot(des_ang_vel_body))
        e_omega = curr_ang_vel_body - des_omega_body

        # Recompute a_cmd (without clamp) for thrust calculation
        a_cmd = -self.kp_position * e_p - self.kv_linvel * e_v - gravity
        body_z = R[:, 2]
        f_total = self.mass * float(np.dot(a_cmd, body_z))
        if f_total < 0.0:
            f_total = 0.0

        # Torques: -k_R e_R - k_w e_w + omega x I omega
        tau = -self.kr_rotmat * e_R - self.kw_angvel * e_omega
        I_omega = curr_ang_vel_body * self.inertia
        tau += np.cross(curr_ang_vel_body, I_omega)

        thrusts = self.mix_wrench_to_thrusts(f_total, tau)
        return self.thrusts_to_motor_speeds(thrusts)

    @staticmethod
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

    @staticmethod
    def quaternion_to_euler_yaw(q):
        w, x, y, z = q
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

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

    @staticmethod
    def compute_rotation_error(R, R_d):
        R_err = R_d.T.dot(R) - R.T.dot(R_d)
        return np.array(
            [
                0.5 * (R_err[2, 1] - R_err[1, 2]),
                0.5 * (R_err[0, 2] - R_err[2, 0]),
                0.5 * (R_err[1, 0] - R_err[0, 1]),
            ]
        )

    def mix_wrench_to_thrusts(self, total_thrust, torque):
        if total_thrust < 0.0:
            total_thrust = 0.0
        tau_x, tau_y, tau_z = float(torque[0]), float(torque[1]), float(torque[2])
        L = self.lever_arm
        kf = self.rotor_cf
        km = self.rotor_cd
        c = (km / kf) if kf != 0 else 0.0

        # Sign of rotor positions
        sx = [1.0 if p[0] >= 0 else -1.0 for p in self.rotor_positions]
        sy = [1.0 if p[1] >= 0 else -1.0 for p in self.rotor_positions]

        # Start with equal thrust share
        f = [0.25 * total_thrust] * 4

        # Roll (tau_x)
        ax = (tau_x / (4.0 * L)) if L != 0 else 0.0
        for i in range(4):
            f[i] += ax * sy[i]
        # Pitch (tau_y)
        ay = (-tau_y / (4.0 * L)) if L != 0 else 0.0
        for i in range(4):
            f[i] += ay * sx[i]
        # Yaw (tau_z)
        az = (tau_z / (4.0 * c)) if c != 0 else 0.0
        for i in range(4):
            f[i] += az * self.yaw_signs[i]

        # Ensure non-negative thrust
        for i in range(4):
            if f[i] < 0.0:
                f[i] = 0.0
        return f

    def thrusts_to_motor_speeds(self, thrusts):
        if len(thrusts) != 4:
            raise ValueError("Expected 4 thrust values")
        speeds = []
        for f in thrusts:
            w = math.sqrt(f / self.rotor_cf) if self.rotor_cf > 0 else 0.0
            if w > self.motor_max_rot_velocity:
                w = self.motor_max_rot_velocity
            speeds.append(w)
        return speeds
