"""
Geometric Controller

T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a quadrotor
UAV on SE(3)," Proceedings of the 49th IEEE Conference on Decision and Control (CDC), 2010.
"""

import math

import numpy as np
from geometry_msgs.msg import Pose, Twist
from sdrl_geometric_controller.transform import quat_to_rotmat, quat_to_euler
from sdrl_geometric_controller.motor_mixing import wrench_to_motor_speeds
from sdrl_geometric_controller.quadcopter_params import QuadcopterParams, GRAVITY


def rotation_error(rot_current: np.ndarray, rot_desired: np.ndarray) -> np.ndarray:
    """Return twice the standard geometric attitude error, preserving the tuned gain scale."""
    rot_err = rot_desired.T @ rot_current - rot_current.T @ rot_desired
    return np.array(
        [
            0.5 * (rot_err[2, 1] - rot_err[1, 2]),
            0.5 * (rot_err[0, 2] - rot_err[2, 0]),
            0.5 * (rot_err[1, 0] - rot_err[0, 1]),
        ]
    )


class GeometricController:
    def __init__(self):
        # Proportional gain on position error for position control
        self.kp_position = 3.0
        # Gain on the world-frame velocity setpoint error
        self.kv_linvel = 5.0
        # Proportional gain on rotation matrix error for rotation matrix control
        self.kr_rotmat = np.array([6.0, 6.0, 3.0])
        # Damping gain on measured body angular velocity
        self.kw_angvel = np.array([3.0, 3.0, 1.5])

        self.drone_params = QuadcopterParams()

    def compute_motor_speeds(
        self, curr_pose: Pose, curr_twist: Twist, desired_pose: Pose, desired_twist: Twist
    ) -> np.ndarray:
        """Compute 4 motor speeds (rad/s) from current and desired states.

        Inputs:
        - curr_pose: actual body pose in world coordinates.
        - curr_twist: world-relative velocities expressed in the actual body frame.
        - desired_pose: world position setpoint and reference-frame orientation.
        - desired_twist: world-relative linear velocity setpoint expressed in that
          reference frame; angular velocity must be zero (no rate feedforward).

        Position and velocity are independent setpoints. The reference orientation
        supplies the velocity basis and heading; thrust determines the control attitude.
        """
        force, torque = self.compute_wrench(curr_pose, curr_twist, desired_pose, desired_twist)
        return wrench_to_motor_speeds(
            force,
            torque,
            self.drone_params.rotor_positions,
            self.drone_params.rotor_cf,
            self.drone_params.rotor_cd,
            self.drone_params.yaw_signs,
            self.drone_params.motor_max_rot_velocity,
        )

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
        curr_rot = quat_to_rotmat(curr_wxyz[0], curr_wxyz[1], curr_wxyz[2], curr_wxyz[3])

        # curr_twist.linear is in Body Frame. Convert to World Frame.
        curr_linvel_body = np.array(
            [curr_twist.linear.x, curr_twist.linear.y, curr_twist.linear.z], dtype=float
        )
        curr_linvel = curr_rot @ curr_linvel_body

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
        _, _, des_yaw = quat_to_euler(des_wxyz[0], des_wxyz[1], des_wxyz[2], des_wxyz[3])

        cmd_linvel_ref = np.array(
            [desired_twist.linear.x, desired_twist.linear.y, desired_twist.linear.z], dtype=float
        )
        reference_rot = quat_to_rotmat(des_wxyz[0], des_wxyz[1], des_wxyz[2], des_wxyz[3])
        des_lin_vel = reference_rot @ cmd_linvel_ref

        e_pos = curr_pos - des_pos
        e_linvel = curr_linvel - des_lin_vel

        # Compute force
        acc_cmd = (
            -self.kp_position * e_pos
            - self.kv_linvel * e_linvel
            + GRAVITY * np.array([0.0, 0.0, 1.0])
        )  # Independent position/velocity feedback and gravity compensation.
        # Build desired attitude from a saturated acceleration command so the command is physically
        # feasible under thrust/tilt constraints.
        control_rot, acc_cmd = self.compute_desired_orientation(acc_cmd, des_yaw)
        # Thrust is along body z-axis, so project commanded world acceleration onto body z-axis.
        force = self.drone_params.mass * float(np.dot(acc_cmd, curr_rot[:, 2]))
        # Explicit force saturation to match actuator capability.
        force = float(
            np.clip(force, self.drone_params.force_z_limit[0], self.drone_params.force_z_limit[1])
        )

        # Attitude feedback and actual-body-rate damping; no angular-rate feedforward.
        e_rot = rotation_error(curr_rot, control_rot)
        torque = (
            -self.kr_rotmat * e_rot
            - self.kw_angvel * curr_angvel
            + np.cross(curr_angvel, self.drone_params.inertia * curr_angvel)
        )
        return force, torque

    def compute_desired_orientation(self, acc, yaw):
        """Compute desired attitude from acceleration and yaw.

        Returns:
            R_d: desired rotation matrix (body axes in world frame)
            a: saturated acceleration command actually used to build R_d

        Saturation strategy:
        1) Keep vertical acceleration positive to avoid inverted thrust.
        2) Enforce thrust-limited total acceleration (max_accel).
        3) Enforce tilt-limited horizontal acceleration.
        """
        a = acc.copy()
        # Keep positive vertical acceleration so desired thrust direction stays
        # well-defined and points generally upward.
        if a[2] < 1e-6:
            a[2] = 1e-6
        # Prioritize vertical channel first; altitude authority is more critical
        # than lateral authority during aggressive commands.
        if a[2] > self.drone_params.max_accel:
            a[2] = self.drone_params.max_accel
        horiz = math.hypot(a[0], a[1])
        # Horizontal acceleration must satisfy BOTH:
        # - thrust sphere: ||a|| <= max_accel
        # - tilt cone:     ||a_xy|| <= tan(max_tilt) * |a_z|
        max_horiz_by_thrust = math.sqrt(max(0.0, self.drone_params.max_accel**2 - a[2] ** 2))
        max_horiz_by_tilt = math.tan(self.drone_params.max_tilt_angle) * abs(a[2])
        max_horiz = min(max_horiz_by_thrust, max_horiz_by_tilt)
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
        # If heading direction and z-axis become nearly collinear, perturb yaw
        # slightly to keep cross-products numerically stable.
        if abs(float(np.dot(z_w_des, x_c_des))) > 0.999:
            x_c_des = np.array([math.cos(yaw + 0.01), math.sin(yaw + 0.01), 0.0])
        # Compute orthonormal basis
        y_w_des = np.cross(z_w_des, x_c_des)
        y_w_des /= float(np.linalg.norm(y_w_des))
        x_w_des = np.cross(y_w_des, z_w_des)
        x_w_des /= float(np.linalg.norm(x_w_des))
        # Rotation matrix columns are body axes in world frame
        R_d = np.column_stack((x_w_des, y_w_des, z_w_des))
        return R_d, a
