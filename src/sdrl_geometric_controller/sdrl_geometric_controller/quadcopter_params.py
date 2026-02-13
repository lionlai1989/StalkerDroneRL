import math

import numpy as np

GRAVITY = 9.81


class QuadcopterParams:
    def __init__(self):
        # NOTE: Physical parameters defined here MUST match the values defined in
        # "src/sdrl_lionquadcopter/models/lion_quadcopter.sdf"
        # "src/sdrl_lionquadcopter/models/x3_uav/model.sdf"
        # Mass of the drone (kg)
        self.mass = 1.5
        # Diagonal inertia [Ixx, Iyy, Izz]
        self.inertia = np.array([0.0347563, 0.07, 0.0977], dtype=float)
        # Thrust coefficient cf or motorConstant (N / (rad/s)^2)
        self.rotor_cf = 8.54858e-06
        # Moment constant (meter, i.e. torque (N*m) / thrust (N) ratio) from gz-sim <momentConstant>
        # Read https://github.com/gazebosim/gz-sim/blob/gz-sim10/src/systems/multicopter_motor_model/MulticopterMotorModel.cc
        self.moment_constant = 0.016
        # Drag (yaw torque) coefficient cd or k_m (N*m / (rad/s)^2)
        self.rotor_cd = self.rotor_cf * self.moment_constant
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

        # If a rotor spins:
        #   CCW, the body experiences a CW reaction torque
        #   CW, the body experiences a CCW reaction torque
        # In ROS (read https://www.ros.org/reps/rep-0103.html):
        #   +Z is up
        #   +yaw is CCW when viewed from above
        # So, if a rotor spins:
        #   CW, the body experiences a CCW reaction torque, yaw sign is +1.
        #   CCW, the body experiences a CW reaction torque, yaw sign is -1.
        self.yaw_signs = np.array([-1, -1, +1, +1])

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
        assert self.moment_constant > 0.0, "moment_constant must be positive"
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

        self.hover_thrust = self.mass * GRAVITY * 1.1  # Add 10% to help the drone hover.
        # Torque X = sum(y_i * F_i)
        ys = self.rotor_positions[:, 1]
        # Max positive torque: motors with positive y at max thrust
        tx_max = np.sum(ys[ys > 0]) * max_thrust_per_motor
        # Min negative torque: motors with negative y at max thrust
        tx_min = np.sum(ys[ys < 0]) * max_thrust_per_motor
        torque_x_limit = (tx_min, tx_max)

        # Torque Y = sum(-x_i * F_i)
        xs = self.rotor_positions[:, 0]
        # Max positive torque: motors with negative x (so -x > 0) at max thrust
        ty_max = np.sum(-xs[xs < 0]) * max_thrust_per_motor
        # Min negative torque: motors with positive x (so -x < 0) at max thrust
        ty_min = np.sum(-xs[xs > 0]) * max_thrust_per_motor
        torque_y_limit = (ty_min, ty_max)

        # Torque Z = sum(yaw_sign_i * M_i)
        # M_i = rotor_cd * omega_i^2 = rotor_cd * (max_vel^2)
        max_moment_per_motor = self.rotor_cd * (self.motor_max_rot_velocity**2)
        # Max positive torque: motors with positive yaw_sign at max speed
        tz_max = np.sum(self.yaw_signs[self.yaw_signs > 0]) * max_moment_per_motor
        # Min negative torque: motors with negative yaw_sign at max speed
        tz_min = np.sum(self.yaw_signs[self.yaw_signs < 0]) * max_moment_per_motor
        torque_z_limit = (tz_min, tz_max)
        # Print the wrench limits shows:
        print(f"force_z_limit: {force_z_limit}")  # (0.0, 21.8843648) -> OK
        print(f"torque_x_limit: {torque_x_limit}")  # (-2.297858304, 2.297858304) -> OK
        print(f"torque_y_limit: {torque_y_limit}")  # (-1.422483712, 1.422483712) -> OK
        print(f"torque_z_limit: {torque_z_limit}")  # (-0.1750749184, 0.1750749184) -> OK

        # TODO: To prevent the drone from crashing, limit torque. Consider curriculum learning.
        torque_x_limit = (-0.1, 0.1)
        torque_y_limit = (-0.1, 0.1)
        torque_z_limit = (-0.1, 0.1)
        return force_z_limit, torque_x_limit, torque_y_limit, torque_z_limit
