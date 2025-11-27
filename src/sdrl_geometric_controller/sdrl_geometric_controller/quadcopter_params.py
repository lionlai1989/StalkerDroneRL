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

        self.hover_thrust = self.mass * GRAVITY + 1.5  # 1.5 * 9.81 = 14.715 N, +0.5 or +1.0?

        # TODO: To prevent the drone from crashing, limit rpy torque. Loosen them later.
        torque_x_limit = (-0.1, 0.1)
        torque_y_limit = (-0.1, 0.1)
        torque_z_limit = (-0.1, 0.1)
        return force_z_limit, torque_x_limit, torque_y_limit, torque_z_limit
