import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ControllerConfig:
    # Gains and limits
    kp_position: float = 3.0  # Proportional gain on position error for position control
    kv_linvel: float = 5.0  # Derivative gain on linear velocity error for linear velocity control
    kr_rotmat: float = 6.0  # Proportional gain on rotation matrix error for rotation matrix control
    kw_angvel: float = 3.0  # Derivative gain on angular velocity error for angular velocity control
    max_accel: float = 12.0  # max acceleration for the drone
    max_tilt_angle: float = np.pi / 12  # (rad) ~15 degrees

    # Motor/mixer parameters
    # Overwrite rotors setting defined in "gz-sim-multicopter-motor-model-system"
    # NOTE: The values defined here MUST be the same as the value defined in MulticopterMotorModel.
    motor_max_rot_velocity: float = 800.0  # Maximum rotor angular velocity (rad/s)
    rotor_cf: float = 8.54858e-06  # Thrust coefficient cf or motorConstant (N / (rad/s)^2)
    rotor_cd: float = 0.016  # Drag coefficient cd or momentConstant (N*m / (rad/s)^2)

    # Physical parameters
    mass: float = 1.5
    inertia_diag: Tuple[float, float, float] = (0.0347563, 0.07, 0.0977)
    lever_arm: float = 0.255539
    yaw_signs: Tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)


DEFAULT_CONFIG = ControllerConfig()


def compute_rotor_positions(lever_arm: float):
    """Return 4 rotor XY positions on diagonals with radius = lever_arm.

    The order matches the plugin convention:
      rotor_0: x+, y-
      rotor_1: x-, y+
      rotor_2: x+, y+
      rotor_3: x-, y-
    """
    L = lever_arm / math.sqrt(2.0)
    return [
        np.array([+L, -L, 0.0], dtype=float),
        np.array([-L, +L, 0.0], dtype=float),
        np.array([+L, +L, 0.0], dtype=float),
        np.array([-L, -L, 0.0], dtype=float),
    ]


def apply_to_controller(conf: ControllerConfig, controller) -> None:
    controller.kp_position = conf.kp_position
    controller.kv_linvel = conf.kv_linvel
    controller.kr_rotmat = conf.kr_rotmat
    controller.kw_angvel = conf.kw_angvel
    controller.max_accel = conf.max_accel
    controller.max_tilt_angle = conf.max_tilt_angle

    controller.motor_max_rot_velocity = conf.motor_max_rot_velocity
    controller.rotor_cf = conf.rotor_cf
    controller.rotor_cd = conf.rotor_cd

    controller.mass = conf.mass
    controller.inertia = np.array(conf.inertia_diag, dtype=float)
    controller.lever_arm = conf.lever_arm
    controller.yaw_signs = list(conf.yaw_signs)
    controller.rotor_positions = compute_rotor_positions(conf.lever_arm)
