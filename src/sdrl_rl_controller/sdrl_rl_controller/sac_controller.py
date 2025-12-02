import math
import numpy as np
from stable_baselines3 import SAC
from sdrl_geometric_controller.transform import quat_to_rotmat, quat_to_euler, roll_pitch_to_tilt
from sdrl_geometric_controller.motor_mixing import wrench_to_motor_speeds
from sdrl_geometric_controller.quadcopter_params import QuadcopterParams
from sdrl_rl_controller.observation_state_action import (
    compute_tracking_error,
    compute_normalized_observation,
    action_to_wrench,
)


class SacController:
    def __init__(self, model_path: str):
        self.model = SAC.load(model_path)
        # QuadcopterParams is used to:
        # 1. Get the wrench limits
        # 2. Convert normalized actions to motor speeds via wrench
        self.drone_params = QuadcopterParams()
        self.wrench_limits = {
            "force_z": self.drone_params.force_z_limit,
            "torque_x": self.drone_params.torque_x_limit,
            "torque_y": self.drone_params.torque_y_limit,
            "torque_z": self.drone_params.torque_z_limit,
        }

    def compute_motor_speeds(
        self,
        current_pose,
        current_twist,
        desired_pose,
        desired_twist,
    ) -> np.ndarray:
        # Compute observation
        dx, dy, dz, dvx, dvy, dvz, roll, pitch, yaw, p, q, r = compute_tracking_error(
            current_pose, current_twist, desired_pose, desired_twist
        )

        obs = compute_normalized_observation(dx, dy, dz, dvx, dvy, dvz, roll, pitch, p, q, r)

        # Predict action
        action, _ = self.model.predict(obs, deterministic=True)

        # Convert to motor speeds
        return self._action_to_motors(action)

    def _action_to_motors(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized actions to motor speeds via wrench.
        In this implementation, we do not use expert control signal for residual learning. If we use
        expert control signal, the initial reward/return will be very high, and no matter what
        the agent does, it will never be higher than the initial reward/return which results in
        the agent never learning to explore.

        """
        force_rl, torque_rl = action_to_wrench(
            action, self.wrench_limits, self.drone_params.hover_thrust
        )

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
