import numpy as np


def wrench_to_motor_speeds(
    force: float,
    torque: np.ndarray,
    rotor_positions: np.ndarray,
    rotor_cf: float,
    rotor_cd: float,
    yaw_signs: np.ndarray,
    motor_max_rot_velocity: float,
) -> np.ndarray:
    """
    Map a desired wrench (force + torque) to rotor speeds using the
    sign-based lever-arm mixing approach.
    """

    # Initialize thrusts equally (baseline total force)
    thrusts = np.full(4, 0.25 * force, dtype=float)

    # Roll and pitch distribution using sign-based lever arms
    x = rotor_positions[:, 0]
    y = rotor_positions[:, 1]
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
    c = rotor_cd / rotor_cf
    assert c > 0.0, "c must be positive"
    az = torque[2] / (4.0 * c)
    thrusts += az * yaw_signs

    # Ensure non-negative thrust
    thrusts = np.clip(thrusts, 0.0, None)

    # Convert thrust to motor speeds
    speeds = np.sqrt(thrusts / rotor_cf)
    return np.clip(speeds, 0.0, motor_max_rot_velocity)


def wrench_to_motor_speeds_problematic(
    force: float,
    torque: np.ndarray,
    rotor_positions: np.ndarray,
    rotor_cf: float,
    rotor_cd: float,
    yaw_signs: np.ndarray,
    motor_max_rot_velocity: float,
) -> np.ndarray:
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
    x = rotor_positions[:, 0]
    y = rotor_positions[:, 1]

    mat = np.vstack(
        [
            rotor_cf * np.ones(4),
            rotor_cf * y,
            -rotor_cf * x,
            rotor_cd * yaw_signs,
        ]
    )

    wrench = np.array([force, torque[0], torque[1], torque[2]], dtype=float)
    w = np.linalg.pinv(mat) @ wrench
    w = np.clip(w, 0.0, None)
    speeds = np.sqrt(w)
    return np.clip(speeds, 0.0, motor_max_rot_velocity)
