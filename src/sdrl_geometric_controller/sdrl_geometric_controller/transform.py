import math
import numpy as np


def roll_pitch_to_tilt(roll: float, pitch: float) -> float:
    """
    Calculates the combined tilt angle from roll and pitch angles,
    assuming input and output units are radians.

    Args:
        roll: The roll angle in radians (rotation around X-axis).
        pitch: The pitch angle in radians (rotation around Y-axis).

    Returns:
        The total tilt angle in radians.
    """
    # Calculate the cosine of the angle between the device's Z-axis and the vertical vector.
    cos_val = math.cos(pitch) * math.cos(roll)

    # Constrain the value to the valid range [-1, 1] to handle potential
    # floating-point inaccuracies near the limits of the acos function.
    if cos_val > 1.0:
        cos_val = 1.0
    elif cos_val < -1.0:
        cos_val = -1.0

    return math.acos(cos_val)


def quat_to_rotmat(w, x, y, z) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    n = w * w + x * x + y * y + z * z

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
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def quat_to_euler(w, x, y, z) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z
