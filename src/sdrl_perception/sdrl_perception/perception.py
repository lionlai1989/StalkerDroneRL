from typing import Optional, Tuple

import cv2
import numpy as np


def camera_info_to_intrinsics(camera_info) -> Tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from a ROS 2 sensor_msgs/CameraInfo-like object.

    The object is expected to have a 9-element list/array attribute `k` with:
    [fx, 0, cx, 0, fy, cy, 0, 0, 1].
    """
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])
    return fx, fy, cx, cy


def detect_red_ball(cv_image: np.ndarray) -> Optional[Tuple[float, float]]:
    """Detect a red ball in a BGR image using HSV thresholding.

    Returns (u, v) pixel coordinates of the centroid if found, else None.
    """
    assert isinstance(cv_image, np.ndarray), "cv_image must be a numpy array"

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Red spans around 0/180 hue; use two bands
    lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 100, 100], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 5.0:
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = float(moments["m10"] / moments["m00"])  # u
    cy = float(moments["m01"] / moments["m00"])  # v
    return (cx, cy)


def compute_ray_from_pixel(
    uv: Tuple[float, float],
    camera_pose,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a world-space ray from a pixel using intrinsics and camera pose.

    - uv: (u, v) pixel coordinates
    - camera_pose: object with attributes position.{x,y,z} and orientation.{x,y,z,w}
    Returns (camera_center_world, direction_world), both numpy arrays.
    """
    u, v = float(uv[0]), float(uv[1])

    # Optical ray in OpenCV optical frame (x right, y down, z forward)
    dir_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    dir_opt /= np.linalg.norm(dir_opt)

    # Map optical frame to camera link frame (project-specific mapping)
    dir_cam = np.array([dir_opt[1], -dir_opt[0], dir_opt[2]], dtype=np.float64)
    dir_cam /= np.linalg.norm(dir_cam)

    # Rotation from camera to world from quaternion
    qx = float(camera_pose.orientation.x)
    qy = float(camera_pose.orientation.y)
    qz = float(camera_pose.orientation.z)
    qw = float(camera_pose.orientation.w)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    R_wc = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

    camera_center_world = np.array(
        [
            float(camera_pose.position.x),
            float(camera_pose.position.y),
            float(camera_pose.position.z),
        ],
        dtype=np.float64,
    )
    direction_world = R_wc @ dir_cam
    direction_world /= np.linalg.norm(direction_world)

    return camera_center_world, direction_world


def intersect_ray_with_plane_z(
    ray: Tuple[np.ndarray, np.ndarray], z_height: float
) -> Optional[np.ndarray]:
    """Intersect (origin + t * dir) with plane z = z_height.

    Returns 3D point (numpy array) or None if no forward intersection exists.
    """
    if ray is None:
        return None
    origin, direction = ray
    dz = float(direction[2])
    if abs(dz) < 1e-8:
        return None
    t = (float(z_height) - float(origin[2])) / dz
    if t < 0.0:
        return None
    point = origin + t * direction
    point[2] = float(z_height)
    return point


def transform_camera_to_world(point_camera: np.ndarray, camera_pose) -> np.ndarray:
    """Transform a point in camera frame to world frame using camera pose.

    camera_pose: object with orientation {x,y,z,w} and position {x,y,z}
    point_camera: shape (3, 1)
    Returns p_w with shape (3, 1)
    """
    assert point_camera.shape == (3, 1), "point_camera must be a 3x1 matrix"
    qx = float(camera_pose.orientation.x)
    qy = float(camera_pose.orientation.y)
    qz = float(camera_pose.orientation.z)
    qw = float(camera_pose.orientation.w)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    R_wc = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    t_wc = np.array(
        [
            [float(camera_pose.position.x)],
            [float(camera_pose.position.y)],
            [float(camera_pose.position.z)],
        ],
        dtype=np.float64,
    )
    p_w = R_wc @ point_camera + t_wc
    assert p_w.shape == (3, 1), "p_w must be a 3x1 matrix"
    return p_w


def transform_world_to_camera(point_world: np.ndarray, camera_pose) -> np.ndarray:
    """Transform a world point into the camera frame using camera pose.

    camera_pose: object with orientation {x,y,z,w} and position {x,y,z}
    Returns p_c with shape (3,)
    """
    qx = float(camera_pose.orientation.x)
    qy = float(camera_pose.orientation.y)
    qz = float(camera_pose.orientation.z)
    qw = float(camera_pose.orientation.w)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    R_wc = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    t_wc = np.array(
        [
            float(camera_pose.position.x),
            float(camera_pose.position.y),
            float(camera_pose.position.z),
        ],
        dtype=np.float64,
    )
    p_w = np.asarray(point_world, dtype=np.float64).reshape(3)
    p_c = R_wc.T @ (p_w - t_wc)
    return p_c
