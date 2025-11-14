"""
Navigator node.
The Navigator has two control modes:

If the control_mode is "geometric", the Navigator will use the GeometricController to compute the
motor speeds and publish them to the /X3/ros/motor_speed topic.

If the control_mode is "rl", it will load the RL model and use it to compute the motor speeds and
publish them to the /X3/ros/motor_speed topic.

During the RL training, if the episode is terminated or truncated, the `train_sac.py` will request
the Navigator to reset the drone to its initial pose and clear the internal state.

There are many timer-based callback functions. I list some of them here:
- state_machine_step: 4 Hz to update the state machine and compute the command odometry
- synced_image_pose_callback: 10 Hz to detect the ball and update the ball state
- controller_step: 100 Hz to compute the motor speeds

Here, I don't think state_machine_step should run as fast as synced_image_pose_callback. 4 Hz should
be enough to update the state machine and the command odometry.
"""

import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Twist
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger

from sdrl_geometric_controller import GeometricController
from sdrl_navigator import gz_client
from sdrl_perception import (
    camera_info_to_intrinsics,
    compute_ray_from_pixel,
    detect_red_ball,
    intersect_ray_with_plane_z,
)

WORLD = "ground_plane_world"
QUADCOPTER_MODEL = "lion_quadcopter"

# initial pose of the quadcopter
INITIAL_POSE = (0.0, 0.0, 0.0)


def quat_to_rpy(quat) -> Tuple[float, float, float]:
    """Convert a quaternion to roll, pitch, yaw."""
    roll = np.arctan2(
        2.0 * (quat.w * quat.x + quat.y * quat.z), 1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
    )
    pitch = np.arcsin(2.0 * (quat.w * quat.y - quat.z * quat.x))
    yaw = np.arctan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    )
    return roll, pitch, yaw


def rpy_to_tilt_angle(roll: float, pitch: float, yaw: float) -> float:
    """Get the tilt angle from roll and pitch angles. Yaw is ignored."""
    return float(np.sqrt(roll * roll + pitch * pitch))


def yaw_from_odom(odom: Odometry) -> float:
    q = odom.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """Convert roll, pitch, yaw to quaternion."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class BallState:
    """
    Keep track of the current and previous 3D position and velocity of the ball.
    """

    def __init__(self):
        self.curr_pos: np.ndarray | None = None
        self.curr_vel: np.ndarray | None = None
        self.curr_time_s: float | None = None
        self.prev_pos: np.ndarray | None = None
        self.prev_vel: np.ndarray | None = None
        self.prev_time_s: float | None = None
        self.detected: bool = False

    def update(self, pos: np.ndarray, time_s: float) -> None:
        self.detected = True

        # Initialize on first observation
        if self.curr_pos is None or self.curr_time_s is None:
            self.curr_pos = pos
            self.curr_time_s = time_s
            self.curr_vel = np.zeros_like(pos, dtype=float)
            self.prev_pos = pos
            self.prev_time_s = time_s
            self.prev_vel = np.zeros_like(pos, dtype=float)
            return

        # Require strictly increasing timestamps
        assert time_s > self.curr_time_s, "BallState.update requires strictly increasing time"

        # Shift current state to previous
        self.prev_pos = self.curr_pos
        self.prev_time_s = self.curr_time_s
        self.prev_vel = self.curr_vel

        # Compute new current state
        dt = time_s - self.prev_time_s
        vel = (pos - self.prev_pos) / dt
        self.curr_pos = pos
        self.curr_time_s = time_s
        self.curr_vel = vel

    def reset(self) -> None:
        """Reset the ball state to initial values."""
        self.curr_pos = None
        self.curr_vel = None
        self.curr_time_s = None
        self.prev_pos = None
        self.prev_vel = None
        self.prev_time_s = None
        self.detected = False


class NaviStateMachine:
    """Navigation state machine for the drone.
    there are 4 states:
    LANDED: the drone is landed on the ground. Immediately switch to TAKINGOFF state.

    TAKINGOFF: take off to predetermined height. If the current position to the target position is
    less than takeoff_tolerance, switch to FLYING state. If the tilt angle is greater than
    crashed_tilt_angle, switch to CRASHED state.

    FLYING: track the ball. If the tilt angle is greater than crashed_tilt_angle, switch to CRASHED
    state. If the altitude is less than crashed_height, switch to CRASHED state.

    CRASHED: the drone has crashed. Do nothing.
    """

    def __init__(
        self,
        takeoff_target: Tuple[float, float, float],
        cruising_altitude: float,
        cruising_tolerance: float,
        crashed_height: float,
        crashed_tilt_angle: float,
    ):
        self.state = "LANDED"
        self.takeoff_target = np.array(takeoff_target)  # (3,)
        self.cruising_altitude = cruising_altitude
        self.cruising_tolerance = cruising_tolerance
        self.crashed_height = crashed_height
        self.crashed_tilt_angle = crashed_tilt_angle

    def update_state(self, odom: Odometry) -> None:
        """Advance the internal state based on observations. No command output here."""
        if odom is None:
            return

        curr_pos = odom.pose.pose.position
        curr_quat = odom.pose.pose.orientation

        if self.state == "LANDED":
            # Transition immediately to TAKINGOFF when odom is available
            self.state = "TAKINGOFF"
            return

        if self.state == "TAKINGOFF":
            # Crash check
            if rpy_to_tilt_angle(*quat_to_rpy(curr_quat)) > self.crashed_tilt_angle:
                self.state = "CRASHED"
                return

            # Check if the drone has reached the cruising altitude
            if abs(curr_pos.z - self.cruising_altitude) < self.cruising_tolerance:
                self.state = "FLYING"
            return

        if self.state == "FLYING":
            # Crash check
            if curr_pos.z < self.crashed_height:
                self.state = "CRASHED"
                return
            if rpy_to_tilt_angle(*quat_to_rpy(curr_quat)) > self.crashed_tilt_angle:
                self.state = "CRASHED"
                return
            return

        if self.state == "CRASHED":
            return

        raise ValueError(f"Invalid state: {self.state}")


class Navigator(Node):
    def __init__(self):
        super().__init__("navigator")

        # `control_mode` SHALL NOT be changed after initialization.
        self.declare_parameter("control_mode", "geometric")  # "geometric" or "rl"
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value

        self.cv_bridge = CvBridge()  # Create CV bridge for image conversion

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=32
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=16
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "/X3/ros_bottom_cam/camera_info", self.camera_info_callback, qos_reliable
        )

        self.gt_odom_subscription = self.create_subscription(
            Odometry, "/X3/gt_odom", self.gt_odom_callback, qos_best_effort
        )

        self.cmd_odom_publisher = self.create_publisher(Odometry, "/X3/cmd_odom", qos_reliable)
        self.ros_motor_publisher = self.create_publisher(
            Float32MultiArray, "/X3/ros/motor_speed", qos_reliable
        )
        self.navi_state_publisher = self.create_publisher(String, "/X3/navi_state", qos_reliable)

        # Service to allow external nodes (e.g., RL env) to request a reset to initial pose
        self.reset_service = self.create_service(
            Trigger, "/X3/reset_drone_initial_pose", self.handle_reset_service
        )

        # Synchronized subscribers for image and camera pose using exact-time policy
        self.img_sub = Subscriber(
            self, Image, "/X3/ros_bottom_cam/image_raw", qos_profile=qos_best_effort
        )
        self.pose_sub = Subscriber(
            self, PoseStamped, "/X3/ros_bottom_cam/pose", qos_profile=qos_best_effort
        )
        self.exact_sync = TimeSynchronizer([self.img_sub, self.pose_sub], queue_size=10)
        self.exact_sync.registerCallback(self.synced_image_pose_callback)

        # Navigation state machine
        self.cruising_altitude = 5.0
        self.takeoff_target = np.array([0.0, 0.0, self.cruising_altitude])
        self.navi_sm = NaviStateMachine(
            takeoff_target=self.takeoff_target,
            cruising_altitude=self.cruising_altitude,
            cruising_tolerance=0.5,
            crashed_height=0.5,
            crashed_tilt_angle=np.pi / 4,  # 45 degrees
        )

        self.get_logger().info(f"Navigator initialized with control mode: {self.control_mode}")

        # Latest camera data
        self.camera_info = None
        self.latest_image = None  # numpy image (BGR)
        self.latest_cam_pose = None

        # Latest 3D ball position in world frame
        self.latest_ball_state = BallState()

        # Latest ground truth odometry
        self.latest_gt_odom = None

        # Latest desired odometry to store timestamped pose and twist.
        # Should initialize to None or default Odometry()? And update pose and twist later?
        self.latest_desired_odom: Optional[Odometry] = None

        # Smoothed commanded lateral velocities in flying state
        self.flying_linvel_x: float = 0.0
        self.flying_linvel_y: float = 0.0

        self.reset_navigator()

        # Timer to drive the high-level state machine
        self.navi_state_timer_period = 1 / 4.0  # 4 Hz
        self.navi_state_timer = self.create_timer(
            self.navi_state_timer_period, self.state_machine_step
        )

        # Controller operating at motor level.
        self.controller = None
        if self.control_mode == "geometric":
            self.get_logger().info("Initializing geometric controller")
            self.controller = GeometricController()
        elif self.control_mode == "rl":
            from sdrl_rl_controller import SacController

            rl_model_path = Path("/home/lion/StalkerDroneRL/checkpoints/sac_quadcopter_final.zip")
            if not rl_model_path.exists():
                self.get_logger().warning("RL model not found. Starting RL in training mode.")
                self.controller = None
            else:
                self.get_logger().info(f"Initializing RL controller with model: {rl_model_path}")
                self.controller = SacController(model_path=str(rl_model_path))
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        self.controller_timer_period = 1 / 100.0  # 100 Hz
        self.controller_timer = self.create_timer(
            self.controller_timer_period, self.controller_step
        )

    def gt_odom_callback(self, msg: Odometry):
        """Ground truth odometry callback."""
        self.latest_gt_odom = msg
        # self.get_logger().info(f"latest_gt_odom: {self.latest_gt_odom}")

    def synced_image_pose_callback(self, image_msg: Image, pose_msg: PoseStamped):
        """Exact-time synchronized callback for image and camera pose.

        Assumes both messages share an identical timestamp and are aligned.
        """
        assert image_msg.header.stamp == pose_msg.header.stamp, "Image and pose timestamps differ"

        # Cache latest synchronized image (numpy) and pose
        self.latest_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        self.latest_cam_pose = pose_msg.pose

        # Detect ball and cache world position if possible
        red_point = detect_red_ball(self.latest_image)
        if red_point is None:
            self.latest_ball_state.detected = False
            return

        # Compute intrinsics
        assert self.camera_info is not None, "camera_info is None"
        fx, fy, cx, cy = camera_info_to_intrinsics(self.camera_info)

        ray = compute_ray_from_pixel(red_point, self.latest_cam_pose, fx, fy, cx, cy)
        assert ray is not None, "ray cannot be None"
        z_height = 0.15  # ball radius (0.15m)
        ball_position = intersect_ray_with_plane_z(ray, z_height)
        if ball_position is None:
            self.latest_ball_state.detected = False
            return
        # Use synchronized message timestamp for monotonicity with sensor data.
        # Time Representation: seconds + nanoseconds
        time_s = float(pose_msg.header.stamp.sec) + float(pose_msg.header.stamp.nanosec) * 1e-9
        self.latest_ball_state.update(ball_position, time_s)

    def compute_desired_pose_twist(
        self,
        state: str,
    ) -> Tuple[Optional[Pose], Optional[Twist]]:
        """Compute desired Pose and Twist for the current state.

        Returns a tuple (Pose, Twist); returns (None, None) if no command should be sent.
        """
        if self.latest_gt_odom is None:
            return None, None

        if state == "LANDED":
            # Stay at the current pose
            pose = Pose()
            pose.position.x = self.latest_gt_odom.pose.pose.position.x
            pose.position.y = self.latest_gt_odom.pose.pose.position.y
            pose.position.z = self.latest_gt_odom.pose.pose.position.z
            pose.orientation.x = self.latest_gt_odom.pose.pose.orientation.x
            pose.orientation.y = self.latest_gt_odom.pose.pose.orientation.y
            pose.orientation.z = self.latest_gt_odom.pose.pose.orientation.z
            pose.orientation.w = self.latest_gt_odom.pose.pose.orientation.w
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            return pose, twist

        if state == "TAKINGOFF":
            pose = Pose()
            pose.position.x = self.takeoff_target[0]
            pose.position.y = self.takeoff_target[1]
            pose.position.z = self.takeoff_target[2]
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            # Vertical feedforward velocity toward target altitude (saturated)
            z_err = self.takeoff_target[2] - self.latest_gt_odom.pose.pose.position.z
            k_vz = 0.3  # gain to tune
            vz_max = 1.0  # maximum vertical velocity (m/s)
            twist.linear.z = np.clip(k_vz * z_err, -vz_max, vz_max)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            return pose, twist

        if state == "FLYING":
            yaw = yaw_from_odom(self.latest_gt_odom)
            qx, qy, qz, qw = rpy_to_quat(0.0, 0.0, yaw)
            pose = Pose()
            if self.latest_ball_state.detected is False:  # No ball detected
                if self.latest_desired_odom is not None:  # Stay at desired xy
                    pose.position.x = self.latest_desired_odom.pose.pose.position.x
                    pose.position.y = self.latest_desired_odom.pose.pose.position.y
                else:  # Stay at ground truth xy
                    pose.position.x = self.latest_gt_odom.pose.pose.position.x
                    pose.position.y = self.latest_gt_odom.pose.pose.position.y
            else:  # ball detected
                pose.position.x = self.latest_ball_state.curr_pos[0]
                pose.position.y = self.latest_ball_state.curr_pos[1]
            pose.position.z = self.cruising_altitude  # hold cruising altitude
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            twist = Twist()
            self.compute_lateral_feedforward()
            twist.linear.x = self.flying_linvel_x
            twist.linear.y = self.flying_linvel_y
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            return pose, twist

        if state == "CRASHED":
            return None, None

        raise ValueError(f"Invalid state: {state}")

    def compute_lateral_feedforward(self) -> None:
        """Compute smoothed, clamped, slew-limited lateral feedforward velocity from ball velocity.
        Updates self.cmd_linvel_x and self.cmd_linvel_y.
        """
        # exponential moving average (EMA) factor for smoothing
        alpha = 0.1
        # feedforward gain
        k_ff = 0.1
        # speed clamp
        v_max = 0.5  # m/s
        # deadband to suppress minor jitter
        deadband = 0.1  # m/s
        # slew-rate limit (acceleration cap)
        a_max = 1.0  # m/s^2

        if self.latest_ball_state.detected and self.latest_ball_state.curr_vel is not None:
            target_vx = k_ff * self.latest_ball_state.curr_vel[0]
            target_vy = k_ff * self.latest_ball_state.curr_vel[1]
            if abs(target_vx) < deadband:
                target_vx = 0.0
            if abs(target_vy) < deadband:
                target_vy = 0.0
        else:
            target_vx = 0.0
            target_vy = 0.0

        # EMA smoothing toward target
        smoothed_vx = alpha * target_vx + (1.0 - alpha) * self.flying_linvel_x
        smoothed_vy = alpha * target_vy + (1.0 - alpha) * self.flying_linvel_y
        # Clamp speed
        smoothed_vx = np.clip(smoothed_vx, -v_max, v_max)
        smoothed_vy = np.clip(smoothed_vy, -v_max, v_max)
        # Slew-rate limit (acceleration cap)
        dvx = float(
            np.clip(
                smoothed_vx - self.flying_linvel_x,
                -a_max * self.navi_state_timer_period,
                a_max * self.navi_state_timer_period,
            )
        )
        dvy = float(
            np.clip(
                smoothed_vy - self.flying_linvel_y,
                -a_max * self.navi_state_timer_period,
                a_max * self.navi_state_timer_period,
            )
        )
        self.flying_linvel_x = self.flying_linvel_x + dvx
        self.flying_linvel_y = self.flying_linvel_y + dvy

    def state_machine_step(self):
        """High-level state machine executed periodically."""
        # Always publish current state, even if odometry is not yet available
        state_msg = String()
        state_msg.data = self.navi_sm.state
        self.navi_state_publisher.publish(state_msg)

        if self.latest_gt_odom is None:
            return

        prev_state = self.navi_sm.state
        self.navi_sm.update_state(self.latest_gt_odom)
        # Publish again after potential update
        state_msg = String()
        state_msg.data = self.navi_sm.state
        self.navi_state_publisher.publish(state_msg)
        if prev_state != self.navi_sm.state:
            self.get_logger().info(f"navi state: {prev_state} -> {self.navi_sm.state}")

        desired_pose, desired_twist = self.compute_desired_pose_twist(
            state=self.navi_sm.state,
        )
        if desired_pose is None or desired_twist is None:
            return
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "/X3/odom"
        odom.child_frame_id = "/X3/base_footprint"
        odom.pose.pose = desired_pose
        odom.twist.twist = desired_twist
        self.latest_desired_odom = odom

        if self.navi_sm.state == "CRASHED":
            return

        # Publish latest desired Odometry command
        self.cmd_odom_publisher.publish(self.latest_desired_odom)

    def handle_reset_service(self, request, response):
        """Handle reset_drone_initial_pose service requests.

        Repositions the drone to its initial pose and resets internal navigator state.
        """
        try:
            # self.get_logger().info("Received reset_drone_initial_pose request")
            if self.navi_state_timer is not None and not self.navi_state_timer.is_canceled():
                # self.get_logger().info("Stopping state machine timer")
                self.destroy_timer(self.navi_state_timer)
                self.navi_state_timer = None

            self.reposition_drone_to_initial_pose()
            self.reset_navigator()

            response.success = True
            response.message = "Reset to initial pose complete"
        except Exception as e:
            response.success = False
            response.message = f"Reset failed: {e}"
            self.get_logger().error(response.message)
            self.get_logger().error(traceback.format_exc())

            # Re-raise the exception to stop the program. Reset must succeed before continuing.
            raise e
        finally:
            if self.navi_state_timer is None:
                # self.get_logger().info("Restarting state machine timer")
                self.navi_state_timer = self.create_timer(
                    self.navi_state_timer_period, self.state_machine_step
                )
        return response

    def reposition_drone_to_initial_pose(self):
        """Reposition the drone to its initial pose and zero motor speeds.

        This avoids tearing down and recreating the model. We send a few zero-motor
        commands to clear rotor state, then call Gazebo's set_pose service for
        the model named "lion_quadcopter" in world "ground_plane_world".
        """
        # self.get_logger().info("Repositioning drone to initial pose")

        # Publish zero motor speeds a few times to clear rotor state
        zero_msg = Float32MultiArray()
        zero_msg.data = [0.0, 0.0, 0.0, 0.0]
        for _ in range(3):
            self.ros_motor_publisher.publish(zero_msg)
            time.sleep(0.05)

        # Use persistent Gazebo Transport client (no subprocess) and retry until success
        # Ensure world is unpaused so the request is processed promptly
        try:
            gz_client.world_control(WORLD, pause=False, timeout_ms=10000)
        except Exception as exc:
            self.get_logger().warning(f"world_control(unpause) failed before set_pose: {exc}")
            raise exc

        x, y, z = INITIAL_POSE
        attempt = 0
        while True:
            attempt += 1
            try:
                gz_client.set_pose(
                    WORLD,
                    QUADCOPTER_MODEL,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    qw=1.0,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                    timeout_ms=10000,
                )
                break
            except Exception as exc:
                self.get_logger().warning(f"set_pose attempt {attempt} failed: {exc}")
                time.sleep(0.3)

    def controller_step(self):
        """Low-level control loop: compute motor speeds and publish to /X3/ros/motor_speed."""
        if self.controller is None:
            return
        if self.latest_gt_odom is None:
            return
        if self.latest_desired_odom is None:
            return

        motor_speeds = self.controller.compute_motor_speeds(
            self.latest_gt_odom.pose.pose,
            self.latest_gt_odom.twist.twist,
            self.latest_desired_odom.pose.pose,
            self.latest_desired_odom.twist.twist,
        ).tolist()
        msg = Float32MultiArray()
        msg.data = motor_speeds
        self.ros_motor_publisher.publish(msg)

    def reset_navigator(self):
        # self.get_logger().info("Resetting navigator")

        # Reset internal state to LANDED and clear cached data
        self.navi_sm.state = "LANDED"
        self.latest_image = None
        self.latest_cam_pose = None
        self.latest_ball_state.reset()
        self.latest_gt_odom = None
        self.latest_desired_odom = None
        self.flying_linvel_x = 0.0
        self.flying_linvel_y = 0.0

    def camera_info_callback(self, msg: CameraInfo):
        """Callback for camera info messages"""
        self.camera_info = msg
        # self.get_logger().info(f"Received camera info: {msg.width}x{msg.height}")


def main(args=None):
    rclpy.init(args=args)

    navigator = Navigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
