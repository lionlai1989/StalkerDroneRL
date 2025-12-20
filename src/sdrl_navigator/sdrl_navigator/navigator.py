"""
Navigator node.
The Navigator has three control modes:

If the control_mode is "geometric", the Navigator will use the GeometricController to compute the
motor speeds and publish them to the /X3/ros/motor_speed topic.

If the control_mode is "rl", it will load the RL model and use it to compute the motor speeds and
publish them to the /X3/ros/motor_speed topic.

If the control_mode is "rl_train", it will start the RL training mode. The motro speeds will be
published by the `train_sac.py` script.

During the RL training, if the episode is terminated or truncated, the `train_sac.py` will request
the Navigator to reset the drone to its initial pose and clear the internal state.

There are many timer-based callback functions. I list some of them here:
- state_machine_step: 2 Hz to update the state machine and compute the command odometry
- synced_image_pose_callback: 3 Hz to detect the ball and update the ball state
- controller_step: 100 Hz to compute the motor speeds

Here, I don't think state_machine_step should run as fast as synced_image_pose_callback. 4 Hz should
be enough to update the state machine and the command odometry.
"""

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
from sdrl_perception import (
    camera_info_to_intrinsics,
    compute_ray_from_pixel,
    detect_red_ball,
    intersect_ray_with_plane_z,
)
from sdrl_rl_controller import SacController
from sdrl_geometric_controller.transform import (
    quat_to_euler,
    roll_pitch_to_tilt,
    euler_to_quat,
    quat_to_rotmat,
)


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

        # Handle potential time jumps or duplicate messages
        if time_s <= self.curr_time_s:
            # This can happen if the clock jumps back or if we receive duplicate messages.
            # We should skip the update to prevent division by zero or negative dt.
            return

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
        max_altitude: float,
        crashed_height: float,
        crashed_tilt_angle: float,
    ):
        self.state = "LANDED"
        self.takeoff_target = np.array(takeoff_target)  # (3,)
        self.cruising_altitude = cruising_altitude
        self.max_altitude = max_altitude
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
            # If the distance between the current position and the takeoff target is too large, the
            # drone switches to CRASHED state.
            dist_to_takeoff = np.linalg.norm(
                np.array([curr_pos.x, curr_pos.y, curr_pos.z]) - self.takeoff_target
            )
            if dist_to_takeoff > 3.0 * self.cruising_altitude:
                self.state = "CRASHED"
                return

            if (
                roll_pitch_to_tilt(
                    *quat_to_euler(curr_quat.w, curr_quat.x, curr_quat.y, curr_quat.z)[:2]
                )
                > self.crashed_tilt_angle
            ):
                self.state = "CRASHED"
                return
            if curr_pos.z > self.max_altitude:
                self.state = "CRASHED"
                return

            # Check if the drone has reached the cruising altitude
            if abs(curr_pos.z - self.cruising_altitude) < 1:  # 0.5
                self.state = "FLYING"
            return

        if self.state == "FLYING":
            # Crash check
            if curr_pos.z < self.crashed_height:
                self.state = "CRASHED"
                return
            if (
                roll_pitch_to_tilt(
                    *quat_to_euler(curr_quat.w, curr_quat.x, curr_quat.y, curr_quat.z)[:2]
                )
                > self.crashed_tilt_angle
            ):
                self.state = "CRASHED"
                return
            if curr_pos.z > self.max_altitude:
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
        self.declare_parameter("control_mode", "geometric")  # "geometric", "rl", "rl_train"
        self.control_mode = self.get_parameter("control_mode").get_parameter_value().string_value

        self.initial_pose = (0.0, 0.0, 0.0)

        # When use_sim_time is true, this clock uses simulation time from /clock topic
        # When use_sim_time is false, this clock uses wall-clock time.
        self.clock = self.get_clock()

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

        # Service to allow external nodes (e.g., RL env) to request a reset of internal state
        self.reset_service = self.create_service(
            Trigger, "/X3/reset_navigator", self.handle_reset_service
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
        self.max_altitude = 1.5 * self.cruising_altitude
        self.takeoff_target = np.array([0.0, 0.0, self.cruising_altitude])
        self.navi_sm = NaviStateMachine(
            takeoff_target=self.takeoff_target,
            cruising_altitude=self.cruising_altitude,
            max_altitude=self.max_altitude,
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
        self.navi_state_timer_period = 1 / 10.0  # 10 Hz
        self.navi_state_timer = self.create_timer(
            self.navi_state_timer_period, self.state_machine_step, clock=self.clock
        )

        # Controller operating at motor level.
        self.controller = None
        if self.control_mode == "geometric":
            self.get_logger().info("Initializing geometric controller")
            self.controller = GeometricController()
        elif self.control_mode == "rl":
            rl_model_path = Path("/home/lion/StalkerDroneRL/sac_quadcopter_final.zip")
            self.get_logger().info(f"Initializing RL controller with model: {rl_model_path}")
            self.controller = SacController(model_path=str(rl_model_path))
        elif self.control_mode == "rl_train":
            self.get_logger().info("Initializing RL training mode")
            self.controller = None
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        self.controller_timer_period = 1 / 100.0  # 100 Hz
        self.controller_timer = self.create_timer(
            self.controller_timer_period, self.controller_step, clock=self.clock
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

        # Get current rotation for Body Frame conversion
        curr_quat = self.latest_gt_odom.pose.pose.orientation
        R = quat_to_rotmat(curr_quat.w, curr_quat.x, curr_quat.y, curr_quat.z)

        if state == "LANDED":
            # Stay at the initial pose
            pose = Pose()
            pose.position.x = self.initial_pose[0]
            pose.position.y = self.initial_pose[1]
            pose.position.z = self.initial_pose[2]
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
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

            x_err = self.takeoff_target[0] - self.latest_gt_odom.pose.pose.position.x
            y_err = self.takeoff_target[1] - self.latest_gt_odom.pose.pose.position.y
            z_err = self.takeoff_target[2] - self.latest_gt_odom.pose.pose.position.z
            v_max = 1.0  # maximum velocity (m/s)
            k_vx, k_vy, k_vz = 0.1, 0.1, 0.3  # gain to tune

            vx_w = np.clip(k_vx * x_err, -v_max, v_max)
            vy_w = np.clip(k_vy * y_err, -v_max, v_max)
            vz_w = np.clip(k_vz * z_err, -v_max, v_max)

            # Convert World Frame velocity to Body Frame
            v_world = np.array([vx_w, vy_w, vz_w])
            v_body = R.T @ v_world

            twist.linear.x = v_body[0]
            twist.linear.y = v_body[1]
            twist.linear.z = v_body[2]

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            return pose, twist

        if state == "FLYING":
            _, _, yaw = quat_to_euler(
                self.latest_gt_odom.pose.pose.orientation.w,
                self.latest_gt_odom.pose.pose.orientation.x,
                self.latest_gt_odom.pose.pose.orientation.y,
                self.latest_gt_odom.pose.pose.orientation.z,
            )
            qw, qx, qy, qz = euler_to_quat(0.0, 0.0, yaw)
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

            vx_w = self.flying_linvel_x
            vy_w = self.flying_linvel_y
            vz_w = 0.0

            # Convert World Frame velocity to Body Frame
            v_world = np.array([vx_w, vy_w, vz_w])
            v_body = R.T @ v_world

            twist.linear.x = v_body[0]
            twist.linear.y = v_body[1]
            twist.linear.z = v_body[2]

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
        k_ff = 0.3
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

        if self.navi_sm.state == "CRASHED":
            return

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

        # Publish latest desired Odometry command
        self.cmd_odom_publisher.publish(self.latest_desired_odom)

    def handle_reset_service(self, request, response):
        """Handle reset_navigator service requests.

        Resets internal navigator state (state machine, ball state, etc).
        """
        try:
            self.reset_navigator()
            response.success = True
            response.message = "Navigator internal state reset complete"
        except Exception as e:
            response.success = False
            response.message = f"Reset failed: {e}"
            self.get_logger().error(response.message)
            self.get_logger().error(traceback.format_exc())

            # Re-raise the exception to stop the program. Reset must succeed before continuing.
            raise e
        return response

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

        # Clear odometry so we don't act on stale data
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
