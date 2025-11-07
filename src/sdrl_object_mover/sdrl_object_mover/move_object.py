import math
import subprocess

import rclpy
from rclpy.node import Node

import numpy as np


class ObjectMover(Node):
    def __init__(self):
        super().__init__("object_mover")

        # Parameters
        self.declare_parameter("entity_name", "red_ball")
        self.declare_parameter("world_name", "ground_plane_world")
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("center_z", 0.0)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("speed", 0.0)
        self.declare_parameter("trajectory", "circle")  # "circle" or "random"

        self.entity_name: str = self.get_parameter("entity_name").get_parameter_value().string_value
        self.world_name: str = self.get_parameter("world_name").get_parameter_value().string_value
        self.center_x: float = self.get_parameter("center_x").get_parameter_value().double_value
        self.center_y: float = self.get_parameter("center_y").get_parameter_value().double_value
        self.center_z: float = self.get_parameter("center_z").get_parameter_value().double_value
        self.initial_x: float = self.get_parameter("initial_x").get_parameter_value().double_value
        self.initial_y: float = self.get_parameter("initial_y").get_parameter_value().double_value
        self.speed: float = self.get_parameter("speed").get_parameter_value().double_value
        self.trajectory: str = self.get_parameter("trajectory").get_parameter_value().string_value

        dx = self.initial_x - self.center_x
        dy = self.initial_y - self.center_y
        self.radius = math.hypot(dx, dy)
        self.angle = math.atan2(dy, dx)
        self.omega = self.speed / self.radius
        assert self.radius > 0.0, "Radius must be positive"
        assert self.speed >= 0.0, "Speed must be non-negative"
        assert self.omega >= 0.0, "Omega must be non-negative"

        self.rng = np.random.default_rng(42)

        self.get_logger().info(
            f"Moving entity '{self.entity_name}' with trajectory={self.trajectory}"
        )

        # Gazebo topic to command velocity
        self.cmd_topic = f"/model/{self.entity_name}/cmd_vel"

        self.timer = self.create_timer(3.0, self.update)  # (seconds)

    def update(self):
        if self.trajectory == "circle":
            v = self.speed
            w = self.omega
            payload = f"linear: {{x: {v}}}, angular: {{z: {w}}}"

        elif self.trajectory == "random":
            v = self.speed
            w = self.omega

            v += self.rng.uniform(0.0, 0.5)
            w += self.rng.uniform(0.0, 0.5)

            payload = f"linear: {{x: {v}, y: {v}}}, angular: {{z: {w}}}"

        else:
            raise ValueError(f"Invalid trajectory: {self.trajectory}")

        self.send_gz_twist(payload)

    def send_gz_twist(self, payload: str):
        cmd = [
            "gz",
            "topic",
            "-t",
            self.cmd_topic,
            "-m",
            "gz.msgs.Twist",
            "-p",
            payload,
        ]
        try:
            # NOTE: If using timeout, it will raise an exception if the command takes too long to
            # complete. Interestingly, this command takes longer than I expected. I don't know why.
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as exc:
            self.get_logger().warn(f"Failed to send gz twist: {exc}")
            raise RuntimeError(f"Failed to send gz twist: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
