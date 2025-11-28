import math

import numpy as np
import rclpy
from rclpy.node import Node


# Harmonic Debian packages expose versioned subpackages
import gz.transport13 as gz_transport
from gz.msgs10.twist_pb2 import Twist


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

        # When use_sim_time is true, this clock uses simulation time from /clock topic
        # When use_sim_time is false, this clock uses wall-clock time.
        self.clock = self.get_clock()

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

        # Initialize Gazebo transport node and publisher
        self.gz_node = gz_transport.Node()
        self.gz_pub = self.gz_node.advertise(self.cmd_topic, Twist)

        self.timer = self.create_timer(3.0, self.update, clock=self.clock)  # (seconds)

    def update(self):
        msg = Twist()
        if self.trajectory == "circle":
            v = self.speed
            w = self.omega
            msg.linear.x = v
            msg.angular.z = w

        elif self.trajectory == "random":
            v = self.speed
            w = self.omega

            v += self.rng.uniform(0.0, 0.5)
            w += self.rng.uniform(0.0, 0.5)

            msg.linear.x = v
            msg.linear.y = v
            msg.angular.z = w

        else:
            raise ValueError(f"Invalid trajectory: {self.trajectory}")

        self.gz_pub.publish(msg)


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
