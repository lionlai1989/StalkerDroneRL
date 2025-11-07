/**
 * This file implements the lion quadcopter system plugin.
 */

#pragma once

#include <atomic>
#include <geometry_msgs/msg/accel.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gz/math/Matrix3.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/camera_info.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/AngularVelocity.hh>
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sdf/sdf.hh>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace sdrl {

class LionQuadcopter : public gz::sim::System,
                       public gz::sim::ISystemConfigure,
                       public gz::sim::ISystemPreUpdate,
                       public gz::sim::ISystemPostUpdate {
  public:
    LionQuadcopter(void);
    ~LionQuadcopter(void) override;

    // Overridden function from ISystemConfigure
    void Configure(const gz::sim::Entity &entity, const std::shared_ptr<const sdf::Element> &sdf,
                   gz::sim::EntityComponentManager &ecm, gz::sim::EventManager &) override;

    // Overridden function from ISystemPreUpdate
    void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override;

    // Overridden function from ISystemPostUpdate
    void PostUpdate(const gz::sim::UpdateInfo &info,
                    const gz::sim::EntityComponentManager &ecm) override;

    void init_ros_subscribers(std::string cmd_odom_topic = "cmd_odom",
                              std::string motor_speed_topic = "ros/motor_speed");

    void init_ros_publishers(std::string gt_odom_topic = "gt_odom",
                             std::string camera_image_topic = "ros_bottom_cam/image_raw",
                             std::string camera_info_topic = "ros_bottom_cam/camera_info",
                             std::string camera_pose_topic = "ros_bottom_cam/pose");

    void init_gz_subscribers(std::string image_topic = "/X3/gz_bottom_cam/image_raw",
                             std::string camera_info_topic = "/X3/gz_bottom_cam/camera_info");

    // ROS Node Subscribers Callbacks
    void cb_cmd_odom(const nav_msgs::msg::Odometry::SharedPtr cmd);
    void cb_ros_motor_cmd(const std_msgs::msg::Float32MultiArray::SharedPtr cmd);

    // Gazebo Node Subscribers Callbacks
    void repub_ros_image(const gz::msgs::Image &gzImg);
    void repub_ros_camera_info(const gz::msgs::CameraInfo &gzInfo);

    void load_drone_config(const std::shared_ptr<const sdf::Element> &sdf,
                           gz::sim::EntityComponentManager &ecm);

    // Gazebo variables
    gz::sim::Entity model_entity{gz::sim::kNullEntity};    // indicates x3 drone model
    gz::sim::Entity baselink_entity{gz::sim::kNullEntity}; // indicates baselink/body of x3 drone

    bool camera_info_callback_enabled{true}; // TODO: add this flag

    // rclcpp configuration
    rclcpp::Node::SharedPtr ros_node{nullptr};
    rclcpp::CallbackGroup::SharedPtr callback_group{nullptr};

    // ROS Interfaces
    geometry_msgs::msg::Pose cmd_pose;
    geometry_msgs::msg::Twist cmd_twist;
    nav_msgs::msg::Odometry odom; // why do we need this if we have gt_pose

    // ROS Node Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr cmd_odom_subscriber{nullptr};
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr ros_motor_subscriber{nullptr};

    // ROS Node Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_gt_odom{nullptr};
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_camera_image{nullptr};
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_camera_info{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_camera_pose{nullptr};

    // Ground truth state of the drone acquired from Gazebo. These values are used for geometric
    // control. In theory, we should implement a pose estimation filter using IMU and GPS which is a
    // different topic. We can just use ground truth pose for control for now.
    gz::math::Pose3d gt_pose;
    gz::math::Vector3d gt_linear_velocity;
    gz::math::Vector3d gt_linear_accel; // calculated from gt_linear_velocity
    gz::math::Vector3d gt_angular_velocity;

    // Fixed transformation (R + T) from drone base to camera frame
    gz::math::Pose3d transform_b2c;

    // Gazebo Node
    gz::transport::Node gz_node;
    gz::transport::Node::Publisher motor_pub;
    std::string motor_cmd_topic;
    // Gazebo subscription topic names for later unsubscription on teardown
    std::string gz_image_topic;
    std::string gz_camera_info_topic;

    // motor command cache (from ROS topic)
    std::array<double, 4> ros_motor_speeds{0.0, 0.0, 0.0, 0.0};
    bool ros_motor_cmd_available{false};

    /**
     * Teardown concurrency guard:
     * - shutting_down: set true at the start of the destructor to stop accepting new work.
     *   All callbacks check this flag and return early when it is set.
     * - inflight_callbacks: number of active callbacks. Each callback increments on entry and
     *   decrements on exit (via an RAII guard). The destructor waits until this reaches zero
     *   before destroying ROS/Gazebo objects.
     *
     * Why needed:
     * Gazebo transport and ROS 2 run callbacks on background threads. During plugin unload
     * (destructor), another thread may still be inside a callback using members like
     * ros_node, publishers, subscribers, etc. Without this guard you risk use-after-free
     * and crashes.
     *
     * Shutdown sequence:
     * 1) Set shutting_down = true (reject new work in callbacks and update hooks).
     * 2) Unsubscribe Gazebo topics to stop new transport callbacks.
     * 3) Wait until inflight_callbacks == 0 (let current callbacks finish).
     * 4) Reset/destroy subscribers, publishers, and the node.
     *
     * This yields a deterministic, thread-safe teardown across Gazebo/ROS callback threads.
     */
    std::atomic<bool> shutting_down{false};
    std::atomic<int> inflight_callbacks{0};
};

} // namespace sdrl
