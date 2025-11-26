#include "sdrl_lionquadcopter/lionquadcopter.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <geometry_msgs/msg/accel.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gz/math/Angle.hh>
#include <gz/math/Matrix3.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/camera_info.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/wrench.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocity.hh>
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/Pose.hh>
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <thread>
#include <vector>

namespace sdrl {

LionQuadcopter::LionQuadcopter() {
    // log via std::cout before ROS node exists
    std::cout << "LionQuadcopter constructor" << std::endl;
}

LionQuadcopter::~LionQuadcopter() {
    this->shutting_down.store(true, std::memory_order_relaxed);
    // Unsubscribe Gazebo transport callbacks first to prevent late callbacks
    this->gz_node.Unsubscribe(this->gz_image_topic);
    this->gz_node.Unsubscribe(this->gz_camera_info_topic);
    // Wait briefly for any in-flight callbacks to complete
    for (int i = 0; i < 50; ++i) { // up to ~500ms
        if (this->inflight_callbacks.load(std::memory_order_relaxed) == 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Defensively forces a deterministic teardown order for topic publishers before ros_node
    this->cmd_odom_subscriber.reset();
    this->ros_motor_subscriber.reset();
    this->pub_gt_odom.reset();
    this->pub_camera_image.reset();
    this->pub_camera_info.reset();
    this->pub_camera_pose.reset();
    this->callback_group.reset();

    this->ros_node.reset();

    /**
     * Do NOT call rclcpp::shutdown() here.
     * This system plugin runs inside Gazebo's process alongside other plugins that may also
     * create ROS 2 nodes on the shared default rclcpp context. Shutting down the global
     * context would terminate those nodes as well. We instead destroy only our own ROS
     * objects above (publishers, subscribers, node) so their resources are released.
     * If per-plugin isolation is needed, create a private rclcpp::Context for this plugin
     * and shut down that context in the destructor.
     */
    if (rclcpp::ok()) {
        // rclcpp::shutdown();
    }
}

/**
 * https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1ISystemConfigure.html
 * Configure is called after the system is instantiated and all entities and components are loaded
 * from the corresponding SDF world, and before simulation begins exectution.
 */
void LionQuadcopter::Configure(const gz::sim::Entity &entity,
                               const std::shared_ptr<const sdf::Element> &sdf,
                               gz::sim::EntityComponentManager &ecm, gz::sim::EventManager &) {
    // sdf.get()->GetName() -> "plugin". Why is the name of the plugin "plugin"?

    // Why is the entity happened to be "x3" drone model? why not ground plane? or sun?
    // Store the model entity; link and geometry will be resolved in load_drone_config
    // Use parent entity of the passed in argument entity to get pose
    // entity is just a number. It doesn't have any meaning. We should always use name.
    // What should the parent entity be? ecm.ParentEntity(entity)
    this->model_entity = entity;

    // Initialize ROS context if not already initialized
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    // Create ROS node with namespace from SDF if provided, else default to /X3
    std::string ns = "/X3";
    if (sdf && sdf->HasElement("namespace")) {
        ns = sdf->Get<std::string>("namespace");
        if (ns.empty()) {
            ns = "/X3";
        }
    }
    this->ros_node = std::make_shared<rclcpp::Node>("lion_quadcopter", ns);

    // Follow Gazebo simulation time via the /clock topic.
    this->ros_node->set_parameter(rclcpp::Parameter("use_sim_time", true));

    // Create callback group
    this->callback_group =
        this->ros_node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    this->init_ros_subscribers();
    this->init_ros_publishers();
    this->init_gz_subscribers();

    // Service: reset dynamics (zero linear and angular velocity on the base link)
    this->reset_dynamics_service = this->ros_node->create_service<std_srvs::srv::Trigger>(
        "reset_dynamics", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            // Mark a reset to be applied in the next PreUpdate, where we have write access
            // to the EntityComponentManager.
            this->pending_reset_dynamics.store(true, std::memory_order_relaxed);
            response->success = true;
            response->message = "Dynamics reset requested";
        });

    // Load drone config (odom and limits) and geometry from SDF/ECM
    this->load_drone_config(sdf, ecm);

    // Tell Gazebo to generate velocity checks for the link
    gz::sim::Link(this->baselink_entity).EnableVelocityChecks(ecm, true);

    // Fixed transformation (R + T) from drone base to camera frame.
    // The camera rotates 180 deg about Y axis to look towards -Z direction.
    // Thus, the transformation matrix from base to camera frame is the inverse of the rotation
    // matrix. -180 is used because of inverse of rotation matrix.
    this->transform_b2c =
        gz::math::Pose3d(gz::math::Vector3d(0.0, 0.0, 0.0), gz::math::Quaterniond(0.0, -M_PI, 0.0));

    this->motor_cmd_topic = "/X3/gazebo/command/motor_speed";
    // Advertise motor speed publisher
    this->motor_pub = this->gz_node.Advertise<gz::msgs::Actuators>(this->motor_cmd_topic);
    if (!this->motor_pub) {
        throw std::runtime_error("Failed to advertise motor command topic");
    }
}

/**
 * PreUpdate: called every simulation iteration before physics integration.
 * Gazebo Sim (gz-physics) performs physics integration between PreUpdate and PostUpdate.
 * It is called when the simulation is paused or unpaused.
 */
void LionQuadcopter::PreUpdate(const gz::sim::UpdateInfo &info,
                               gz::sim::EntityComponentManager &ecm) {
    // Process ROS callbacks
    if (!this->shutting_down.load(std::memory_order_relaxed) && this->ros_node && rclcpp::ok()) {
        // spin_some runs as long as it takes to execute all currently queued callbacks. It
        // processes all pending ROS callbacks on this node, including /clock topic.
        rclcpp::spin_some(this->ros_node);
    }

    static std::chrono::steady_clock::duration lastSimTime{0};
    auto simTime = info.simTime;
    double dt = std::chrono::duration<double>(simTime - lastSimTime).count();
    if (lastSimTime == std::chrono::steady_clock::duration{0}) {
        // Force dt to be 0.0 for the very first iteration.
        dt = 0.0;
    }
    lastSimTime = simTime;

    // Read ground truth pose. Use parent entity of model_entity. TODO: figure this shit out
    if (auto pose_model =
            ecm.Component<gz::sim::components::Pose>(ecm.ParentEntity(this->model_entity))) {
        std::lock_guard<std::mutex> lock(this->state_mutex);
        this->gt_pose = pose_model->Data();
    } else {
        throw std::runtime_error("No pose found on model_entity");
    }
    // Read ground truth linear velocity
    if (auto linvel_comp =
            ecm.Component<gz::sim::components::LinearVelocity>(this->baselink_entity)) {
        this->gt_linear_velocity = linvel_comp->Data();
    } else {
        throw std::runtime_error("No linear velocity found on baselink_entity");
    }
    // Read ground truth angular velocity
    if (auto angvel_comp =
            ecm.Component<gz::sim::components::AngularVelocity>(this->baselink_entity)) {
        this->gt_angular_velocity = angvel_comp->Data();
    } else {
        throw std::runtime_error("No angular velocity found on model or link");
    }

    // Optionally zero dynamics when requested.
    if (this->pending_reset_dynamics.load(std::memory_order_relaxed)) {
        if (auto linvel_comp =
                ecm.Component<gz::sim::components::LinearVelocity>(this->baselink_entity)) {
            linvel_comp->Data().Set(0.0, 0.0, 0.0);
            this->gt_linear_velocity.Set(0.0, 0.0, 0.0);
        }
        if (auto angvel_comp =
                ecm.Component<gz::sim::components::AngularVelocity>(this->baselink_entity)) {
            angvel_comp->Data().Set(0.0, 0.0, 0.0);
            this->gt_angular_velocity.Set(0.0, 0.0, 0.0);
        }
        this->pending_reset_dynamics.store(false, std::memory_order_relaxed);
    }

    // Publish ground truth pose. should i publish before or after updating dynamics?
    // geometry_msgs::msg::Pose gt_pose;
    // gt_pose.position.x = this->gt_pose.Pos().X();
    // gt_pose.position.y = this->gt_pose.Pos().Y();
    // gt_pose.position.z = this->gt_pose.Pos().Z();
    // gt_pose.orientation.w = this->gt_pose.Rot().W();
    // gt_pose.orientation.x = this->gt_pose.Rot().X();
    // gt_pose.orientation.y = this->gt_pose.Rot().Y();
    // gt_pose.orientation.z = this->gt_pose.Rot().Z();
    // this->pub_gt_pose->publish(gt_pose);

    // Forward latest ROS motor speeds to the MulticopterMotorModel
    if (this->motor_pub && this->ros_motor_cmd_available) {
        gz::msgs::Actuators msg;
        for (double w : this->ros_motor_speeds) {
            msg.add_velocity(w);
        }
        this->motor_pub.Publish(msg);
    }
}

/**
 * PostUpdate: called every simulation iteration after physics integration.
 * It is called when the simulation is paused or unpaused.
 */
void LionQuadcopter::PostUpdate(const gz::sim::UpdateInfo &info,
                                const gz::sim::EntityComponentManager &ecm) {
    (void)info;
    // Is reading from ecm time-consuming?
    // Read ground truth pose. Use parent entity of model_entity. TODO: figure this shit out
    if (auto pose_model =
            ecm.Component<gz::sim::components::Pose>(ecm.ParentEntity(this->model_entity))) {
        std::lock_guard<std::mutex> lock(this->state_mutex);
        this->gt_pose = pose_model->Data();
    } else {
        throw std::runtime_error("No pose found on model_entity");
    }
    // Read ground truth linear velocity
    if (auto linvel_comp =
            ecm.Component<gz::sim::components::LinearVelocity>(this->baselink_entity)) {
        this->gt_linear_velocity = linvel_comp->Data();
    } else {
        throw std::runtime_error("No linear velocity found on baselink_entity");
    }
    // Read ground truth angular velocity
    if (auto angvel_comp =
            ecm.Component<gz::sim::components::AngularVelocity>(this->baselink_entity)) {
        this->gt_angular_velocity = angvel_comp->Data();
    } else {
        throw std::runtime_error("No angular velocity found on model or link");
    }

    // Construct ground truth odometry
    nav_msgs::msg::Odometry post_odom;
    post_odom.header.stamp = this->ros_node->now();
    post_odom.header.frame_id = std::string(this->ros_node->get_namespace()) + "/odom";
    post_odom.child_frame_id = std::string(this->ros_node->get_namespace()) + "/base_footprint";
    // Copy pose from gz::math::Pose3d
    post_odom.pose.pose.position.x = this->gt_pose.Pos().X();
    post_odom.pose.pose.position.y = this->gt_pose.Pos().Y();
    post_odom.pose.pose.position.z = this->gt_pose.Pos().Z();
    post_odom.pose.pose.orientation.w = this->gt_pose.Rot().W();
    post_odom.pose.pose.orientation.x = this->gt_pose.Rot().X();
    post_odom.pose.pose.orientation.y = this->gt_pose.Rot().Y();
    post_odom.pose.pose.orientation.z = this->gt_pose.Rot().Z();

    // Copy twist from gz::math::Vector3d
    post_odom.twist.twist.linear.x = this->gt_linear_velocity.X();
    post_odom.twist.twist.linear.y = this->gt_linear_velocity.Y();
    post_odom.twist.twist.linear.z = this->gt_linear_velocity.Z();
    post_odom.twist.twist.angular.x = this->gt_angular_velocity.X();
    post_odom.twist.twist.angular.y = this->gt_angular_velocity.Y();
    post_odom.twist.twist.angular.z = this->gt_angular_velocity.Z();

    // Always publish post-updated odometry
    if (!this->shutting_down.load(std::memory_order_relaxed) && this->pub_gt_odom &&
        this->ros_node && rclcpp::ok()) {
        this->pub_gt_odom->publish(post_odom);
    }
}

void LionQuadcopter::init_ros_subscribers(std::string cmd_odom_topic,
                                          std::string motor_speed_topic) {
    assert(!cmd_odom_topic.empty());

    this->cmd_odom_subscriber = this->ros_node->create_subscription<nav_msgs::msg::Odometry>(
        cmd_odom_topic, rclcpp::QoS(rclcpp::KeepLast(32)).reliable(),
        std::bind(&LionQuadcopter::cb_cmd_odom, this, std::placeholders::_1));

    this->ros_motor_subscriber =
        this->ros_node->create_subscription<std_msgs::msg::Float32MultiArray>(
            motor_speed_topic, rclcpp::QoS(rclcpp::KeepLast(32)).reliable(),
            std::bind(&LionQuadcopter::cb_ros_motor_cmd, this, std::placeholders::_1));
}

void LionQuadcopter::init_ros_publishers(std::string gt_odom_topic, std::string camera_image_topic,
                                         std::string camera_info_topic,
                                         std::string camera_pose_topic) {
    assert(!gt_odom_topic.empty() && !camera_image_topic.empty() && !camera_info_topic.empty() &&
           !camera_pose_topic.empty());

    this->pub_camera_info = this->ros_node->create_publisher<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    this->pub_gt_odom = this->ros_node->create_publisher<nav_msgs::msg::Odometry>(
        gt_odom_topic, rclcpp::QoS(rclcpp::KeepLast(32)).best_effort());
    this->pub_camera_image = this->ros_node->create_publisher<sensor_msgs::msg::Image>(
        camera_image_topic, rclcpp::QoS(rclcpp::KeepLast(32)).best_effort());
    this->pub_camera_pose = this->ros_node->create_publisher<geometry_msgs::msg::PoseStamped>(
        camera_pose_topic, rclcpp::QoS(rclcpp::KeepLast(32)).best_effort());
}

void LionQuadcopter::init_gz_subscribers(std::string image_topic, std::string camera_info_topic) {
    assert(!image_topic.empty() && !camera_info_topic.empty());
    // Gazebo uses its own transport system (gz::transport) with gz::msgs::Image.
    this->gz_image_topic = image_topic;
    this->gz_camera_info_topic = camera_info_topic;
    this->gz_node.Subscribe(this->gz_image_topic, &LionQuadcopter::repub_ros_image, this);
    this->gz_node.Subscribe(this->gz_camera_info_topic, &LionQuadcopter::repub_ros_camera_info,
                            this);
}

void LionQuadcopter::cb_cmd_odom(const nav_msgs::msg::Odometry::SharedPtr cmd) {
    // Copy desired pose
    this->cmd_pose = cmd->pose.pose;
    // Copy desired twist
    this->cmd_twist = cmd->twist.twist;
}

void LionQuadcopter::cb_ros_motor_cmd(const std_msgs::msg::Float32MultiArray::SharedPtr cmd) {
    if (cmd->data.size() >= 4) {
        this->ros_motor_speeds[0] = static_cast<double>(cmd->data[0]);
        this->ros_motor_speeds[1] = static_cast<double>(cmd->data[1]);
        this->ros_motor_speeds[2] = static_cast<double>(cmd->data[2]);
        this->ros_motor_speeds[3] = static_cast<double>(cmd->data[3]);
        this->ros_motor_cmd_available = true;
    }
}

/**
 * It converts the gz::msgs::Image to sensor_msgs::Image and gets the camera pose. Then republishes
 * the image and the camera pose.
 */
void LionQuadcopter::repub_ros_image(const gz::msgs::Image &gz_img) {
    if (this->shutting_down.load(std::memory_order_relaxed))
        return;
    // Increment inflight_callbacks
    this->inflight_callbacks.fetch_add(1, std::memory_order_relaxed);
    // Decrement inflight_callbacks on exit
    auto on_exit = [&]() { this->inflight_callbacks.fetch_sub(1, std::memory_order_relaxed); };
    // When Guard is destroyed, on_exit is called and inflight_callbacks is decremented.
    struct Guard {
        std::function<void()> f;
        ~Guard() {
            if (f)
                f();
        }
    } guard{on_exit};
    if (!this->ros_node || !rclcpp::ok() || !this->pub_camera_image || !this->pub_camera_pose)
        return;

    // Convert gz::msgs::Image to sensor_msgs::Image
    sensor_msgs::msg::Image ros_img;
    ros_img.header.stamp = ros_node->now();
    std::string ns = this->ros_node->get_namespace();
    std::string camera_frame = ns + "/downward_camera_frame";
    std::string base_frame = ns + "/base_footprint";
    std::string odom_frame = ns + "/odom";
    ros_img.header.frame_id = camera_frame;
    ros_img.height = gz_img.height();
    ros_img.width = gz_img.width();
    ros_img.encoding = sensor_msgs::image_encodings::RGB8; // assuming R8G8B8
    ros_img.step = gz_img.step();                          // bytes per row
    // Copy image data buffer
    ros_img.data.assign(gz_img.data().begin(), gz_img.data().end());
    this->pub_camera_image->publish(ros_img);

    // Compute camera world pose using cached drone pose and fixed base->camera transform. This
    // approach may be inaccurate because gt_pose is not synchronized with the image. But the
    // updating frequency of gt_pose (1000 HZ) is much higher than the image's updating frequency.
    gz::math::Pose3d current_gt_pose;
    {
        std::lock_guard<std::mutex> lock(this->state_mutex);
        current_gt_pose = this->gt_pose;
    }
    gz::math::Pose3d cam_world_pose = current_gt_pose * this->transform_b2c;

    // Publish camera pose as PoseStamped in odom frame
    geometry_msgs::msg::PoseStamped cam_pose_msg;
    cam_pose_msg.header.stamp = ros_img.header.stamp;
    cam_pose_msg.header.frame_id = odom_frame;
    cam_pose_msg.pose.position.x = cam_world_pose.Pos().X();
    cam_pose_msg.pose.position.y = cam_world_pose.Pos().Y();
    cam_pose_msg.pose.position.z = cam_world_pose.Pos().Z();
    cam_pose_msg.pose.orientation.w = cam_world_pose.Rot().W();
    cam_pose_msg.pose.orientation.x = cam_world_pose.Rot().X();
    cam_pose_msg.pose.orientation.y = cam_world_pose.Rot().Y();
    cam_pose_msg.pose.orientation.z = cam_world_pose.Rot().Z();
    this->pub_camera_pose->publish(cam_pose_msg);
}

void LionQuadcopter::repub_ros_camera_info(const gz::msgs::CameraInfo &gzInfo) {
    if (this->shutting_down.load(std::memory_order_relaxed))
        return;
    this->inflight_callbacks.fetch_add(1, std::memory_order_relaxed);
    auto on_exit = [&]() { this->inflight_callbacks.fetch_sub(1, std::memory_order_relaxed); };
    struct Guard {
        std::function<void()> f;
        ~Guard() {
            if (f)
                f();
        }
    } guard{on_exit};
    if (!this->ros_node || !rclcpp::ok() || !this->pub_camera_info)
        return;

    // CameraInfo
    // gzInfo: header {
    //   stamp {
    //   }
    //   data {
    //     key: "frame_id"
    //     value: "camera_model::camera_link::downward_camera"
    //   }
    // }
    // width: 640
    // height: 480
    // distortion {
    //   k: 0
    //   k: 0
    //   k: 0
    //   k: 0
    //   k: 0
    // }
    // intrinsics {
    //   k: 467.74272918701172
    //   k: 0
    //   k: 320
    //   k: 0
    //   k: 467.74269104003906
    //   k: 240
    //   k: 0
    //   k: 0
    //   k: 1
    // }
    // projection {
    //   p: 467.74272918701172
    //   p: 0
    //   p: 320
    //   p: 0
    //   p: 0
    //   p: 467.74269104003906
    //   p: 240
    //   p: 0
    //   p: 0
    //   p: 0
    //   p: 1
    //   p: 0
    // }

    // Create ROS CameraInfo message
    sensor_msgs::msg::CameraInfo ros_info;

    // Set header
    ros_info.header.stamp = ros_node->now();
    std::string ns = this->ros_node->get_namespace();
    ros_info.header.frame_id = ns + "/downward_camera_frame";

    // Set image dimensions
    ros_info.width = gzInfo.width();
    ros_info.height = gzInfo.height();

    // Extract intrinsic matrix (3x3) from Gazebo intrinsics
    // Gazebo stores intrinsics as a flat array: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    assert(gzInfo.intrinsics().k_size() == 9);
    ros_info.k[0] = gzInfo.intrinsics().k(0); // fx
    ros_info.k[1] = gzInfo.intrinsics().k(1); // 0
    ros_info.k[2] = gzInfo.intrinsics().k(2); // cx
    ros_info.k[3] = gzInfo.intrinsics().k(3); // 0
    ros_info.k[4] = gzInfo.intrinsics().k(4); // fy
    ros_info.k[5] = gzInfo.intrinsics().k(5); // cy
    ros_info.k[6] = gzInfo.intrinsics().k(6); // 0
    ros_info.k[7] = gzInfo.intrinsics().k(7); // 0
    ros_info.k[8] = gzInfo.intrinsics().k(8); // 1

    // Extract distortion parameters
    // Gazebo stores distortion as [k1, k2, p1, p2, k3]
    assert(gzInfo.distortion().k_size() == 5);
    ros_info.distortion_model = "plumb_bob"; // Standard distortion model
    ros_info.d.resize(5);
    ros_info.d[0] = gzInfo.distortion().k(0); // k1
    ros_info.d[1] = gzInfo.distortion().k(1); // k2
    ros_info.d[2] = gzInfo.distortion().k(2); // p1
    ros_info.d[3] = gzInfo.distortion().k(3); // p2
    ros_info.d[4] = gzInfo.distortion().k(4); // k3

    // Extract projection matrix (3x4) from Gazebo projection
    // Gazebo stores projection as a flat array: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
    assert(gzInfo.projection().p_size() == 12);
    ros_info.p[0] = gzInfo.projection().p(0);   // fx
    ros_info.p[1] = gzInfo.projection().p(1);   // 0
    ros_info.p[2] = gzInfo.projection().p(2);   // cx
    ros_info.p[3] = gzInfo.projection().p(3);   // 0
    ros_info.p[4] = gzInfo.projection().p(4);   // 0
    ros_info.p[5] = gzInfo.projection().p(5);   // fy
    ros_info.p[6] = gzInfo.projection().p(6);   // cy
    ros_info.p[7] = gzInfo.projection().p(7);   // 0
    ros_info.p[8] = gzInfo.projection().p(8);   // 0
    ros_info.p[9] = gzInfo.projection().p(9);   // 0
    ros_info.p[10] = gzInfo.projection().p(10); // 1
    ros_info.p[11] = gzInfo.projection().p(11); // 0

    // Set rectification matrix (identity for unrectified images)
    ros_info.r[0] = 1.0;
    ros_info.r[1] = 0.0;
    ros_info.r[2] = 0.0;
    ros_info.r[3] = 0.0;
    ros_info.r[4] = 1.0;
    ros_info.r[5] = 0.0;
    ros_info.r[6] = 0.0;
    ros_info.r[7] = 0.0;
    ros_info.r[8] = 1.0;

    // Set binning (no binning)
    ros_info.binning_x = 1;
    ros_info.binning_y = 1;

    // Set ROI (full image)
    ros_info.roi.x_offset = 0;
    ros_info.roi.y_offset = 0;
    ros_info.roi.height = ros_info.height;
    ros_info.roi.width = ros_info.width;
    ros_info.roi.do_rectify = false;

    this->pub_camera_info->publish(ros_info);
}

void LionQuadcopter::load_drone_config(const std::shared_ptr<const sdf::Element> &sdf,
                                       gz::sim::EntityComponentManager &ecm) {

    // Mass of the drone
    double mass;
    // Moment of inertia matrix about body axes
    gz::math::Matrix3d inertia;

    // Store 4 rotor poses. rotor_0, rotor_1, rotor_2, rotor_3.
    std::vector<gz::math::Pose3d> rotor_poses(4);

    // the drone body(base_link) is placed at (0, 0, 0.053302) in the world frame.
    // so if you read the world pose of 4 rotors, you should be able to calculate the length of
    // lever arm.

    // it's fine to use the body (base_link) mass and inertia. it's way larger than the rotors.

    // check sdf. if invalid, throw error. otherwise, clone it.
    if (!sdf) {
        RCLCPP_ERROR(this->ros_node->get_logger(), "No SDF provided to drone config");
        throw std::runtime_error("No SDF provided to drone config");
    }

    // Get model from model_entity. model.Name(ecm)=="X3".
    gz::sim::Model model(this->model_entity);
    // Get all linked entities from the model. model_linked_entities.size()==5
    std::vector<gz::sim::Entity> model_linked_entities = model.Links(ecm);

    // Iterate all linked entities. Expect 1 base_link and 4 rotors.
    for (const auto &ent_id : model_linked_entities) {

        // scoped_name: lion_quadcopter/X3/X3/{base_link,rotor_0,rotor_1,rotor_2,rotor_3}
        const std::string scoped_name = gz::sim::scopedName(ent_id, ecm, "/", false);

        // Base link: capture inertial and remember the link entity for fast state access
        if (scoped_name.find("/base_link") != std::string::npos) {
            this->baselink_entity = ent_id;
            if (auto inertial = ecm.Component<gz::sim::components::Inertial>(ent_id)) {
                mass = inertial->Data().MassMatrix().Mass();   // 1.5
                inertia = inertial->Data().MassMatrix().Moi(); // 0.0347563 0 0 0 0.07 0 0 0 0.0977
            } else {
                throw std::runtime_error("No inertial found on base_link");
            }
            continue;
        }

        auto assign_rotor_info = [&](int idx) {
            if (auto poseComp = ecm.Component<gz::sim::components::Pose>(ent_id)) {
                rotor_poses[idx] = poseComp->Data();
            } else {
                throw std::runtime_error("Rotor " + std::to_string(idx) +
                                         " missing Pose component");
            }
            // TODO: Figure out how to get the world pose later
        };

        if (scoped_name.find("/rotor_0") != std::string::npos) {
            assign_rotor_info(0);
        } else if (scoped_name.find("/rotor_1") != std::string::npos) {
            assign_rotor_info(1);
        } else if (scoped_name.find("/rotor_2") != std::string::npos) {
            assign_rotor_info(2);
        } else if (scoped_name.find("/rotor_3") != std::string::npos) {
            assign_rotor_info(3);
        }
    }

    // Calculate rotor geometry and lever arm
    // In the drone's body frame, the rotors are assigned as:
    // rotor_0: x+, y-, z+
    // rotor_1: x-, y+, z+
    // rotor_2: x+, y+, z+
    // rotor_3: x-, y-, z+
    // Notice that rotor_0 and rotor_1 are opposite to each other and rotor_2 and rotor_3 are
    // opposite to each other. It is consistent with turningDirection in the SDF file.
    // Do not use front/back/left/right to indicate which rotor is which.
    // The drone's body frame x-axis is pointing toward between rotor_0 and rotor_2.
    // The drone's body frame z-axis is pointing upward. Four rotors' z-axis are all positive.
    assert(rotor_poses[0].Pos().X() > 0);
    assert(rotor_poses[0].Pos().Y() < 0);
    assert(rotor_poses[1].Pos().X() < 0);
    assert(rotor_poses[1].Pos().Y() > 0);
    assert(rotor_poses[2].Pos().X() > 0);
    assert(rotor_poses[2].Pos().Y() > 0);
    assert(rotor_poses[3].Pos().X() < 0);
    assert(rotor_poses[3].Pos().Y() < 0);

    assert(std::abs(mass - 1.5) < 1e-6); // 1.5 kg is the mass of the drone
}

} // namespace sdrl

GZ_ADD_PLUGIN(sdrl::LionQuadcopter, gz::sim::System, sdrl::LionQuadcopter::ISystemConfigure,
              sdrl::LionQuadcopter::ISystemPreUpdate, sdrl::LionQuadcopter::ISystemPostUpdate)