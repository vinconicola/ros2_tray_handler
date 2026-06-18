#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/empty.hpp>
#include <jaka_msgs/srv/set_io.hpp>
#include <jaka_msgs/srv/set_payload.hpp>
#include <jaka_msgs/srv/set_collision.hpp>
#include <jaka_msgs/msg/robot_msg.hpp>
#include <jaka_msgs/srv/clear_error.hpp>


using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct Rack {
    int id;
    std::string frame;
    Eigen::Vector3d position;
    geometry_msgs::msg::TransformStamped transform;
    std::array<bool, 10> slots = {};
};

enum class RobotOperation { IDLE, PICKING, PLACING, TRANSIT };

struct RobotState {
    RobotOperation operation  = RobotOperation::IDLE;
    int            rack_index = -1;   // index into current_racks_
    int            slot_index = -1;
    bool           holding_tray = false;
};

// ---------------------------------------------------------------------------
// TrayHandler
// ---------------------------------------------------------------------------

class TrayHandler : public rclcpp::Node {
public:
    TrayHandler() : Node("tray_handler") {
        this->set_parameter(rclcpp::Parameter("use_sim_time"));
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        base_config_     = {1.57080, 2.513, -2.471, -0.541, -1.5708, 0.7854};
        approach_config_ = {1.5708, 1.937, -2.758, 0.820, -1.5708, 0.7854};
        gripper_closed_  = {0.015};
        gripper_open_    = {0.065};
        tray_slot_base_height_ = 0.15;
        tray_slot_offset_      = 0.11;
    }

    const geometry_msgs::msg::Pose BOX_POSE = []() {
        geometry_msgs::msg::Pose p;
        p.position.x    =  -0.08;
        p.position.y    =  1.15;
        p.position.z    =  0.73;
        p.orientation.x =  0.0;
        p.orientation.y =  0.0;
        p.orientation.z =  -0.7071068;
        p.orientation.w =  0.7071068;
        return p;
    }();

    // -----------------------------------------------------------------------
    void init() {
        arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "jaka_s12");

        if (use_real_gripper_) {
            io_client_        = this->create_client<jaka_msgs::srv::SetIO>("/jaka_driver/set_io");
            payload_client_   = this->create_client<jaka_msgs::srv::SetPayload>("/jaka_driver/set_payload");
            collision_client_ = this->create_client<jaka_msgs::srv::SetCollision>("/jaka_driver/set_collision_level");
        } else {
            gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "gripper");
        }

        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);

        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_vel",  1.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_acc",  2.25);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_dec", -5.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_rot_vel",    1.57);

        trigger_client_ = this->create_client<std_srvs::srv::Trigger>("/yolo_inference/trigger_detection");

        collision_client_ = this->create_client<jaka_msgs::srv::SetCollision>("/jaka_driver/set_collision_level");
        collision_recover_client_ = this->create_client<std_srvs::srv::Empty>("/jaka_driver/collision_recover");
        clear_error_client_       = this->create_client<std_srvs::srv::Empty>("/jaka_driver/clear_error");

        // Watchdog: subscribe to robot states published by jaka_driver at 10 Hz
        robot_state_sub_ = this->create_subscription<jaka_msgs::msg::RobotMsg>(
            "/jaka_driver/robot_states", 10,
            [this](const jaka_msgs::msg::RobotMsg::SharedPtr msg) {
                if (msg->collision_state == 1 && !collision_detected_.load()) {
                    RCLCPP_WARN(this->get_logger(), "COLLISION DETECTED via watchdog!");
                    collision_detected_.store(true);
                    collision_cv_.notify_all();
                }
            });
    }

    // -----------------------------------------------------------------------
    // Step executor — runs a callable and monitors for collision.
    // Returns true on success, false if collision occurred and recovery failed.
    // -----------------------------------------------------------------------
    bool execute_step(
        const std::string&         description,
        RobotOperation             operation,
        int                        rack_index,
        int                        slot_index,
        std::function<bool()>      step_fn)
    {
        // Save state before starting
        robot_state_.operation   = operation;
        robot_state_.rack_index  = rack_index;
        robot_state_.slot_index  = slot_index;

        RCLCPP_INFO(this->get_logger(), "Step: %s", description.c_str());

        // Clear any stale collision flag before starting this step
        collision_detected_.store(false);

        bool step_result = step_fn();

        // Check collision flag after the step completes
        // (the watchdog may have set it during execution)
        if (collision_detected_.load()) {
            RCLCPP_WARN(this->get_logger(),
                "Collision flagged during step: %s — starting recovery", description.c_str());

            bool recovered = handle_collision_recovery();
            if (!recovered) {
                RCLCPP_ERROR(this->get_logger(), "Recovery failed, stopping execution");
                return false;
            }

            // Retry the step once after recovery
            RCLCPP_INFO(this->get_logger(), "Retrying step after recovery: %s", description.c_str());
            collision_detected_.store(false);
            step_result = step_fn();

            if (collision_detected_.load()) {
                RCLCPP_ERROR(this->get_logger(),
                    "Second collision on retry of step: %s — waiting for operator", description.c_str());
                wait_for_operator_intervention();
                return false;
            }
        }

        return step_result;
    }

    // -----------------------------------------------------------------------
    // Recovery handler
    // -----------------------------------------------------------------------
    bool handle_collision_recovery() {
        RCLCPP_WARN(this->get_logger(), "--- RECOVERY START ---");
        RCLCPP_WARN(this->get_logger(),
            "State: op=%d rack=%d slot=%d holding=%s",
            (int)robot_state_.operation,
            robot_state_.rack_index,
            robot_state_.slot_index,
            robot_state_.holding_tray ? "YES" : "NO");

        // 1. Abort motion
        arm_->stop();
        rclcpp::sleep_for(500ms);

        // 2. Recover from collision state via JAKA driver services
        if (!call_collision_recover()) {
            RCLCPP_ERROR(this->get_logger(), "collision_recover service failed");
            return false;
        }
        rclcpp::sleep_for(500ms);

        if (!call_clear_error()) {
            RCLCPP_ERROR(this->get_logger(), "clear_error service failed");
            return false;
        }
        
        rclcpp::sleep_for(1s);

        if (robot_state_.operation == RobotOperation::PICKING){
            robot_state_.holding_tray = false;
            was_holding_tray_at_collision_ = false;
        }
        // Re-apply known-good settings
        set_collision_level(3);
        if (robot_state_.holding_tray) {
            set_robot_payload(4, 0.0, 0.0, 243.0);
        } else {
            set_robot_payload(1.77, 0.0, 0.0, 65.0);
        }

        // 3. Go to safe approach config
        RCLCPP_INFO(this->get_logger(), "Recovery: moving to approach config");
        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.2);
        arm_->setMaxAccelerationScalingFactor(0.2);
        std::vector<double> recovery_state = approach_config_;
        recovery_state[0] = arm_->getCurrentJointValues()[0];
        arm_->setJointValueTarget(recovery_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Recovery: failed to reach approach config");
            return false;
        }

        // 4. If holding a tray — place it on the box first
        if (robot_state_.holding_tray) {
            RCLCPP_INFO(this->get_logger(), "Recovery: placing tray on box");
            if (!place_on_box(BOX_POSE)) {
                RCLCPP_ERROR(this->get_logger(), "Recovery: place_on_box failed");
                return false;
            }
            robot_state_.holding_tray = false;
        }

        // 5. Re-scan known racks (refine only — we already know where they are)
        RCLCPP_INFO(this->get_logger(), "Recovery: refining rack positions");
        if (current_racks_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Recovery: no known racks to refine");
            return false;
        }
        current_racks_ = refine_racks(current_racks_);

        // 6. If we had a tray on the box — pick it back up

        // Simpler: just check if the operation was PICKING or PLACING — those are the
        // operations that require the tray to be in hand for the retry.
        if (was_holding_tray_at_collision_) {
            RCLCPP_INFO(this->get_logger(), "Recovery: picking tray back from box");
            if (!pick_from_box(BOX_POSE)) {
                RCLCPP_ERROR(this->get_logger(), "Recovery: pick_from_box failed");
                return false;
            }
            robot_state_.holding_tray = true;
        }

        RCLCPP_INFO(this->get_logger(), "--- RECOVERY COMPLETE ---");
        collision_detected_.store(false);
        return true;
    }

    // -----------------------------------------------------------------------
    // Operator intervention wait — blocks until collision flag is manually cleared
    // -----------------------------------------------------------------------
    void wait_for_operator_intervention() {
        RCLCPP_ERROR(this->get_logger(),
            "=== OPERATOR INTERVENTION REQUIRED ===");
        RCLCPP_ERROR(this->get_logger(),
            "Robot stopped after two consecutive collisions.");
        RCLCPP_ERROR(this->get_logger(),
            "Please clear the fault on the pendant and restart the program.");
        // Block forever — operator must restart the node
        while (rclcpp::ok()) {
            rclcpp::sleep_for(1s);
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    void useOMPL() {
        arm_->setPlanningPipelineId("ompl");
        arm_->setPlannerId("RRTConnectkConfigDefault");
        arm_->setGoalPositionTolerance(0.005);
        arm_->setGoalOrientationTolerance(0.01);
        arm_->setPlanningTime(5.0);
    }

    void usePILZ(const std::string& planner = "PTP") {
        arm_->setPlanningPipelineId("pilz_industrial_motion_planner");
        arm_->setPlannerId(planner);
        arm_->setGoalPositionTolerance(0.005);
        arm_->setGoalOrientationTolerance(0.05);
        arm_->setPlanningTime(5.0);
    }

    void request_yolo_scan() {
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        last_scan_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Requesting YOLO scan...");
        while (!trigger_client_->wait_for_service(1s)) {
            RCLCPP_INFO(this->get_logger(), "Waiting for YOLO service...");
        }
        trigger_client_->async_send_request(request);
        rclcpp::sleep_for(1s);
        RCLCPP_INFO(this->get_logger(), "YOLO scan wait complete");
    }

    void setHeightConstraint(double max_z) {
        moveit_msgs::msg::Constraints constraints;
        moveit_msgs::msg::PositionConstraint pcm;
        pcm.header.frame_id = "world";
        pcm.link_name = arm_->getEndEffectorLink();

        shape_msgs::msg::SolidPrimitive box;
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        box.dimensions.resize(3);
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 10.0;
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 2.6;
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = max_z;

        geometry_msgs::msg::Pose box_pose;
        box_pose.position.x = 0.0;
        box_pose.position.y = 0.0;
        box_pose.position.z = max_z / 2.0;
        box_pose.orientation.w = 1.0;

        pcm.constraint_region.primitives.push_back(box);
        pcm.constraint_region.primitive_poses.push_back(box_pose);
        pcm.weight = 1.0;
        constraints.position_constraints.push_back(pcm);
        arm_->setPathConstraints(constraints);
    }

    void clearConstraints() { arm_->clearPathConstraints(); }

    void set_gripper(bool close) {
        if (use_real_gripper_) {
            auto req    = std::make_shared<jaka_msgs::srv::SetIO::Request>();
            req->signal = "digital";
            req->type   = 0;
            req->index  = 7;
            req->value  = close ? 1.0f : 0.0f;
            auto future = io_client_->async_send_request(req);
            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (future.wait_for(10ms) != std::future_status::ready) {
                if (std::chrono::steady_clock::now() > timeout) {
                    RCLCPP_ERROR(this->get_logger(), "set_gripper: timed out"); return;
                }
            }
            rclcpp::sleep_for(500ms);
        } else {
            gripper_->setJointValueTarget(close ? gripper_closed_ : gripper_open_);
            gripper_->move();
        }
    }

    void set_robot_payload(float weight, double x = 0, double y = 0, double z = 0) {
        if (!use_real_gripper_) return;
        auto req  = std::make_shared<jaka_msgs::srv::SetPayload::Request>();
        req->mass = weight;
        req->xc   = x; req->yc = y; req->zc = z;
        auto future  = payload_client_->async_send_request(req);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "set_payload timed out!"); return;
            }
        }
        rclcpp::sleep_for(100ms);
    }

    void set_collision_level(int level) {
        if (!use_real_gripper_) return;
        auto req   = std::make_shared<jaka_msgs::srv::SetCollision::Request>();
        req->value = level;
        auto future  = collision_client_->async_send_request(req);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "set_collision_level timed out!"); 
                return;
            }
        }
    }

    bool pick(const Rack& rack, int slot_index) {
        robot_state_.holding_tray = false;
        was_holding_tray_at_collision_ = false;
        return face_rack_slot(rack, slot_index, true, false);
    }

    bool place(const Rack& rack, int slot_index) {
        return face_rack_slot(rack, slot_index, false, true);
    }

    // -----------------------------------------------------------------------
    // Scanning
    // -----------------------------------------------------------------------
    std::vector<Rack> find_racks() {
        RCLCPP_INFO(this->get_logger(), "Starting rack scan...");
        std::vector<Rack> found_racks;

        usePILZ("PTP");
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setMaxVelocityScalingFactor(0.4);

        std::vector<double> current_state = base_config_;
        double step = 0.7;

        for (double angle = 0.9; angle < 1. ; angle += step) {
            current_state[0] = angle + base_config_[0];
            arm_->setJointValueTarget(current_state);
            arm_->move();
            rclcpp::sleep_for(200ms);
            request_yolo_scan();

            for (int i = 0; i < 10; ++i) {
                std::string frame = "rack_" + std::to_string(i);
                if (!frame_exists(frame)) break;
                try {
                    auto tf = tf_buffer_->lookupTransform("world", frame, tf2::TimePointZero);
                    Eigen::Vector3d pos(
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z);
                    if (is_new_rack(pos, found_racks)) {
                        found_racks.push_back({i, frame, pos, tf});
                        RCLCPP_INFO(this->get_logger(), "Found unique rack: %s", frame.c_str());
                    }
                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(), "Rack TF lookup failed: %s", ex.what());
                    break;
                }
            }
        }
        usePILZ("PTP");
        return found_racks;
    }

    void scan_rack_occupancy(Rack& rack) {
        RCLCPP_INFO(this->get_logger(), "Starting rack refine");
        Eigen::Isometry3d rack_eigen = tf2::transformToEigen(rack.transform);
        double rack_x = rack_eigen.translation().x();
        double rack_y = rack_eigen.translation().y();

        rack.slots.fill(false);
        request_yolo_scan();

        for (int t = 0; t < 20; ++t) {
            std::string tray_frame = "tray_" + std::to_string(t);
            if (!frame_exists(tray_frame)) break;
            try {
                auto tf = tf_buffer_->lookupTransform("world", tray_frame, tf2::TimePointZero);
                Eigen::Vector3d tray_pos(
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z);
                double xy_dist = std::hypot(tray_pos.x() - rack_x, tray_pos.y() - rack_y);
                if (xy_dist > 0.3) continue;
                int slot = std::round((tray_pos.z() - tray_slot_base_height_) / tray_slot_offset_);
                if (slot < 0 || slot >= 10) continue;
                rack.slots[slot] = true;
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "Tray TF lookup failed: %s", ex.what());
            }
        }
    }

    std::vector<Rack> refine_racks(const std::vector<Rack>& rough_racks) {
        std::vector<Rack> refined;
        usePILZ("PTP");

        // Sort closest to furthest
        std::vector<Rack> sorted_racks = rough_racks;
        std::sort(sorted_racks.begin(), sorted_racks.end(), [](const Rack& a, const Rack& b) {
            return a.position.head<2>().norm() < b.position.head<2>().norm();
        });

        for (const auto& rack : sorted_racks) {
            double angle = std::atan2(rack.position.y(), rack.position.x());
            std::vector<double> face_state = base_config_;
            face_state[0] = angle;

            arm_->setJointValueTarget(face_state);
            arm_->move();
            rclcpp::sleep_for(200ms);
            request_yolo_scan();

            try {
                auto tf = tf_buffer_->lookupTransform("world", rack.frame, tf2::TimePointZero, 2s);
                Eigen::Vector3d pos(
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z);

                bool duplicate = false;
                for (const auto& r : refined) {
                    if ((pos.head<2>() - r.position.head<2>()).norm() < 0.7) {
                        duplicate = true; break;
                    }
                }
                if (!duplicate) {
                    Rack refined_rack = {rack.id, rack.frame, pos, tf};
                    scan_rack_occupancy(refined_rack);
                    refined.push_back(refined_rack);
                }
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "Refine TF lookup failed: %s", ex.what());
                refined.push_back(rack);
            }
        }

        std::vector<double> approach_state = approach_config_;
        approach_state[0] = arm_->getCurrentJointValues()[0];
        arm_->setJointValueTarget(approach_state);
        arm_->move();
        usePILZ("PTP");
        return refined;
    }

    // -----------------------------------------------------------------------
    // face_rack_slot — pick or place at a rack slot
    // -----------------------------------------------------------------------
    bool face_rack_slot(const Rack& rack, int slot_index, bool pick = false, bool place = false) {
        double angle = std::atan2(rack.position.y(), rack.position.x());
        std::vector<double> approach_state = approach_config_;
        approach_state[0] = angle;

        arm_->setJointValueTarget(approach_state);
        arm_->move();

        Eigen::Isometry3d rack_eigen  = tf2::transformToEigen(rack.transform);
        Eigen::Vector3d   rack_normal = rack_eigen.rotation().col(0).normalized();

        float pick_place_offset = 0;
        if (pick)       pick_place_offset = -0.005;
        else if (place) pick_place_offset =  0.03;

        Eigen::Vector3d slot_pos_w(
            rack_eigen.translation().x() + rack_normal.x() * 0.055,
            rack_eigen.translation().y() + rack_normal.y() * 0.055,
            tray_slot_base_height_ + (slot_index * tray_slot_offset_) + pick_place_offset);

        Eigen::Vector3d z_axis   = -rack_normal;
        Eigen::Vector3d world_up(0, 0, 1);
        Eigen::Vector3d x_axis   = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis   = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = z_axis;
        Eigen::Quaterniond target_quat(rot_mat);

        geometry_msgs::msg::Pose target;
        target.position    = tf2::toMsg(slot_pos_w);
        target.orientation = tf2::toMsg(target_quat);

        geometry_msgs::msg::Pose approach;
        approach.orientation = tf2::toMsg(target_quat);

        bool approach_reached = false;
        setHeightConstraint(1.2);
        for (double dist : {0.70, 0.70, 0.65}) {
            Eigen::Vector3d approach_pos_w = slot_pos_w + (rack_normal * dist);
            approach.position = tf2::toMsg(approach_pos_w);

            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.4);
            arm_->setMaxAccelerationScalingFactor(0.4);
            arm_->setPoseTarget(approach);

            if (arm_->move() == moveit::core::MoveItErrorCode::SUCCESS) {
                approach_reached = true;
                break;
            }
            // Check collision between approach attempts
            if (collision_detected_.load()) return false;
        }

        if (!approach_reached) {
            RCLCPP_ERROR(this->get_logger(), "Failed to reach approach at any distance");
            return false;
        }

        if (pick) {
            request_yolo_scan();
            set_gripper(false);
            for (int t = 0; t < 20; ++t) {
                std::string tray_frame = "tray_" + std::to_string(t);
                if (!frame_exists(tray_frame)) {
                    target.position.x -= rack_normal.x() * 0.06;
                    target.position.y -= rack_normal.y() * 0.06;
                    target.position.z -= 0.01;
                    break;
                }
                try {
                    auto tf = tf_buffer_->lookupTransform("world", tray_frame, tf2::TimePointZero);
                    Eigen::Vector3d tray_pos(
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z);

                    double xy_dist = std::hypot(
                        tray_pos.x() - rack_eigen.translation().x(),
                        tray_pos.y() - rack_eigen.translation().y());
                    double z_dist = std::abs(
                        tray_pos.z() - (tray_slot_base_height_ + (slot_index * tray_slot_offset_)));
                    if (xy_dist > 0.15 || z_dist > 0.05) continue;
                    
                    RCLCPP_WARN(this->get_logger(), "Tray detected");

                    Eigen::Isometry3d tray_eigen = tf2::transformToEigen(tf);
                    Eigen::Vector3d tray_normal  = tray_eigen.rotation().col(0).normalized();

                    Eigen::Vector2d delta(
                        tray_pos.x() - target.position.x,
                        tray_pos.y() - target.position.y);
                    Eigen::Vector2d normal_2d(rack_normal.x(), rack_normal.y());
                    Eigen::Vector2d lateral_2d(-rack_normal.y(), rack_normal.x());

                    if (std::abs(delta.dot(normal_2d)) < 0.08 && std::abs(delta.dot(lateral_2d)) < 0.03) {
                        target.position.x = tray_pos.x() - tray_normal.x() * 0.02;
                        target.position.y = tray_pos.y() - tray_normal.y() * 0.02;
                        target.position.z = tray_pos.z() - 0.03;
                    } else{
                        RCLCPP_INFO(this->get_logger(), "tray TF not close enough to the center of the rack");
                        target.position.x -= rack_normal.x() * 0.06;
                        target.position.y -= rack_normal.y() * 0.06;
                        target.position.z -= 0.01;
                    }
                    break;
                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(), "Tray TF lookup failed: %s", ex.what());
                }
            }
        } else if (place) {
            target.position.x -= rack_normal.x() * 0.06;
            target.position.y -= rack_normal.y() * 0.06;
        }

        // Cartesian insertion
        if (place){
            set_collision_level(5);
            arm_->setMaxVelocityScalingFactor(0.2);
            arm_->setMaxAccelerationScalingFactor(0.015);
        } else {
            arm_->setMaxVelocityScalingFactor(0.2);
            arm_->setMaxAccelerationScalingFactor(0.15);
        }
        

        std::vector<geometry_msgs::msg::Pose> waypoints = { approach, target };
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = arm_->computeCartesianPath(waypoints, 0.01, 2.0, trajectory, false);

        RCLCPP_INFO(this->get_logger(), "Cartesian path: %.1f%%", fraction * 100.0);
        if (fraction < 0.96) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian path only %.1f%% complete", fraction * 100.0);
            return false;
        }

        moveit::core::RobotStatePtr state = arm_->getCurrentState(1.0);
        robot_trajectory::RobotTrajectory rt(arm_->getRobotModel(), arm_->getName());
        rt.setRobotTrajectoryMsg(*state, trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        if (place) {
            if (!iptp.computeTimeStamps(rt, 0.2, 0.015)) {
                RCLCPP_ERROR(this->get_logger(), "Time parameterization failed");
                return false;
            }
        } else {
            if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
                RCLCPP_ERROR(this->get_logger(), "Time parameterization failed");
                return false;
            }
        }
        
        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);

        // Check collision immediately after cartesian execution
        if (collision_detected_.load()) return false;

        // Tray interaction
        if (pick) {
            set_collision_level(0);

            geometry_msgs::msg::PoseStamped current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose lift_pose = current_pose.pose;
            lift_pose.position.z += 0.02;

            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.1);
            arm_->setMaxAccelerationScalingFactor(0.1);
            arm_->setPoseTarget(lift_pose);
            arm_->move();

            set_gripper(true);
            set_robot_payload(4, 0.0, 0.0, 243.0);

            // Mark state as holding tray
            robot_state_.holding_tray        = true;
            was_holding_tray_at_collision_   = true;

            geometry_msgs::msg::Pose lift_pose2 = lift_pose;
            lift_pose2.position.z += 0.02;
            arm_->setPoseTarget(lift_pose2);
            arm_->move();

            set_collision_level(3);

        } else if (place) {
            set_collision_level(0);


            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.1);
            arm_->setMaxAccelerationScalingFactor(0.1);
            geometry_msgs::msg::PoseStamped current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose drop_pose = current_pose.pose;
            drop_pose.position.z -= 0.025;
            arm_->setPoseTarget(drop_pose);
            arm_->move();

            set_gripper(false);

            geometry_msgs::msg::Pose retreat_pose = drop_pose;
            retreat_pose.position.z -= 0.02;
            arm_->setPoseTarget(retreat_pose);
            arm_->move();

            set_robot_payload(1.77, 0.0, 0.0, 65.0);

            // Mark state as no longer holding tray
            robot_state_.holding_tray      = false;
            was_holding_tray_at_collision_ = false;

            set_collision_level(3);
        }

        if (collision_detected_.load()) return false;

        // Cartesian retraction
        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose retract_target         = current_stamped.pose;
        geometry_msgs::msg::Pose retract_approach;
        retract_approach.orientation = retract_target.orientation;

        fraction = 0.0;
        for (double dist : {0.70, 0.70, 0.65}) {
            Eigen::Vector3d retract_pos_w = slot_pos_w + (rack_normal * dist);
            if (pick)       retract_pos_w.z() += 0.02;
            else if (place) retract_pos_w.z() -= 0.02;

            retract_approach.position = tf2::toMsg(retract_pos_w);

            std::vector<geometry_msgs::msg::Pose> ret_waypoints = { retract_target, retract_approach };
            fraction = arm_->computeCartesianPath(ret_waypoints, 0.01, 2.0, trajectory, false);

            RCLCPP_INFO(this->get_logger(), "Retraction at %.2fm: %.1f%%", dist, fraction * 100.0);
            if (fraction >= 0.96) break;
        }

        if (fraction < 0.96) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian retraction only %.1f%% complete", fraction * 100.0);
            return false;
        }

        state = arm_->getCurrentState(1);
        rt.setRobotTrajectoryMsg(*state, trajectory);
        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed on retraction");
            return false;
        }
        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);

        if (collision_detected_.load()) return false;

        arm_->setJointValueTarget(approach_state);
        arm_->move();
        return true;
    }

    // -----------------------------------------------------------------------
    // place_on_box / pick_from_box  (unchanged logic, kept intact)
    // -----------------------------------------------------------------------
    bool place_on_box(const geometry_msgs::msg::Pose& box_pose) {
        double angle = std::atan2(box_pose.position.y, box_pose.position.x);
        std::vector<double> face_state = approach_config_;
        face_state[0] = angle;

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        Eigen::Quaterniond box_quat(
            box_pose.orientation.w, box_pose.orientation.x,
            box_pose.orientation.y, box_pose.orientation.z);
        Eigen::Vector3d box_normal = box_quat.toRotationMatrix().col(0).normalized();

        Eigen::Vector3d z_axis  = -box_normal;
        Eigen::Vector3d world_up(0.0, 0.0, 1.0);
        Eigen::Vector3d x_axis  = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis  = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis; rot_mat.col(1) = y_axis; rot_mat.col(2) = z_axis;
        Eigen::Quaterniond target_quat(rot_mat);

        geometry_msgs::msg::Pose target;
        target.position.x  = box_pose.position.x + box_normal.x() * 0.15;
        target.position.y  = box_pose.position.y + box_normal.y() * 0.15;
        target.position.z  = box_pose.position.z + 0.02;
        target.orientation = tf2::toMsg(target_quat);

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.3);
        arm_->setMaxAccelerationScalingFactor(0.2);
        arm_->setPoseTarget(target);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        set_collision_level(0);
        target.position.z = box_pose.position.z;
        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);
        arm_->setPoseTarget(target);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        set_gripper(false);
        set_robot_payload(1.77, 0.0, 0.0, 65.0);

        robot_state_.holding_tray = false;

        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose drop_pose = current_stamped.pose;
        drop_pose.position.z -= 0.01;
        arm_->setPoseTarget(drop_pose);
        arm_->move();

        set_collision_level(1);

        geometry_msgs::msg::Pose retreat_pose = drop_pose;
        retreat_pose.position.x += box_normal.x() * 0.30;
        retreat_pose.position.y += box_normal.y() * 0.30;
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.3);
        arm_->setPoseTarget(retreat_pose);
        arm_->move();

        set_collision_level(3);

        std::vector<double> return_state = approach_config_;
        return_state[0] = face_state[0];
        arm_->setJointValueTarget(return_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        return true;
    }

    bool pick_from_box(const geometry_msgs::msg::Pose& box_pose) {
        double angle = std::atan2(box_pose.position.y, box_pose.position.x);
        std::vector<double> face_approach_state = approach_config_;
        face_approach_state[0] = angle;
        face_approach_state[1] = approach_config_[1] - (10.0 * M_PI / 180.0);

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_approach_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        request_yolo_scan();

        geometry_msgs::msg::TransformStamped best_tray_tf;
        bool tray_found = false;

        for (int t = 0; t < 20; ++t) {
            std::string tray_frame = "tray_" + std::to_string(t);
            if (!frame_exists(tray_frame)) break;
            try {
                auto tf = tf_buffer_->lookupTransform("world", tray_frame, tf2::TimePointZero);
                Eigen::Vector3d tray_pos(
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z);
                double dist = std::hypot(
                    tray_pos.x() - box_pose.position.x,
                    tray_pos.y() - box_pose.position.y);
                if (dist < 0.40) { best_tray_tf = tf; tray_found = true; break; }
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            }
        }

        if (!tray_found) {
            std::vector<double> untilted = approach_config_;
            untilted[0] = angle;
            arm_->setJointValueTarget(untilted);
            arm_->move();
            return false;
        }

        Eigen::Isometry3d tray_eigen  = tf2::transformToEigen(best_tray_tf);
        Eigen::Vector3d   tray_pos    = tray_eigen.translation();
        Eigen::Vector3d   tray_normal = tray_eigen.rotation().col(0).normalized();

        Eigen::Vector3d z_axis  = -tray_normal;
        Eigen::Vector3d world_up(0.0, 0.0, 1.0);
        Eigen::Vector3d x_axis  = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis  = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis; rot_mat.col(1) = y_axis; rot_mat.col(2) = z_axis;
        Eigen::Quaterniond tray_quat(rot_mat);

        geometry_msgs::msg::Pose target;
        target.position.x  = tray_pos.x() - tray_normal.x() * 0.02;
        target.position.y  = tray_pos.y() - tray_normal.y() * 0.02;
        target.position.z  = tray_pos.z() - 0.02;
        target.orientation = tf2::toMsg(tray_quat);

        geometry_msgs::msg::Pose approach;
        approach.position.x  = tray_pos.x() + tray_normal.x() * 0.20;
        approach.position.y  = tray_pos.y() + tray_normal.y() * 0.20;
        approach.position.z  = tray_pos.z() - 0.02;
        approach.orientation = tf2::toMsg(tray_quat);

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.3);
        arm_->setMaxAccelerationScalingFactor(0.3);
        arm_->setPoseTarget(approach);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        set_gripper(false);

        arm_->setMaxVelocityScalingFactor(0.2);
        arm_->setMaxAccelerationScalingFactor(0.15);

        std::vector<geometry_msgs::msg::Pose> waypoints = { approach, target };
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = arm_->computeCartesianPath(waypoints, 0.01, 2.0, trajectory, false);
        if (fraction < 0.98) return false;

        moveit::core::RobotStatePtr state = arm_->getCurrentState(1.0);
        robot_trajectory::RobotTrajectory rt(arm_->getRobotModel(), arm_->getName());
        rt.setRobotTrajectoryMsg(*state, trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) return false;
        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);

        set_collision_level(0);

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);

        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose lift_pose = current_stamped.pose;
        lift_pose.position.z += 0.02;
        arm_->setPoseTarget(lift_pose);
        arm_->move();

        set_gripper(true);
        set_robot_payload(4, 0.0, 0.0, 243.0);

        robot_state_.holding_tray      = true;
        was_holding_tray_at_collision_ = true;

        set_collision_level(3);

        face_approach_state[1] = approach_config_[1];
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_approach_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) return false;

        return true;
    }

    // Expose current_racks_ so main can store the result of find/refine
    std::vector<Rack> current_racks_;

private:
    // -----------------------------------------------------------------------
    // JAKA recovery service calls (direct SDK calls via ROS services)
    // -----------------------------------------------------------------------
    bool call_collision_recover() {
        RCLCPP_INFO(this->get_logger(), "Calling collision_recover");
        if (!collision_recover_client_->wait_for_service(3s)) {
            RCLCPP_ERROR(this->get_logger(), "collision_recover service not available");
            return false;
        }
        auto request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto future  = collision_recover_client_->async_send_request(request);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "collision_recover timed out");
                return false;
            }
        }
        future.get();
        
        return true;
    }

    bool call_clear_error() {
        RCLCPP_INFO(this->get_logger(), "Calling clear_error");
        if (!clear_error_client_->wait_for_service(3s)) {
            RCLCPP_ERROR(this->get_logger(), "clear_error service not available");
            return false;
        }
        auto request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto future  = clear_error_client_->async_send_request(request);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "clear_error timed out");
                return false;
            }
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // TF helpers
    // -----------------------------------------------------------------------
    bool frame_exists(const std::string& frame) const {
        try {
            auto tf = tf_buffer_->lookupTransform("world", frame, tf2::TimePointZero);
            rclcpp::Time tf_time(tf.header.stamp);
            if (last_scan_time_.nanoseconds() == 0) return true;
            return tf_time >= (last_scan_time_ - rclcpp::Duration::from_seconds(0.5));
        } catch (const tf2::TransformException&) { return false; }
    }

    bool is_new_rack(const Eigen::Vector3d& pos, const std::vector<Rack>& known) {
        for (const auto& r : known) {
            if ((pos.head<2>() - r.position.head<2>()).norm() < 0.5) return false;
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    bool use_real_gripper_ = true;

    std::shared_ptr<tf2_ros::Buffer>                                    tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                         tf_listener_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface>     arm_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface>     gripper_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr                   trigger_client_;
    rclcpp::Client<jaka_msgs::srv::SetIO>::SharedPtr                    io_client_;
    rclcpp::Client<jaka_msgs::srv::SetPayload>::SharedPtr               payload_client_;
    rclcpp::Client<jaka_msgs::srv::SetCollision>::SharedPtr             collision_client_;
    rclcpp::Subscription<jaka_msgs::msg::RobotMsg>::SharedPtr           robot_state_sub_;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr                     collision_recover_client_;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr                     clear_error_client_;

    std::vector<double> base_config_;
    std::vector<double> approach_config_;
    std::vector<double> gripper_open_;
    std::vector<double> gripper_closed_;
    double tray_slot_base_height_;
    double tray_slot_offset_;
    rclcpp::Time last_scan_time_{0, 0, RCL_ROS_TIME};

    // Collision watchdog
    std::atomic<bool>       collision_detected_{false};
    std::condition_variable collision_cv_;
    std::mutex              collision_mutex_;

    // Execution state
    RobotState robot_state_;
    bool       was_holding_tray_at_collision_ = false;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrayHandler>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread([&executor]() { executor.spin(); }).detach();

    node->init();
    node->set_collision_level(3);
    node->set_robot_payload(1.77, 0.0, 0.0, 65.0);

    // Initial full scan
    auto rough         = node->find_racks();
    node->current_racks_ = node->refine_racks(rough);

    if (!node->current_racks_.empty()) {
        // Each operation is wrapped in execute_step so collisions are caught
        // and recovery runs automatically before the step is retried.

        node->execute_step(
            "pick rack[0] slot 7",
            RobotOperation::PICKING, 0, 7,
            [&]{ return node->pick(node->current_racks_[0], 7); });

        node->execute_step(
            "place on box",
            RobotOperation::TRANSIT, -1, -1,
            [&]{ return node->place_on_box(node->BOX_POSE); });

        node->execute_step(
            "pick from box",
            RobotOperation::TRANSIT, -1, -1,
            [&]{ return node->pick_from_box(node->BOX_POSE); });

        node->execute_step(
            "place rack[0] slot 8",
            RobotOperation::PLACING, 0, 8,
            [&]{ return node->place(node->current_racks_[0], 8); });
        
        // // load carrello
        // node->execute_step(
        //     "pick from box",
        //     RobotOperation::TRANSIT, -1, -1,
        //     [&]{ return node->pick_from_box(node->BOX_POSE); });
        
        // node->execute_step(
        //     "place rack[0] slot 7",
        //     RobotOperation::PLACING, 0, 7,
        //     [&]{ return node->place(node->current_racks_[0], 7); });
        
        // node->execute_step(
        //     "pick from box",
        //     RobotOperation::TRANSIT, -1, -1,
        //     [&]{ return node->pick_from_box(node->BOX_POSE); });
        
        // node->execute_step(
        //     "place rack[0] slot 8",
        //     RobotOperation::PLACING, 0, 8,
        //     [&]{ return node->place(node->current_racks_[0], 8); });
        
        // node->execute_step(
        //     "pick from box",
        //     RobotOperation::TRANSIT, -1, -1,
        //     [&]{ return node->pick_from_box(node->BOX_POSE); });
        
        // node->execute_step(
        //     "place rack[0] slot 9",
        //     RobotOperation::PLACING, 0, 9,
        //     [&]{ return node->place(node->current_racks_[0], 9); });


        // // swap vassoi
        // node->execute_step(
        //     "pick rack[0] slot 7",
        //     RobotOperation::PICKING, 0, 7,
        //     [&]{ return node->pick(node->current_racks_[0], 7); });

        // node->execute_step(
        //     "place on box",
        //     RobotOperation::TRANSIT, -1, -1,
        //     [&]{ return node->place_on_box(node->BOX_POSE); });

        // node->execute_step(
        //     "pick rack[0] slot 8",
        //     RobotOperation::PICKING, 0, 8,
        //     [&]{ return node->pick(node->current_racks_[0], 8); });

        // node->execute_step(
        //     "place rack[0] slot 7",
        //     RobotOperation::PLACING, 0, 7,
        //     [&]{ return node->place(node->current_racks_[0], 7); });

        // node->execute_step(
        //     "pick from box",
        //     RobotOperation::TRANSIT, -1, -1,
        //     [&]{ return node->pick_from_box(node->BOX_POSE); });

        // node->execute_step(
        //     "place rack[0] slot 8",
        //     RobotOperation::PLACING, 0, 8,
        //     [&]{ return node->place(node->current_racks_[0], 8); });
        
    }

    rclcpp::shutdown();
    return 0;
}