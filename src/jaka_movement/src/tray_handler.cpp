#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <std_srvs/srv/trigger.hpp>
#include <jaka_msgs/srv/set_io.hpp>
#include <jaka_msgs/srv/set_payload.hpp>
#include <jaka_msgs/srv/set_collision.hpp> // <-- Changed from set_collision_level.hpp

using namespace std::chrono_literals;

struct Rack {
    int id;
    std::string frame;
    Eigen::Vector3d position;
    geometry_msgs::msg::TransformStamped transform;
    std::array<bool, 10> slots = {}; // false = empty, true = occupied
};


class TrayHandler : public rclcpp::Node {
public:
    TrayHandler() : Node("tray_handler") {
        this->set_parameter(rclcpp::Parameter("use_sim_time"));
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        base_config_ = {1.57080, 2.513, -2.471, -0.541, -1.5708, 0.7854};
        //approach_config_ = {0.0, 0.716, -2.496, 1.780, -1.5708, 0.7854};
        approach_config_ = {1.5708, 1.937, -2.758, 0.820, -1.5708, 0.7854};
        gripper_closed_ = {0.015};
        gripper_open_ = {0.065};
        tray_slot_base_height_ = 0.15;
        tray_slot_offset_ = 0.11;
        
    }
    const geometry_msgs::msg::Pose BOX_POSE = []() {
        geometry_msgs::msg::Pose p;
        p.position.x    =  -0.08;
        p.position.y    =  1.15;
        p.position.z    =  0.73;
        p.orientation.x =  0.0;
        p.orientation.y =  0.0;
        p.orientation.z =  -0.7071068; //normal along -y
        p.orientation.w =  0.7071068;
        return p;
    }();

    void init() {
        arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "jaka_s12");
        if (use_real_gripper_) {
            io_client_ = this->create_client<jaka_msgs::srv::SetIO>("/jaka_driver/set_io");
            payload_client_ = this->create_client<jaka_msgs::srv::SetPayload>("/jaka_driver/set_payload");
            collision_client_ = this->create_client<jaka_msgs::srv::SetCollision>("/jaka_driver/set_collision_level");
        } else {
            gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "gripper");
        }
        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);

        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_vel", 1.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_acc", 2.25);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_dec", -5.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_rot_vel",   1.57);

        trigger_client_ = this->create_client<std_srvs::srv::Trigger>("/yolo_inference/trigger_detection");

        // joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        //     "/joint_states", 10, 
        //     [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        //         last_joint_state_ = *msg;
        //     });
    }

    // --- helpers ---
    void useOMPL() {
        arm_->setPlanningPipelineId("ompl");
        arm_->setPlannerId("RRTConnectkConfigDefault");
        arm_->setGoalPositionTolerance(0.005);      // 1mm position tolerance
        arm_->setGoalOrientationTolerance(0.01);    // ~0.5° orientation tolerance
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
        
        // Give YOLO a moment to finish processing and TF to update
        rclcpp::sleep_for(2s); 
        RCLCPP_INFO(this->get_logger(), "YOLO scan wait complete");
    }

    void set_gripper(bool close) {
        if (use_real_gripper_) {
            auto req = std::make_shared<jaka_msgs::srv::SetIO::Request>();
            req->signal = "digital";   // check exact string with your driver docs
            req->type   = 0;      // 0 = digital
            req->index  = 7;      // DO8
            req->value  = close ? 1.0f : 0.0f;

            auto future = io_client_->async_send_request(req);

            auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (future.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready) {
                if (std::chrono::steady_clock::now() > timeout) {
                    RCLCPP_ERROR(this->get_logger(), "set_gripper: service call timed out");
                    return;
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
    
        auto req = std::make_shared<jaka_msgs::srv::SetPayload::Request>();
        req->mass = weight;
        req->xc = x;
        req->yc = y;
        req->zc = z;

        auto future = payload_client_->async_send_request(req);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "set_payload timed out!");
                return;
            }
        }
        rclcpp::sleep_for(100ms); // Small delay for controller to update torque models
    }

    void set_collision_level(int level) {
        if (!use_real_gripper_) return;
        
        auto req = std::make_shared<jaka_msgs::srv::SetCollision::Request>();
        req->value = level; // 0 (off) to 5 (most sensitive)

        auto future = collision_client_->async_send_request(req);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (future.wait_for(10ms) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() > timeout) {
                RCLCPP_ERROR(this->get_logger(), "set_collision_level timed out!");
                return;
            }
        }
    }

    // --- 1. SCANNING ---
    std::vector<Rack> find_racks() {
        RCLCPP_INFO(this->get_logger(), "Starting rack scan...");
        std::vector<Rack> found_racks;

        usePILZ("PTP");
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setMaxVelocityScalingFactor(0.4);
        // arm_->setJointValueTarget(base_config_);
        // arm_->move();

        std::vector<double> current_state = base_config_;
        double step = 1.04; //1.04      

        for (double angle = 0.7; angle < 2.1 - step; angle += step) {
            current_state[0] = angle + base_config_[0];
            RCLCPP_INFO(this->get_logger(),
                "Scan iteration angle=%.3f target_joint_1=%.3f",
                angle, current_state[0]);

            RCLCPP_INFO(this->get_logger(), "Starting arm move for scan iteration");
            arm_->setJointValueTarget(current_state);
            arm_->move();
            RCLCPP_INFO(this->get_logger(), "Arm move completed for scan iteration");
            rclcpp::sleep_for(2s);
            RCLCPP_INFO(this->get_logger(), "Post-move settling wait complete");
            request_yolo_scan();
            RCLCPP_INFO(this->get_logger(), "Starting rack TF lookup sweep");

            
            for (int i = 0; i < 10; ++i) {
                std::string frame = "rack_" + std::to_string(i);
                RCLCPP_INFO(this->get_logger(), "Looking up TF for %s", frame.c_str());
                if (!frame_exists(frame)) {
                    RCLCPP_INFO(this->get_logger(),
                        "Frame %s does not exist. Ending this rack sweep.",
                        frame.c_str());
                    break;
                }
                try {
                    auto tf = tf_buffer_->lookupTransform("world", frame, tf2::TimePointZero);
                    Eigen::Vector3d pos(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
                    if (is_new_rack(pos, found_racks)) {
                        found_racks.push_back({i, frame, pos, tf});
                        RCLCPP_INFO(this->get_logger(), "Found unique rack: %s", frame.c_str());
                    } else {
                        RCLCPP_INFO(this->get_logger(), "Rack %s already known", frame.c_str());
                    }
                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(),
                        "Rack TF lookup failed for %s: %s. Ending this rack sweep.",
                        frame.c_str(), ex.what());
                    break;
                }
            }

            RCLCPP_INFO(this->get_logger(),
                "Rack TF lookup sweep complete, %zu unique racks found so far",
                found_racks.size());
        }

        usePILZ("PTP");
        return found_racks;
    }

    void scan_rack_occupancy(Rack& rack) {
        Eigen::Isometry3d rack_eigen = tf2::transformToEigen(rack.transform);
        double rack_x = rack_eigen.translation().x();
        double rack_y = rack_eigen.translation().y();

        rack.slots.fill(false); // reset
        request_yolo_scan();
        RCLCPP_INFO(this->get_logger(), "Starting tray TF lookup sweep for rack %s", rack.frame.c_str());
        for (int t = 0; t < 20; ++t) {
            std::string tray_frame = "tray_" + std::to_string(t);
            RCLCPP_INFO(this->get_logger(), "Looking up TF for %s", tray_frame.c_str());
            if (!frame_exists(tray_frame)) {
                RCLCPP_INFO(this->get_logger(),
                    "Frame %s does not exist. Ending tray sweep for rack %s.",
                    tray_frame.c_str(), rack.frame.c_str());
                break;
            }
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
                RCLCPP_WARN(this->get_logger(),
                    "Tray TF lookup failed for %s: %s",
                    tray_frame.c_str(), ex.what());
                continue;
            }
        }

        RCLCPP_INFO(this->get_logger(), "Tray TF lookup sweep complete for rack %s", rack.frame.c_str());

        for (int i = 0; i < 10; ++i) {
            RCLCPP_INFO(this->get_logger(), "  rack=%s slot[%d]: %s",
                rack.frame.c_str(), i, rack.slots[i] ? "occupied" : "empty");
        }
    }

    std::vector<Rack> refine_racks(const std::vector<Rack>& rough_racks) {
        std::vector<Rack> refined;

        usePILZ("PTP");

        for (const auto& rack : rough_racks) {
            double angle = std::atan2(rack.position.y(), rack.position.x());
            std::vector<double> face_state = base_config_;
            face_state[0] = angle;

            RCLCPP_INFO(this->get_logger(),
                "Refining rack %s with face angle %.3f",
                rack.frame.c_str(), angle);
            arm_->setJointValueTarget(face_state);
            arm_->move();
            RCLCPP_INFO(this->get_logger(), "Refine move completed for rack %s", rack.frame.c_str());
            rclcpp::sleep_for(3s);
            RCLCPP_INFO(this->get_logger(), "Refine post-move settling wait complete for rack %s", rack.frame.c_str());
            request_yolo_scan();

            try {
                auto tf = tf_buffer_->lookupTransform("world", rack.frame, tf2::TimePointZero, 2s);
                Eigen::Vector3d pos(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);

                // Check if this rack is too close to an already refined one
                bool duplicate = false;
                for (const auto& r : refined) {
                    if ((pos.head<2>() - r.position.head<2>()).norm() < 0.7) {
                        RCLCPP_WARN(this->get_logger(),
                            "Rack %s too close to already refined %s, skipping",
                            rack.frame.c_str(), r.frame.c_str());
                        duplicate = true;
                        break;
                    }
                }

                if (!duplicate) {
                    Rack refined_rack = {rack.id, rack.frame, pos, tf};
                    scan_rack_occupancy(refined_rack);
                    refined.push_back(refined_rack);
                }

            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(),
                    "Refine TF lookup failed for %s: %s",
                    rack.frame.c_str(), ex.what());
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


    void setHeightConstraint(double max_z) {
        moveit_msgs::msg::Constraints constraints;
        
        moveit_msgs::msg::PositionConstraint pcm;
        pcm.header.frame_id = "world";
        pcm.link_name = arm_->getEndEffectorLink();

        // Define the shape of the constraint (a Box)
        shape_msgs::msg::SolidPrimitive box;
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        
        // Set dimensions: X and Y are very large (effectively unconstrained)
        // Z is the max_z
        box.dimensions.resize(3);
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = 10.0; 
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 2.6;
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = max_z;

        // Define the Pose of the Box
        // The box is centered at its origin, so to make it go from Z=0 to Z=max_z,
        // we must place the center of the box at Z = max_z / 2
        geometry_msgs::msg::Pose box_pose;
        box_pose.position.x = 0.0;
        box_pose.position.y = 1.3;
        box_pose.position.z = max_z / 2.0; 
        box_pose.orientation.w = 1.0;

        // Add the box and its pose to the constraint
        pcm.constraint_region.primitives.push_back(box);
        pcm.constraint_region.primitive_poses.push_back(box_pose);
        pcm.weight = 1.0;

        constraints.position_constraints.push_back(pcm);
        
        // If you want to keep existing orientation constraints, 
        // you would append them to this 'constraints' object here.
        
        arm_->setPathConstraints(constraints);
    }

    void clearConstraints() {
        arm_->clearPathConstraints();
    }
    
    // --- 4. INSERTION ---
    bool face_rack_slot(const Rack& rack, int slot_index, bool pick=false, bool place=false) {
        double angle = std::atan2(rack.position.y(), rack.position.x());
        std::vector<double> approach_state = approach_config_;
        approach_state[0] = angle;

        arm_->setJointValueTarget(approach_state);
        arm_->move();
        Eigen::Isometry3d rack_eigen = tf2::transformToEigen(rack.transform);
        Eigen::Vector3d rack_normal  = rack_eigen.rotation().col(0).normalized();
        
        float pick_place_offset = 0;
        if(pick){
            pick_place_offset = -0.005;
        } else if(place){
            pick_place_offset = 0.03;
        }

        Eigen::Vector3d slot_pos_w(
            rack_eigen.translation().x() + rack_normal.x()*0.06,
            rack_eigen.translation().y() + rack_normal.y()*0.06,
            tray_slot_base_height_ + (slot_index * tray_slot_offset_) + pick_place_offset);
        
        Eigen::Vector3d z_axis = -rack_normal;
        Eigen::Vector3d world_up(0, 0, 1);
        Eigen::Vector3d x_axis = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = z_axis;
        Eigen::Quaterniond target_quat(rot_mat);

        geometry_msgs::msg::Pose target;
        target.position    = tf2::toMsg(slot_pos_w);
        target.orientation = tf2::toMsg(target_quat);

        // --- Phase 1: PILZ PTP approach, OMPL fallback ---
        bool approach_reached = false;
        geometry_msgs::msg::Pose approach;
        approach.orientation = tf2::toMsg(target_quat);
        setHeightConstraint(1.2);
        for (double dist : {0.70, 0.70, 0.65}) {
            Eigen::Vector3d approach_pos_w = slot_pos_w + (rack_normal * dist);
            approach.position = tf2::toMsg(approach_pos_w);

            RCLCPP_INFO(this->get_logger(),
                "Trying PILZ PTP approach at %.2fm: [%.3f, %.3f, %.3f]",
                dist, approach_pos_w.x(), approach_pos_w.y(), approach_pos_w.z());

            // Try PILZ PTP first — deterministic, shortest joint path
            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.4);
            arm_->setMaxAccelerationScalingFactor(0.4);
            arm_->setPoseTarget(approach);

            if (arm_->move() == moveit::core::MoveItErrorCode::SUCCESS) {
                RCLCPP_INFO(this->get_logger(), "PILZ PTP approach succeeded at %.2fm", dist);
                approach_reached = true;
                break;
            }

            // RCLCPP_WARN(this->get_logger(), "PILZ PTP failed at %.2fm, trying OMPL fallback", dist);

            // // OMPL fallback — seeded from current state to prevent joint flips
            // moveit::core::RobotStatePtr current_state = arm_->getCurrentState(5.0);
            // if (!current_state) {
            //     RCLCPP_ERROR(this->get_logger(), "Failed to get current state");
            //     return false;
            // }
            // arm_->setStartState(*current_state);
            // useOMPL();
            // arm_->setMaxVelocityScalingFactor(0.4);
            // arm_->setMaxAccelerationScalingFactor(0.3);
            // arm_->setPoseTarget(approach);

            // if (arm_->move() == moveit::core::MoveItErrorCode::SUCCESS) {
            //     RCLCPP_INFO(this->get_logger(), "OMPL fallback approach succeeded at %.2fm", dist);
            //     approach_reached = true;
            //     clearConstraints();
            //     break;
            // }

            // RCLCPP_WARN(this->get_logger(), "OMPL fallback also failed at %.2fm, trying closer", dist);
            // clearConstraints();
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
                    target.position.x -= rack_normal.x() * 0.04;
                    target.position.y -= rack_normal.y() * 0.04;
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
                    double z_dist = std::abs(tray_pos.z() - (tray_slot_base_height_ + (slot_index * tray_slot_offset_)));
                    if (xy_dist > 0.15 || z_dist > 0.05) continue;
                    
                    RCLCPP_WARN(this->get_logger(), "Tray detected");

                    Eigen::Isometry3d tray_eigen = tf2::transformToEigen(tf);
                    Eigen::Vector3d tray_normal  = tray_eigen.rotation().col(0).normalized();

                    Eigen::Vector3d z_axis    = -tray_normal;
                    Eigen::Vector3d world_up(0, 0, 1);
                    Eigen::Vector3d x_axis    = world_up.cross(z_axis).normalized();
                    Eigen::Vector3d y_axis    = z_axis.cross(x_axis).normalized();

                    Eigen::Matrix3d tray_rot;
                    tray_rot.col(0) = x_axis;
                    tray_rot.col(1) = y_axis;
                    tray_rot.col(2) = z_axis;
                    Eigen::Quaterniond tray_quat(tray_rot);
                    Eigen::Vector2d delta(
                        tray_pos.x() - target.position.x,
                        tray_pos.y() - target.position.y);

                    Eigen::Vector2d normal_2d(rack_normal.x(), rack_normal.y());
                    Eigen::Vector2d lateral_2d(-rack_normal.y(), rack_normal.x());

                    double depth_dist   = std::abs(delta.dot(normal_2d));
                    double lateral_dist = std::abs(delta.dot(lateral_2d));

                    if (depth_dist < 0.08 && lateral_dist < 0.04){
                        // Update target position and orientation
                        target.position.x = tray_pos.x()- tray_normal.x() * 0.02;
                        target.position.y = tray_pos.y() - tray_normal.y() * 0.02;
                        target.position.z = tray_pos.z() - 0.03;
                        //target.orientation = tf2::toMsg(tray_quat);
                        //retract_approach.orientation = tf2::toMsg(tray_quat);
                    } else{
                        RCLCPP_INFO(this->get_logger(), "tray TF not close enough to the center of the rack");
                        target.position.x -= rack_normal.x() * 0.04;
                        target.position.y -= rack_normal.y() * 0.04;
                        target.position.z -= 0.01;
                    }
                    
                    break;

                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(), "Tray TF lookup failed for %s: %s", tray_frame.c_str(), ex.what());
                    continue;
                }
            }
        } else if(place){
            target.position.x -= rack_normal.x() * 0.06;
            target.position.y -= rack_normal.y() * 0.06;
        }

        // --- Phase 2: Cartesian insertion seeded from current state ---
        arm_->setMaxVelocityScalingFactor(0.2);
        arm_->setMaxAccelerationScalingFactor(0.15);

        // moveit::core::RobotStatePtr pre_insert_state = arm_->getCurrentState(5.0);
        // if (!pre_insert_state) {
        //     RCLCPP_ERROR(this->get_logger(), "Failed to get state before insertion");
        //     return false;
        // }
        //arm_->setStartState(*pre_insert_state);
        
        //debug
        // geometry_msgs::msg::Pose test;
        // test.orientation = tf2::toMsg(target_quat);
        
        // Eigen::Vector3d approach_pos_w = slot_pos_w + (rack_normal * 0.30);
        // approach.position = tf2::toMsg(approach_pos_w);

        std::vector<geometry_msgs::msg::Pose> waypoints = { approach, target }; //target
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = arm_->computeCartesianPath(
            waypoints,
            0.01,   
            2.0,     // jump_threshold: reject IK solutions where any joint jumps > 2 rad
            trajectory,
            false
        );
        // usePILZ("PTP");
        // arm_->setPoseTarget(test);
        // arm_->move();

        RCLCPP_INFO(this->get_logger(), "Cartesian path: %.1f%%", fraction * 100.0);

        if (fraction < 0.98) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian path only %.1f%% complete", fraction * 100.0);
            return false;
        }

        moveit::core::RobotStatePtr state = arm_->getCurrentState(1.0);
        robot_trajectory::RobotTrajectory rt(arm_->getRobotModel(), arm_->getName());
        rt.setRobotTrajectoryMsg(*state, trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed");
            return false;
        }

        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);
        RCLCPP_INFO(this->get_logger(), "Cartesian insertion succeeded");

        //rclcpp::sleep_for(2s);

        


        if (pick) {

            //Drop sensitivity to prevent a trip from gripper snapping or initial heavy load
            RCLCPP_INFO(this->get_logger(), "Lowering collision sensitivity for picking state.");
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

            //Update payload to loaded mass
            RCLCPP_INFO(this->get_logger(), "Updating JAKA payload to LOADED state.");
            set_robot_payload(4, 0.0, 0.0, 243.0);

            current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose lift_pose2 = current_pose.pose;
            lift_pose2.position.z += 0.02;
            arm_->setPoseTarget(lift_pose2);
            arm_->move();
            
            //restore collision protection
            RCLCPP_INFO(this->get_logger(), "Restoring normal collision sensitivity.");
            set_collision_level(3);

        } else if (place) {
            RCLCPP_INFO(this->get_logger(), "Lowering collision sensitivity for placing state.");
            set_collision_level(0);

            geometry_msgs::msg::PoseStamped current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose drop_pose = current_pose.pose;
            drop_pose.position.z -= 0.02;

            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.1);
            arm_->setMaxAccelerationScalingFactor(0.1);
            arm_->setPoseTarget(drop_pose);
            arm_->move();

            set_gripper(false);

            current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose retreat_pose = current_pose.pose;
            retreat_pose.position.z -= 0.025;
            arm_->setPoseTarget(retreat_pose);
            arm_->move();
            set_robot_payload(1.77, 0.0, 0.0, 65.0);
            set_collision_level(3);
        }

        // --- Phase 3: Cartesian retraction seeded from post-insertion state ---
        // pre_insert_state = arm_->getCurrentState(5.0);
        // if (!pre_insert_state) {
        //     RCLCPP_ERROR(this->get_logger(), "Failed to get state before retraction");
        //     return false;
        // }
        // arm_->setStartState(*pre_insert_state);

        // Target = current pose after pick/place
        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose retract_target = current_stamped.pose;

        geometry_msgs::msg::Pose retract_approach;
        retract_approach.orientation = retract_target.orientation;

        for (double dist : {0.70, 0.70, 0.65}) {
            Eigen::Vector3d retract_pos_w = slot_pos_w + (rack_normal * dist);
            if (pick) {
                retract_pos_w.z() += 0.02;
            } else if (place) {
                retract_pos_w.z() -= 0.02;
            }

            retract_approach.position = tf2::toMsg(retract_pos_w);

            waypoints = { retract_target, retract_approach };
            fraction = arm_->computeCartesianPath(waypoints, 0.01, 2.0, trajectory, false);

            RCLCPP_INFO(this->get_logger(), "Cartesian retraction at %.2fm: %.1f%%", dist, fraction * 100.0);

            if (fraction >= 0.98) {
                break;
            }
            RCLCPP_WARN(this->get_logger(), "Cartesian retraction failed at %.2fm, trying closer", dist);
        }

        if (fraction < 0.98) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian retraction only %.1f%% complete", fraction * 100.0);
            return false;
        }

        state = arm_->getCurrentState(1);
        rt.setRobotTrajectoryMsg(*state, trajectory);

        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed");
            return false;
        }

        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);
        RCLCPP_INFO(this->get_logger(), "Cartesian retraction succeeded");

        state = arm_->getCurrentState(1);
        //rt(arm_->getRobotModel(), arm_->getName());
        rt.setRobotTrajectoryMsg(*state, trajectory);

        iptp;
        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed");
            return false;
        }
                rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);
        RCLCPP_INFO(this->get_logger(), "Cartesian retraction succeeded");

        // std::vector<double> retract_state = approach_state;
        // retract_state[0] = arm_->getCurrentJointValues()[0];

        arm_->setJointValueTarget(approach_state);
        arm_->move();
        return true;
    }

    bool place_on_box(const geometry_msgs::msg::Pose& box_pose) {

        // ── Step 1: Rotate joint 0 to face the box ────────────────────────────────
        double angle = std::atan2(box_pose.position.y, box_pose.position.x);
        std::vector<double> face_state = approach_config_;
        face_state[0] = angle;

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to rotate to face box");
            return false;
        }

        // ── Step 2: Build target orientation — Extract box normal from pose orientation ───────
        Eigen::Quaterniond box_quat(
            box_pose.orientation.w,
            box_pose.orientation.x,
            box_pose.orientation.y,
            box_pose.orientation.z);

        // Box normal = X axis of the box frame (col(0) of rotation matrix)
        Eigen::Vector3d box_normal = box_quat.toRotationMatrix().col(0).normalized();

        Eigen::Vector3d z_axis    = -box_normal;     
        Eigen::Vector3d world_up(0.0, 0.0, 1.0);
        Eigen::Vector3d x_axis    = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis    = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = z_axis;
        Eigen::Quaterniond target_quat(rot_mat);

        // ── Step 3: Target pose — 15cm along normal from box surface ──────────────
        geometry_msgs::msg::Pose target;
        target.position.x  = box_pose.position.x + box_normal.x() * 0.15;
        target.position.y  = box_pose.position.y + box_normal.y() * 0.15;
        target.position.z  = box_pose.position.z + 0.02;
        target.orientation = tf2::toMsg(target_quat);

        // ── Step 4: PTP move to target ────────────────────────────────────────────
        RCLCPP_INFO(this->get_logger(),
                "Trying PILZ PTP approach : [%.3f, %.3f, %.3f]",
                target.position.x, target.position.y, target.position.z);

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.3);
        arm_->setMaxAccelerationScalingFactor(0.2);
        arm_->setPoseTarget(target);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to reach box target pose");
            return false;
        }

        set_collision_level(0);
        target.position.z = box_pose.position.z;

        arm_->setMaxVelocityScalingFactor(0.1);
        arm_->setMaxAccelerationScalingFactor(0.1);
        arm_->setPoseTarget(target);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed lower on box");
            return false;
        }

        

        // ── Step 5: Release tray — lower Z, open gripper, small retreat ───────────

        set_gripper(false);

        // Update payload BACK to empty gripper weight 
        RCLCPP_INFO(this->get_logger(), "Updating JAKA payload back to EMPTY state.");
        set_robot_payload(1.77, 0.0, 0.0, 65.0);

        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose drop_pose = current_stamped.pose;
        drop_pose.position.z -= 0.01;

        arm_->setPoseTarget(drop_pose);
        arm_->move();

        set_collision_level(1);

        // ── Step 6: Retreat 30cm along normal (+X) ────────────────────────────────
        current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose retreat_pose = current_stamped.pose;
        retreat_pose.position.x += box_normal.x() * 0.30;
        retreat_pose.position.y += box_normal.y() * 0.30;

        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.3);
        arm_->setPoseTarget(retreat_pose);
        arm_->move();

        // Restore safety levels for fast travel back to homing positions
        set_collision_level(3);

        // ── Step 7: Return to approach config keeping joint 0 facing box ──────────
        std::vector<double> return_state = approach_config_;
        return_state[0] = face_state[0];

        arm_->setJointValueTarget(return_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to return to approach config");
            return false;
        }

        return true;
    }
    
    bool pick_from_box(const geometry_msgs::msg::Pose& box_pose) {

        // ── Step 1: Go to approach config with joint[0] facing box, joint[1] tilted -10° ──
        double angle = std::atan2(box_pose.position.y, box_pose.position.x);
        std::vector<double> face_approach_state = approach_config_;
        face_approach_state[0] = angle;
        face_approach_state[1] = approach_config_[1] - (10.0 * M_PI / 180.0);  // −10° tilt

        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_approach_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to move to tilted approach config");
            return false;
        }

        // ── Step 2: YOLO scan and verify tray within 40cm of box ─────────────────
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

                if (dist < 0.40) {
                    RCLCPP_INFO(this->get_logger(),
                        "Tray %s found %.3fm from box, using it", tray_frame.c_str(), dist);
                    best_tray_tf = tf;
                    tray_found = true;
                    break;
                }
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "TF lookup failed for %s: %s", tray_frame.c_str(), ex.what());
                continue;
            }
        }

        if (!tray_found) {
            RCLCPP_ERROR(this->get_logger(), "No tray found within 40cm of box, returning to approach config");

            // Go back to untilted approach config, joint[0] still facing box
            std::vector<double> untilted_state = approach_config_;
            untilted_state[0] = angle;
            arm_->setJointValueTarget(untilted_state);
            arm_->move();
            return false;
        }
        
        // ── Step 3: Build target pose from tray TF ────────────────────────────────
        Eigen::Isometry3d tray_eigen = tf2::transformToEigen(best_tray_tf);
        Eigen::Vector3d tray_pos     = tray_eigen.translation();
        Eigen::Vector3d tray_normal  = tray_eigen.rotation().col(0).normalized();

        Eigen::Vector3d z_axis   = -tray_normal;
        Eigen::Vector3d world_up(0.0, 0.0, 1.0);
        Eigen::Vector3d x_axis   = world_up.cross(z_axis).normalized();
        Eigen::Vector3d y_axis   = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = z_axis;
        Eigen::Quaterniond tray_quat(rot_mat);

        // Target: tray center, 2cm lower
        geometry_msgs::msg::Pose target;
        target.position.x  = tray_pos.x() - tray_normal.x() * 0.015;
        target.position.y  = tray_pos.y() - tray_normal.y() * 0.015;
        target.position.z  = tray_pos.z() - 0.02;
        target.orientation = tf2::toMsg(tray_quat);

        // Approach: 20cm along tray normal, also 2cm lower
        geometry_msgs::msg::Pose approach;
        approach.position.x  = tray_pos.x() + tray_normal.x() * 0.20;
        approach.position.y  = tray_pos.y() + tray_normal.y() * 0.20;
        approach.position.z  = tray_pos.z() - 0.02;
        approach.orientation = tf2::toMsg(tray_quat);

        // ── Step 5: PTP to approach pose (20cm out, 2cm low) ─────────────────────
        usePILZ("PTP");
        arm_->setMaxVelocityScalingFactor(0.3);
        arm_->setMaxAccelerationScalingFactor(0.3);
        arm_->setPoseTarget(approach);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to reach approach pose above box tray");
            return false;
        }

        // ── Step 6: Open gripper before insertion ─────────────────────────────────
        set_gripper(false);

        // ── Step 7: Cartesian insertion to tray center (2cm low) ─────────────────
        arm_->setMaxVelocityScalingFactor(0.2);
        arm_->setMaxAccelerationScalingFactor(0.15);

        std::vector<geometry_msgs::msg::Pose> waypoints = { approach, target };
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = arm_->computeCartesianPath(
            waypoints,
            0.01,
            2.0,
            trajectory,
            false
        );

        RCLCPP_INFO(this->get_logger(), "Cartesian insertion path: %.1f%%", fraction * 100.0);
        if (fraction < 0.98) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian insertion only %.1f%% complete, aborting", fraction * 100.0);
            return false;
        }

        moveit::core::RobotStatePtr state = arm_->getCurrentState(1.0);
        robot_trajectory::RobotTrajectory rt(arm_->getRobotModel(), arm_->getName());
        rt.setRobotTrajectoryMsg(*state, trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        if (!iptp.computeTimeStamps(rt, 0.2, 0.15)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed for insertion");
            return false;
        }

        rt.getRobotTrajectoryMsg(trajectory);
        arm_->execute(trajectory);
        RCLCPP_INFO(this->get_logger(), "Cartesian insertion complete");

        // ── Step 8: Disable collision, lift 3cm, close gripper ───────────────────
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

        // ── Step 9: Update payload to loaded state ────────────────────────────────
        RCLCPP_INFO(this->get_logger(), "Updating payload to LOADED state.");
        set_robot_payload(4, 0.0, 0.0, 243.0);

        // ── Step 10: Restore collision protection ─────────────────────────────────
        set_collision_level(3);
        face_approach_state[1] = approach_config_[1];
        // ── Step 11: Return to approach config, joint[0] still facing box ─────────

        arm_->setMaxVelocityScalingFactor(0.4);
        arm_->setMaxAccelerationScalingFactor(0.4);
        arm_->setJointValueTarget(face_approach_state);
        if (arm_->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to return to approach config after pick");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "pick_from_box complete");
        return true;
    }

private:
    bool frame_exists(const std::string& frame) const {
        try {
            auto tf = tf_buffer_->lookupTransform("world", frame, tf2::TimePointZero);
            rclcpp::Time tf_time(tf.header.stamp);

            if (last_scan_time_.nanoseconds() == 0) {
                return true;
            }

            return tf_time >= (last_scan_time_ - rclcpp::Duration::from_seconds(0.5));
        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    bool is_new_rack(const Eigen::Vector3d& pos, const std::vector<Rack>& known) {
        for (const auto& r : known) {
            if ((pos.head<2>() - r.position.head<2>()).norm() < 0.5) return false;
        }
        return true;
    }
    bool use_real_gripper_ = true; // set false for simulation

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_client_;
    rclcpp::Client<jaka_msgs::srv::SetIO>::SharedPtr io_client_;
    rclcpp::Client<jaka_msgs::srv::SetPayload>::SharedPtr payload_client_;
    rclcpp::Client<jaka_msgs::srv::SetCollision>::SharedPtr collision_client_;


    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    //sensor_msgs::msg::JointState last_joint_state_;

    std::vector<double> base_config_;
    std::vector<double> approach_config_;
    std::vector<double> gripper_open_;
    std::vector<double> gripper_closed_;
    double tray_slot_base_height_;
    double tray_slot_offset_;
    rclcpp::Time last_scan_time_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrayHandler>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread([&executor]() { executor.spin(); }).detach();

    node->init();
    node->set_collision_level(3);
    node->set_robot_payload(1.77, 0.0, 0.0, 65.0);
    // node->set_gripper(true);
    // node->set_gripper(false);
    // node->pick_from_box(node->BOX_POSE);
    // rclcpp::sleep_for(10s); 
    // node->place_on_box(node->BOX_POSE);
    
    auto rough = node->find_racks();
    auto refined = node->refine_racks(rough);


    if (!refined.empty()) {
        node->face_rack_slot(refined[0], 8, true);
        node->place_on_box(node->BOX_POSE);
        node->face_rack_slot(refined[0], 7, true);
        node->face_rack_slot(refined[0], 8, false, true);
        node->pick_from_box(node->BOX_POSE);
        node->face_rack_slot(refined[0], 9, false, true);
        // node->face_rack_slot(refined[1], 5);
        // node->face_rack_slot(refined[0], 4);
        // node->face_rack_slot(refined[1], 5);
        // node->face_rack_slot(refined[0], 2);
        // node->face_rack_slot(refined[1], 5);
        // node->face_rack_slot(refined[0], 7);
        // node->face_rack_slot(refined[1], 5);
        // node->face_rack_slot(refined[0], 0);
        // node->face_rack_slot(refined[0], 1);
        // node->face_rack_slot(refined[0], 2);
        // node->face_rack_slot(refined[0], 3);
        // node->face_rack_slot(refined[0], 4);
        // node->face_rack_slot(refined[0], 5);
        // node->face_rack_slot(refined[0], 6);
        // node->face_rack_slot(refined[0], 7);
        // node->face_rack_slot(refined[0], 8);
        // node->face_rack_slot(refined[0], 9);
    }

    rclcpp::shutdown();
    return 0;
}
