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
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        base_config_ = {0.0, 2.513, -2.471, -0.541, -1.5708, 0.8754};
        //approach_config_ = {0.0, 0.716, -2.496, 1.780, -1.5708, 0.8754};
        approach_config_ = {0.0, 1.937, -2.758, 0.820, -1.5708, 0.8754};
        gripper_closed_ = {0.005};
        gripper_open_ = {0.065};
        tray_slot_base_height_ = 0.15;
        tray_slot_offset_ = 0.11;
    }

    void init() {
        arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "jaka_s12");
        gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "gripper");
        arm_->setMaxVelocityScalingFactor(0.8);
        arm_->setMaxAccelerationScalingFactor(0.6);

        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_vel", 1.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_acc", 2.25);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_trans_dec", -5.0);
        this->declare_parameter("robot_description_planning.cartesian_limits.max_rot_vel",   1.57);

        trigger_client_ = this->create_client<std_srvs::srv::Trigger>("/yolo_inference/trigger_detection");

        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, 
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                last_joint_state_ = *msg;
            });
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
        RCLCPP_INFO(this->get_logger(), "Requesting YOLO scan...");
        while (!trigger_client_->wait_for_service(1s)) {
            RCLCPP_INFO(this->get_logger(), "Waiting for YOLO service...");
        }
        trigger_client_->async_send_request(request);
        
        // Give YOLO a moment to finish processing and TF to update
        rclcpp::sleep_for(2s); 
        RCLCPP_INFO(this->get_logger(), "YOLO scan wait complete");
    }

    // --- 1. SCANNING ---
    std::vector<Rack> find_racks() {
        RCLCPP_INFO(this->get_logger(), "Starting rack scan...");
        std::vector<Rack> found_racks;

        usePILZ("PTP");
        arm_->setMaxAccelerationScalingFactor(0.4);
        // arm_->setJointValueTarget(base_config_);
        // arm_->move();

        std::vector<double> current_state = base_config_;
        double step = 1.04; //1.04      

        for (double angle = 0.0; angle < 3.14 - step; angle += step) {
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

        arm_->setJointValueTarget(base_config_);
        arm_->move();
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
            face_state[0] = angle + base_config_[0];

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

        arm_->setJointValueTarget(base_config_);
        arm_->move();
        arm_->setJointValueTarget(approach_config_);
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
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = 10.0;
        box.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = max_z;

        // Define the Pose of the Box
        // The box is centered at its origin, so to make it go from Z=0 to Z=max_z,
        // we must place the center of the box at Z = max_z / 2
        geometry_msgs::msg::Pose box_pose;
        box_pose.position.x = 0.0;
        box_pose.position.y = 0.0;
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
        approach_state[0] = angle + approach_config_[0];

        arm_->setJointValueTarget(approach_state);
        arm_->move();
        Eigen::Isometry3d rack_eigen = tf2::transformToEigen(rack.transform);
        Eigen::Vector3d rack_normal  = rack_eigen.rotation().col(0).normalized();
        
        float pick_place_offset = 0;
        if(pick){
            pick_place_offset = -0.005;
        } else if(place){
            pick_place_offset = 0.01;
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
        for (double dist : {0.60, 0.60, 0.55}) {
            Eigen::Vector3d approach_pos_w = slot_pos_w + (rack_normal * dist);
            approach.position = tf2::toMsg(approach_pos_w);

            RCLCPP_INFO(this->get_logger(),
                "Trying PILZ PTP approach at %.2fm: [%.3f, %.3f, %.3f]",
                dist, approach_pos_w.x(), approach_pos_w.y(), approach_pos_w.z());

            // Try PILZ PTP first — deterministic, shortest joint path
            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.4);
            arm_->setMaxAccelerationScalingFactor(0.3);
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
            gripper_->setJointValueTarget(gripper_open_);
            gripper_->move();
            for (int t = 0; t < 20; ++t) {
                std::string tray_frame = "tray_" + std::to_string(t);
                if (!frame_exists(tray_frame)) break;
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

                    // Update target position and orientation
                    target.position.x  = tray_pos.x() - tray_normal.x() * 0.01;
                    target.position.y  = tray_pos.y() - tray_normal.y() * 0.01;
                    target.position.z  = tray_pos.z() - 0.005;
                    target.orientation = tf2::toMsg(tray_quat);
                
                    approach.orientation = tf2::toMsg(tray_quat);

                    break;

                } catch (const tf2::TransformException& ex) {
                    RCLCPP_WARN(this->get_logger(), "Tray TF lookup failed for %s: %s", tray_frame.c_str(), ex.what());
                    continue;
                }
            }
        }

        // --- Phase 2: Cartesian insertion seeded from current state ---
        arm_->setMaxVelocityScalingFactor(0.2);
        arm_->setMaxAccelerationScalingFactor(0.15);

        moveit::core::RobotStatePtr pre_insert_state = arm_->getCurrentState(5.0);
        if (!pre_insert_state) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get state before insertion");
            return false;
        }
        arm_->setStartState(*pre_insert_state);

        // Finer step + jump threshold to reject IK branch switches
        std::vector<geometry_msgs::msg::Pose> waypoints = { approach, target };
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = arm_->computeCartesianPath(
            waypoints,
            0.005,   
            2.0,     // jump_threshold: reject IK solutions where any joint jumps > 2 rad
            trajectory,
            false
        );

        RCLCPP_INFO(this->get_logger(), "Cartesian path: %.1f%%", fraction * 100.0);

        if (fraction < 0.90) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian path only %.1f%% complete", fraction * 100.0);
            return false;
        }

        moveit::core::RobotStatePtr state = arm_->getCurrentState(5.0);
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

        rclcpp::sleep_for(2s);

        if (pick) {
            geometry_msgs::msg::PoseStamped current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose lift_pose = current_pose.pose;
            lift_pose.position.z += 0.01;

            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.1);
            arm_->setMaxAccelerationScalingFactor(0.1);
            arm_->setPoseTarget(lift_pose);
            arm_->move();

            gripper_->setJointValueTarget(gripper_closed_);
            gripper_->move();

            current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose lift_pose2 = current_pose.pose;
            lift_pose2.position.z += 0.02;
            arm_->setPoseTarget(lift_pose2);
            arm_->move();

        } else if (place) {
            geometry_msgs::msg::PoseStamped current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose drop_pose = current_pose.pose;
            drop_pose.position.z -= 0.01;

            usePILZ("PTP");
            arm_->setMaxVelocityScalingFactor(0.1);
            arm_->setMaxAccelerationScalingFactor(0.1);
            arm_->setPoseTarget(drop_pose);
            arm_->move();

            gripper_->setJointValueTarget(gripper_open_);
            gripper_->move();

            current_pose = arm_->getCurrentPose();
            geometry_msgs::msg::Pose retreat_pose = current_pose.pose;
            retreat_pose.position.z -= 0.01;
            arm_->setPoseTarget(retreat_pose);
            arm_->move();
        }

        // --- Phase 3: Cartesian retraction seeded from post-insertion state ---
        pre_insert_state = arm_->getCurrentState(5.0);
        if (!pre_insert_state) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get state before retraction");
            return false;
        }
        arm_->setStartState(*pre_insert_state);

        // Target = current pose after pick/place
        geometry_msgs::msg::PoseStamped current_stamped = arm_->getCurrentPose();
        geometry_msgs::msg::Pose retract_target = current_stamped.pose;

        // Approach = original approach but offset vertically based on operation
        geometry_msgs::msg::Pose retract_approach = approach;
        if (pick) {
            retract_approach.position.z += 0.02;  // 2cm higher for pick
        } else if (place) {
            retract_approach.position.z -= 0.02;  // 2cm lower for place
        }

        waypoints = { retract_target, retract_approach };
        trajectory;
        fraction = arm_->computeCartesianPath(
            waypoints,
            0.005,
            2.0,
            trajectory,
            false
        );

        RCLCPP_INFO(this->get_logger(), "Cartesian retraction path: %.1f%%", fraction * 100.0);

        if (fraction < 0.90) {
            RCLCPP_ERROR(this->get_logger(), "Cartesian retraction only %.1f%% complete", fraction * 100.0);
            return false;
        }

        state = arm_->getCurrentState(5.0);
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

        std::vector<double> retract_state = approach_state;
        retract_state[0] = arm_->getCurrentJointValues()[0];

        arm_->setJointValueTarget(retract_state);
        arm_->move();
        return true;
    }

private:
    bool frame_exists(const std::string& frame) const {
        const auto frames = tf_buffer_->getAllFrameNames();
        return std::find(frames.begin(), frames.end(), frame) != frames.end();
    }

    bool is_new_rack(const Eigen::Vector3d& pos, const std::vector<Rack>& known) {
        for (const auto& r : known) {
            if ((pos.head<2>() - r.position.head<2>()).norm() < 0.5) return false;
        }
        return true;
    }

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_client_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    sensor_msgs::msg::JointState last_joint_state_;

    std::vector<double> base_config_;
    std::vector<double> approach_config_;
    std::vector<double> gripper_open_;
    std::vector<double> gripper_closed_;
    double tray_slot_base_height_;
    double tray_slot_offset_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrayHandler>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread([&executor]() { executor.spin(); }).detach();

    node->init();

    auto rough = node->find_racks();
    auto refined = node->refine_racks(rough);


    if (!refined.empty()) {
        node->face_rack_slot(refined[0], 7, true);
        node->face_rack_slot(refined[1], 5, false, true);
        node->face_rack_slot(refined[0], 4);
        node->face_rack_slot(refined[1], 5);
        node->face_rack_slot(refined[0], 2);
        node->face_rack_slot(refined[1], 5);
        node->face_rack_slot(refined[0], 7);
        node->face_rack_slot(refined[1], 5);
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
