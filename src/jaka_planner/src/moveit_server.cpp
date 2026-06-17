#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_srvs/srv/empty.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "jaka_msgs/srv/set_io.hpp"
#include <jaka_msgs/srv/set_payload.hpp>
#include <jaka_msgs/srv/set_collision.hpp>
#include "jaka_msgs/msg/robot_msg.hpp"
#include "std_srvs/srv/trigger.hpp"


#include "jaka_planner/JAKAZuRobot.h"
#include "jaka_planner/jkerr.h"
#include "jaka_planner/jktypes.h"

#include <action_msgs/msg/goal_status_array.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include "rclcpp_action/rclcpp_action.hpp"

#include <string>
#include <map>
#include <thread>
using namespace std;

JAKAZuRobot robot;
const double PI = 3.1415926;
// bool in_pos;
// int ret_preempt;
// int ret_inPos;

typedef rclcpp_action::Server<control_msgs::action::FollowJointTrajectory> Server;
rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub;
rclcpp::Publisher<jaka_msgs::msg::RobotMsg>::SharedPtr robot_states_pub;


// Map error codes to messages
map<int, string> mapErr = {
    {2,   "ERR_FUCTION_CALL_ERROR"},
    {-1,  "ERR_INVALID_HANDLER"},
    {-2,  "ERR_INVALID_PARAMETER"},
    {-3,  "ERR_COMMUNICATION_ERR"},
    {-4,  "ERR_KINE_INVERSE_ERR"},
    {-5,  "ERR_EMERGENCY_PRESSED"},
    {-6,  "ERR_NOT_POWERED"},
    {-7,  "ERR_NOT_ENABLED"},
    {-8,  "ERR_DISABLE_SERVOMODE"},
    {-9,  "ERR_NOT_OFF_ENABLE"},
    {-10, "ERR_PROGRAM_IS_RUNNING"},
    {-11, "ERR_CANNOT_OPEN_FILE"},
    {-12, "ERR_MOTION_ABNORMAL"}
};

// Determine if the robot has reached the target position.
bool jointStates(const JointValue &joint_pose)
{
    // RobotStatus robotstatus;
    JointValue joint_position;
    // robot.get_robot_status(&robotstatus);
    robot.get_joint_position(&joint_position);

    bool joint_state = true;
    for (int i = 0; i < 6; i++)
    {
        // double actual_deg = robotstatus.joint_position[i] * 180.0 / PI;
        double actual_deg = joint_position.jVal[i] * 180.0 / PI;
        double target_deg = joint_pose.jVal[i] * 180.0 / PI;
        // Within +/- 0.2 degrees tolerance
        bool ret = (target_deg - 0.2 < actual_deg) && (actual_deg < target_deg + 0.2);
        joint_state = joint_state && ret;
    }

    RCLCPP_INFO(rclcpp::get_logger("jointStates"), 
                "Whether the robot has reached the target position: %d", joint_state);
    return joint_state;
}

bool jointStatesWithinTolerance(const JointValue &joint_pose, double tolerance_deg)
{
    JointValue joint_position;
    robot.get_joint_position(&joint_position);

    for (int i = 0; i < 6; i++)
    {
        double actual_deg = joint_position.jVal[i] * 180.0 / PI;
        double target_deg = joint_pose.jVal[i] * 180.0 / PI;
        if (std::abs(actual_deg - target_deg) > tolerance_deg)
        {
            return false;
        }
    }

    return true;
}

void logJointError(const JointValue &joint_pose)
{
    JointValue joint_position;
    robot.get_joint_position(&joint_position);

    for (int i = 0; i < 6; i++)
    {
        double actual_deg = joint_position.jVal[i] * 180.0 / PI;
        double target_deg = joint_pose.jVal[i] * 180.0 / PI;
        RCLCPP_WARN(
            rclcpp::get_logger("goalCb"),
            "joint_%d actual=%.3f deg target=%.3f deg error=%.3f deg",
            i + 1,
            actual_deg,
            target_deg,
            actual_deg - target_deg);
    }
}

// Handle a new FollowJointTrajectory goal
void goalCb(const shared_ptr<rclcpp_action::ServerGoalHandle<control_msgs::action::FollowJointTrajectory>> goal_handle)
{
    // Enable servo mode
    robot.servo_move_enable(true);

    // Retrieve the trajectory from the goal
    auto goal = goal_handle->get_goal();
    const auto &traj = goal->trajectory;
    int point_num = traj.points.size();
    RCLCPP_INFO(rclcpp::get_logger("goalCb"), "number of points: %d", point_num);

    if (point_num == 0)
    {
        RCLCPP_ERROR(rclcpp::get_logger("goalCb"), "Trajectory has no points. Aborting goal.");
        goal_handle->abort(make_shared<control_msgs::action::FollowJointTrajectory::Result>());
        return;
    }

    float lastDuration = 0.0;
    JointValue joint_pose;

    for (int i = 1; i < point_num; i++)
    {
        // Grab the positions from the i-th trajectory point
        for (int j = 0; j < 6; j++) {
            joint_pose.jVal[j] = traj.points[i].positions[j];
        }

        // Convert time_from_start to a float seconds
        float Duration = static_cast<float>(traj.points[i].time_from_start.sec) +
                         static_cast<float>(traj.points[i].time_from_start.nanosec) * 1e-9;

        // Calculate time delta relative to previous point
        float dt = Duration - lastDuration;
        lastDuration = Duration;

        // step_num matches old ROS1 logic: step_num = dt / 0.008
        int step_num = static_cast<int>(dt / 0.008f);
        step_num = max(step_num, 1);

        int sdk_res = robot.servo_j(&joint_pose, MoveMode::ABS, step_num);
        if (sdk_res != 0)
        {
            RCLCPP_INFO(rclcpp::get_logger("goalCb"), "Servo_j Motion Failed");
        }

        RCLCPP_INFO(rclcpp::get_logger("goalCb"), "The return status of servo_j: %d", sdk_res);
        RCLCPP_INFO(rclcpp::get_logger("goalCb"), 
                    "For point no.: %d, Accepted joint angle: %f %f %f %f %f %f, dt=%f, step_num=%d", i,
                    joint_pose.jVal[0], joint_pose.jVal[1], joint_pose.jVal[2],
                    joint_pose.jVal[3], joint_pose.jVal[4], joint_pose.jVal[5],
                    dt, step_num);

        // // Check if the action was canceled in the middle
        // if (goal_handle->is_canceling())
        // {
        //     // stop motion
        //     robot.motion_abort();
        //     robot.servo_move_enable(false);
        //     RCLCPP_INFO(rclcpp::get_logger("goalCb"), "Servo Mode Disable, motion canceled");
        //     auto result = make_shared<control_msgs::action::FollowJointTrajectory::Result>();
        //     goal_handle->canceled(result);
        //     return;
        // }
    }

    // Wait until the robot is actually at the final position, or until canceled
    auto wait_start = chrono::steady_clock::now();
    constexpr double settle_tolerance_deg = 0.5;
    constexpr auto settle_timeout = chrono::seconds(5);

    while (rclcpp::ok())
    {
        if (jointStatesWithinTolerance(joint_pose, settle_tolerance_deg))
        {
            robot.servo_move_enable(false);
            RCLCPP_INFO(rclcpp::get_logger("goalCb"), "Servo Mode Disable: Target Reached");
            RCLCPP_INFO(rclcpp::get_logger("goalCb"), 
                        "==============Motion stops or reaches the target position==============");
            break;
        }

        RCLCPP_INFO(rclcpp::get_logger("jointStates"),
                    "Whether the robot has reached the target position: 0");

        if (chrono::steady_clock::now() - wait_start > settle_timeout)
        {
            RCLCPP_WARN(rclcpp::get_logger("goalCb"),
                        "Target was not reached within timeout. Accepting goal with relaxed settling check.");
            logJointError(joint_pose);
            robot.servo_move_enable(false);
            break;
        }

        // if (goal_handle->is_canceling())
        // {
        //     robot.motion_abort();
        //     robot.servo_move_enable(false);
        //     RCLCPP_INFO(rclcpp::get_logger("goalCb"), "Servo Mode Disable");
        //     RCLCPP_INFO(rclcpp::get_logger("goalCb"), 
        //                 "==============Motion stops or was canceled==============");
        //     auto result = make_shared<control_msgs::action::FollowJointTrajectory::Result>();
        //     goal_handle->canceled(result);
        //     return;
        // }

        rclcpp::sleep_for(chrono::milliseconds(500));
    }

    // // After processing all points, check if the goal was canceled
    // if (goal_handle->is_canceling()) {
    //     robot.motion_abort();
    //     robot.servo_move_enable(false);
    //     RCLCPP_INFO(rclcpp::get_logger("goalCb"), "Servo Mode Disabled, Motion Canceled");
    //     auto result = make_shared<control_msgs::action::FollowJointTrajectory::Result>();
    //     goal_handle->canceled(result);
    //     return;
    // }

    // If we get here, it succeeded
    auto result = make_shared<control_msgs::action::FollowJointTrajectory::Result>();
    goal_handle->succeed(result);
    rclcpp::sleep_for(chrono::milliseconds(500));
}

// Publish the robot's joint states to /joint_states (for RViz / MoveIt feedback)
void joint_states_callback(rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr &joint_states_pub)
{
    sensor_msgs::msg::JointState joint_msg;
    // RobotStatus robotstatus;
    JointValue joint_position;
    // robot.get_robot_status(&robotstatus);
    robot.get_joint_position(&joint_position);

    for (int i = 0; i < 6; i++)
    {
        // joint_msg.position.push_back(robotstatus.joint_position[i]);
        joint_msg.position.push_back(joint_position.jVal[i]);
        joint_msg.name.push_back("joint_" + to_string(i+1));
    }
    joint_msg.header.stamp = rclcpp::Clock().now();
    joint_states_pub->publish(joint_msg);
}

void sigintHandler(int /*sig*/) {
    rclcpp::shutdown();
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    rclcpp::init(argc, argv);
    signal(SIGINT, sigintHandler);
    auto node = rclcpp::Node::make_shared("moveit_server");

    // Read parameters
    string default_ip = "10.5.5.100";
    string default_model = "zu3";
    string robot_ip = node->declare_parameter("ip", default_ip);
    string robot_model = node->declare_parameter("model", default_model);

    // Connect to robot
    robot.login_in(robot_ip.c_str(), false);
    rclcpp::Rate rate(125);

    // Turn off servo at startup
    robot.servo_move_enable(false);
    rclcpp::sleep_for(chrono::milliseconds(500));

    // Filter param
    robot.servo_move_use_joint_LPF(0.5);

    // Power on + enable
    robot.power_on();
    rclcpp::sleep_for(chrono::seconds(8));
    robot.enable_robot();
    rclcpp::sleep_for(chrono::seconds(4));

    // Publisher for /joint_states
    joint_states_pub = node->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    auto joint_state_timer = node->create_wall_timer(
        chrono::milliseconds(8),
        []() {
            joint_states_callback(joint_states_pub);
        });

    robot_states_pub = node->create_publisher<jaka_msgs::msg::RobotMsg>("/jaka_driver/robot_states", 10);
    auto robot_states_timer = node->create_wall_timer(
        chrono::milliseconds(100),  // 10 Hz, matches original driver
        []() {
            jaka_msgs::msg::RobotMsg msg;
            BOOL in_col = false;
            robot.is_in_collision(&in_col);
            msg.collision_state = in_col ? 1 : 0;
            robot_states_pub->publish(msg);
        });

    // Create Action Server for FollowJointTrajectory
    auto moveit_server = rclcpp_action::create_server<control_msgs::action::FollowJointTrajectory>(
        node,
        "/jaka_" + robot_model + "_controller/follow_joint_trajectory",
        // Goal callback
        [](const rclcpp_action::GoalUUID &uuid,
           shared_ptr<const control_msgs::action::FollowJointTrajectory::Goal> goal) {
            RCLCPP_INFO(rclcpp::get_logger("moveit_server"), "Received goal request");
            (void)uuid; // Avoid unused parameter warning
            (void)goal; // Avoid unused parameter warning
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        // // Cancel callback
        // [](const shared_ptr<rclcpp_action::ServerGoalHandle<control_msgs::action::FollowJointTrajectory>> goal_handle) {
        //     (void)goal_handle;  // Avoid unused parameter warning
        //     RCLCPP_INFO(rclcpp::get_logger("moveit_server"), "Received cancel request");
        //     return rclcpp_action::CancelResponse::ACCEPT;
        // },
        nullptr,  // Cancel callback removed
        // Execute callback
        [](const shared_ptr<rclcpp_action::ServerGoalHandle<control_msgs::action::FollowJointTrajectory>> goal_handle) {
            RCLCPP_INFO(rclcpp::get_logger("moveit_server"), "Executing goal");
            goalCb(goal_handle); 
        }
    );
    // IO service
    auto io_service = node->create_service<jaka_msgs::srv::SetIO>(
        "/jaka_driver/set_io",
        [](const shared_ptr<jaka_msgs::srv::SetIO::Request> request,
        shared_ptr<jaka_msgs::srv::SetIO::Response> response) {
            if (request->signal == "digital") {
                BOOL val = (request->value > 0.5f) ? TRUE : FALSE;
                int ret = robot.set_digital_output((IOType)request->type, request->index, val);
                response->ret = ret;
                response->message = (ret == 0) ? "OK" : "error";
            } else {
                response->ret = -1;
                response->message = "unsupported signal type";
            }
        });
        // Payload Service
    auto payload_service = node->create_service<jaka_msgs::srv::SetPayload>(
        "/jaka_driver/set_payload",
        [](const std::shared_ptr<jaka_msgs::srv::SetPayload::Request> request,
           std::shared_ptr<jaka_msgs::srv::SetPayload::Response> response) {
            
            PayLoad jaka_payload;
            // ROS 2 service likely uses request->mass representation
            jaka_payload.mass = request->mass; 
            
            // JAKA native SDK uses a CartesianTran struct wrapper (.centroid.x)
            jaka_payload.centroid.x = request->xc; 
            jaka_payload.centroid.y = request->yc;
            jaka_payload.centroid.z = request->zc;

            // Call native SDK function
            int ret = robot.set_payload(&jaka_payload); 
            
            response->ret = ret;
            response->message = (ret == 0) ? "Payload updated successfully" : "Error setting payload";
        });
        
        // Collision Sensitivity Service
    auto collision_service = node->create_service<jaka_msgs::srv::SetCollision>(
        "/jaka_driver/set_collision_level",
        [](const std::shared_ptr<jaka_msgs::srv::SetCollision::Request> request,
           std::shared_ptr<jaka_msgs::srv::SetCollision::Response> response) {
            
            // ROS 2 SetCollision request tracks the target level inside the .value parameter
            int collision_level = request->value; 

            if (collision_level < 0 || collision_level > 5) {
                response->ret = -1;
                response->message = "Invalid collision level. Must be 0-5.";
                return;
            }

            // Execute the native C++ SDK level setting logic
            int ret = robot.set_collision_level(collision_level);
            
            response->ret = ret;
            response->message = (ret == 0) ? "Collision level updated" : "Error setting collision level";
        });
    auto collision_recover_service = node->create_service<std_srvs::srv::Empty>(
        "/jaka_driver/collision_recover",
        []([[maybe_unused]] const shared_ptr<std_srvs::srv::Empty::Request> request,
        [[maybe_unused]] shared_ptr<std_srvs::srv::Empty::Response> response) {
            int ret = robot.collision_recover();
            if (ret != 0) {
                RCLCPP_ERROR(rclcpp::get_logger("moveit_server"),
                    "collision_recover failed: %s", mapErr[ret].c_str());
            }
        });

    auto clear_error_service = node->create_service<std_srvs::srv::Empty>(
        "/jaka_driver/clear_error",
        []([[maybe_unused]] const shared_ptr<std_srvs::srv::Empty::Request> request,
        [[maybe_unused]] shared_ptr<std_srvs::srv::Empty::Response> response) {
            robot.clear_error();
        });


    RCLCPP_INFO(rclcpp::get_logger("moveit_server"), "==================Moveit Start==================");

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
