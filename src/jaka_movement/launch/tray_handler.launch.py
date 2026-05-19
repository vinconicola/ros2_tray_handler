# tray_handler.launch.py
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("jaka_s12", package_name="jaka_s12_moveit_config")
        .to_moveit_configs()
    )

    kinematics_yaml_path = os.path.join(
        get_package_share_directory("jaka_s12_moveit_config"),
        "config", "kinematics.yaml"
    )
    with open(kinematics_yaml_path, "r") as f:
        kinematics_yaml = yaml.safe_load(f)
    kinematics_params = {"robot_description_kinematics": kinematics_yaml}

    # Load cartesian limits for PILZ
    # cartesian_limits_path = os.path.join(
    #     get_package_share_directory("jaka_s12_moveit_config"),
    #     "config", "pilz_cartesian_limits.yaml"
    # )
    # with open(cartesian_limits_path, "r") as f:
    #     cartesian_limits_raw = yaml.safe_load(f)

    # # Get the moveit config dict and merge cartesian limits into it
    # moveit_config_dict = moveit_config.to_dict()
    
    # # Merge cartesian limits into robot_description_planning
    # if "robot_description_planning" in moveit_config_dict:
    #     moveit_config_dict["robot_description_planning"].update(
    #         cartesian_limits_raw["robot_description_planning"]
    #     )
    # else:
    #     moveit_config_dict.update(cartesian_limits_raw)

    tray_handler_node = Node(
        package="jaka_movement",
        executable="tray_handler_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            kinematics_params,
            {"use_sim_time": True},
            {"robot_description_planning.cartesian_limits.max_trans_vel": 1.0},
            {"robot_description_planning.cartesian_limits.max_trans_acc": 2.25},
            {"robot_description_planning.cartesian_limits.max_trans_dec": -5.0},
            {"robot_description_planning.cartesian_limits.max_rot_vel": 1.57},
        ],
    )

    return LaunchDescription([tray_handler_node])