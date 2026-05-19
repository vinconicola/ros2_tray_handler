from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("jaka_s12", package_name="jaka_s12_moveit_config").to_moveit_configs()

    package_share = FindPackageShare("jaka_s12_moveit_config")
    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "rsp.launch.py"])
        )
    )
    static_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "static_virtual_joint_tfs.launch.py"])
        )
    )
    move_group_launch = generate_move_group_launch(moveit_config)

    return LaunchDescription([
        rsp_launch,
        static_tf_launch,
        move_group_launch,
    ])
