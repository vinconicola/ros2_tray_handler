from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("jaka_s12", package_name="jaka_s12_moveit_config")
        .robot_description(mappings={"use_rviz_sim": "true"})
        .to_moveit_configs()
    )

    package_share = moveit_config.package_path
    ld = LaunchDescription()

    static_virtual_joints = package_share / "launch" / "static_virtual_joint_tfs.launch.py"
    if static_virtual_joints.exists():
        ld.add_action(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(static_virtual_joints))
            )
        )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_share / "launch" / "rsp.launch.py"))
        )
    )
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_share / "launch" / "move_group.launch.py"))
        )
    )
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_share / "launch" / "moveit_rviz.launch.py"))
        )
    )
    ld.add_action(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                moveit_config.robot_description,
                str(package_share / "config" / "ros2_controllers.yaml"),
                {"use_sim_time": False},
            ],
            output="screen",
        )
    )
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_share / "launch" / "spawn_controllers.launch.py")
            )
        )
    )

    return ld
