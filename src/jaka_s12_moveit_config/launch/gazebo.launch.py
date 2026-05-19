import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def generate_launch_description():
    package_name = "jaka_s12_moveit_config"
    robot_name = "jaka_s12"

    moveit_config = (
        MoveItConfigsBuilder(robot_name, package_name=package_name)
        .robot_description(mappings={"use_gazebo": "true"})
        .to_moveit_configs()
    )

    package_share = moveit_config.package_path
    world_file = package_share / "worlds" / "test.sdf"
    gz_ros2_control_prefix = get_package_prefix("gz_ros2_control")
    jaka_description_share = get_package_share_directory("jaka_description")
    gz_ros2_control_lib = os.path.join(gz_ros2_control_prefix, "lib")

    os.environ["GZ_VERSION"] = "garden"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "entity",
            default_value=robot_name,
            description="Robot entity name",
        )
    )
    ld.add_action(
        DeclareBooleanLaunchArg(
            "use_gazebo",
            default_value=True,
            description="Use Gazebo simulation mode",
        )
    )

    static_virtual_joints = package_share / "launch" / "static_virtual_joint_tfs.launch.py"
    if static_virtual_joints.exists():
        ld.add_action(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(static_virtual_joints))
            )
        )

    ld.add_action(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                {"publish_frequency": 15.0},
                {"use_sim_time": True},
            ],
        )
    )

    ld.add_action(
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            jaka_description_share,
        )
    )
    ld.add_action(
        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            gz_ros2_control_lib + ":" + os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
        )
    )
    ld.add_action(
        SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            gz_ros2_control_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        )
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
            ),
            launch_arguments={
                "gz_args": f"-r -v 4 {world_file}",
                "gz_version": "7",
                "ign_version": "",
            }.items(),
        )
    )

    ld.add_action(
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_robot",
                    output="screen",
                    arguments=[
                        "-world",
                        "learning",
                        "-topic",
                        "/robot_description",
                        "-name",
                        robot_name,
                        "-z",
                        "0.06",
                    ],
                )
            ],
        )
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package_share / "launch" / "move_group_gazebo.launch.py")),
            launch_arguments={"allow_trajectory_execution": "true"}.items(),
        )
    )

    ld.add_action(
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="yolo_inference",
                    executable="yolo_inference_node.py",
                    name="yolo_inference",
                    output="screen",
                    parameters=[{"use_sim_time": True}],
                )
            ],
        )
    )

    return ld
