from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "60"],
        output="screen",
    )

    jaka_s12_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["jaka_s12_controller", "--controller-manager-timeout", "60"],
        output="screen",
    )

    gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager-timeout", "60"],
        output="screen",
    )

    return LaunchDescription(
        [
            joint_state_broadcaster,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster,
                    on_exit=[jaka_s12_controller],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=jaka_s12_controller,
                    on_exit=[gripper_controller],
                )
            ),
        ]
    )
