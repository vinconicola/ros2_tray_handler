from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_gazebo_launch
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("jaka_s12", package_name="jaka_s12_moveit_config").to_moveit_configs()
    launch_description = generate_gazebo_launch(moveit_config)

    launch_description.add_action(
        Node(
            package="jaka_s12_moveit_config",
            executable="sim_effort_estimator_node",
            name="sim_effort_estimator",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "input_joint_states_topic": "/joint_state_broadcaster/joint_states",
                "output_joint_states_topic": "/joint_states",
                "controller_state_topic": "/jaka_s12_controller/controller_state",
                "p_gains": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "i_gains": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "d_gains": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "i_clamps": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "effort_deadband": 0.05,
                "max_effort": 500.0,
                "smoothing_alpha": 0.35,
            }],
        )
    )

    return launch_description