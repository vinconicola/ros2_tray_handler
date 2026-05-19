from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Support both the original parameter names and the names used elsewhere.
        DeclareLaunchArgument('ip', default_value='192.168.0.75', description='Robot controller IP address'),
        DeclareLaunchArgument('model', default_value='s12', description='Robot model suffix used by MoveIt controller names'),
        DeclareLaunchArgument('robot_ip', default_value='', description='Optional alias for ip'),
        DeclareLaunchArgument('robot_name', default_value='', description='Optional alias for model'),

        LogInfo(msg=['moveit_server ip: ', LaunchConfiguration('ip')]),
        LogInfo(msg=['moveit_server model: ', LaunchConfiguration('model')]),
        LogInfo(msg=['moveit_server robot_ip alias: ', LaunchConfiguration('robot_ip')]),
        LogInfo(msg=['moveit_server robot_name alias: ', LaunchConfiguration('robot_name')]),

        # Launch the 'moveit_server' node from the 'jaka_planner' package
        Node(
            package='jaka_planner',
            executable='moveit_server',
            name='moveit_server',
            output='screen',
            parameters=[
                {'ip': LaunchConfiguration('ip')},
                {'model': LaunchConfiguration('model')},
                {'robot_ip': LaunchConfiguration('robot_ip')},
                {'robot_name': LaunchConfiguration('robot_name')},
            ],
        ),
    ])
