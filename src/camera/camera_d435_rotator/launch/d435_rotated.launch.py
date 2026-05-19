from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    serial_no_arg = DeclareLaunchArgument(
        'serial_no',
        default_value="''",
        description='Optional RealSense serial number.',
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py',
            ])
        ),
        launch_arguments={
            'serial_no': LaunchConfiguration('serial_no'),
            'camera_namespace': 'camera',
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'true',
            'rgb_camera.color_profile': '640x480x15',
            'depth_module.depth_profile': '640x480x15',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'pointcloud.ordered_pc': 'true',
        }.items(),
    )

    rotator_node = Node(
        package='camera_d435_rotator',
        executable='realsense_rotator_node.py',
        name='realsense_rotator',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_image_topic': '/camera/camera/color/image_raw',
            'output_image_topic': '/camera/image',
            'input_depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'output_depth_topic': '/camera/depth_image',
            'input_camera_info_topic': '/camera/camera/color/camera_info',
            'output_camera_info_topic': '/camera/camera_info',
            'input_points_topic': '/camera/camera/depth/color/points',
            'output_points_topic': '/camera/points',
            'output_frame_id': 'camera_link',
            'restamp_with_node_clock': True,
        }],
    )

    return LaunchDescription([
        serial_no_arg,
        realsense_launch,
        rotator_node,
    ])
