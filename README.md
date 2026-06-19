# Tray handler per Jaka s12
## Setup
- Container ROS2 humble
- Install and follow tutorial JAKA_ROS https://www.jaka.com/docs/en/guide/ROS/ROS2.html
- File launches must be copied here /opt/ros/humble/lib/python3.10/site-packages/moveit_configs_utils
  - download gz_ros2_control
  - modify path around line 737 - 750 of file launches to match
- Copy sdf worlds here /usr/share/ignition/ignition-gazebo6/worlds/   /usr/share/gz/gz-sim7/worlds (depends on the version of gazebo the JAKA tutorial uses ignition in this repo is used gz7)
  - adjust the .dae reference inside the sdf file
- Add this to bashrc
  source /opt/ros/humble/setup.bash
  
	source <path_to_ws>/install/setup.bash

	source <path_to_ws>/install/local_setup.bash

	export LIBGL_ALWAYS_SOFTWARE=true

	export DISPLAY=unix:1

	export GZ_VERSION=garden

	export LD_LIBRARY_PATH=<path_to_ws>/install/gz_ros2_control/lib:$LD_LIBRARY_PATH

	export GZ_SIM_SYSTEM_PLUGIN_PATH=<path_to_ws>/install/gz_ros2_control/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH
-Install requirements.txt (pip) and apt_packages.txt (apt)

## Notes
To use gazebo data (images and clock) you can run gazebo_bridge.sh (adjust based on which version of gazebo you are using)

## Quick start
### Sim
- ros2 launch jaka_s12_moveit_config gazebo.launch.py
- - gazebo simulation
- bash gazebo_bridge.sh
- - gazebo -> ros topic bridge
- ros2 launch jaka_s12_moveit_config moveit_rviz.launch.py
- - rviz
- ros2 run yolo_inference yolo_inference_node.py
- - yolo inference node
- ros2 launch jaka_movement tray_handler.launch.py --ros-args --remap use_sim_time:=true
- - movement logic
### Real
- ros2 launch camera_d435_rotator d435_rotated.launch.py
- - camera publishing node
- ros2 run yolo_inference yolo_inference_node.py
- - yolo inference node
- ros2 launch jaka_planner moveit_server.launch.py ip:=192.168.0.75 model:=s12
- - moveit server 
- ros2 launch jaka_s12_moveit_config demo.launch.py
- - rviz + various moveit nodes
- ros2 launch jaka_movement tray_handler.launch.py
-  - movement logic
### Utils
- ros2 service call /yolo_inference/trigger_detection std_srvs/srv/Trigger "{}" 
- - yolo scan
- ros2 service call /jaka_driver/set_io jaka_msgs/srv/SetIO "{signal: 'digital', type: 0, index: 7, value: 1.0}"
- - gripper open - close (value 0 - 1)
