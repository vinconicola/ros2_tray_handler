# Tray handler per Jaka s12
## Setup
- Container ROS2 humble
- install and follow tutorial JAKA_ROS https://www.jaka.com/docs/en/guide/ROS/ROS2.html
- file launches must be copied here /opt/ros/humble/lib/python3.10/site-packages/moveit_configs_utils
  - download gz_ros2_control
  - modify path around line 737 - 750 of file launches to match
- copy sdf worlds here /usr/share/ignition/ignition-gazebo6/worlds/   /usr/share/gz/gz-sim7/worlds (depends on the version of gazebo the JAKA tutorial uses ignition in this repo is used gz7)
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
-install requirements.txt (pip) and apt_packages.txt (apt)

## notes
to use gazebo data (images and clock) you can run gazebo_bridge.sh (adjust based on which version of gazebo you are using)

