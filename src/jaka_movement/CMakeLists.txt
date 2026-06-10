cmake_minimum_required(VERSION 3.8)
project(jaka_movement)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_core REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(jaka_msgs REQUIRED)

add_executable(tray_handler_node src/tray_handler.cpp)

# Let ROS 2 handle the include paths and library linking automatically
ament_target_dependencies(tray_handler_node
  rclcpp
  moveit_ros_planning_interface
  moveit_ros_planning
  moveit_core
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  geometry_msgs
  std_srvs
  jaka_msgs
)

# Manual include for Eigen since it's not a pure ament package
target_include_directories(tray_handler_node PRIVATE
  ${Eigen3_INCLUDE_DIRS}
)

install(TARGETS tray_handler_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()