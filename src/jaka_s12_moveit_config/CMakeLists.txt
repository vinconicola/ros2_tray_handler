cmake_minimum_required(VERSION 3.22)
project(jaka_s12_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)

add_executable(sim_effort_estimator_node src/sim_effort_estimator.cpp)
ament_target_dependencies(sim_effort_estimator_node
  rclcpp
  sensor_msgs
  control_msgs
)

install(TARGETS sim_effort_estimator_node
  DESTINATION lib/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../test.sdf")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../test.sdf" DESTINATION share/${PROJECT_NAME}/worlds)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Learning.sdf")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../Learning.sdf" DESTINATION share/${PROJECT_NAME}/worlds)
endif()

ament_package()
