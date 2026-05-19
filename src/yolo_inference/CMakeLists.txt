cmake_minimum_required(VERSION 3.8)
project(yolo_inference)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/yolo_inference_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export the python folders if you have any modules
ament_python_install_package(${PROJECT_NAME})

ament_package()