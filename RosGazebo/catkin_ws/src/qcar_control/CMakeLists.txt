cmake_minimum_required(VERSION 3.0.2)
project(qcar_control)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  TrajectoryMessage.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs nav_msgs message_runtime
)

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  src/mpc_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  # myfile1
  # myfile2
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)