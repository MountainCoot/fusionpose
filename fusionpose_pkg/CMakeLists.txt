cmake_minimum_required(VERSION 3.5)
project(fusionpose_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  vicon2gt
)

catkin_python_setup()

# Message files
add_message_files(
  FILES
  NodeStatus.msg              
)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp roslib rospy rosbag vicon2gt
)
