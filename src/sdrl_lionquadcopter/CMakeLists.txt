cmake_minimum_required(VERSION 3.8)
project(sdrl_lionquadcopter)

# Compile options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# ROS 2 dependencies
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# SDFormat (Harmonic ships sdformat14)
find_package(sdformat14 REQUIRED)
set(SDFORMAT_INCLUDE_DIRS ${sdformat14_INCLUDE_DIRS})
set(SDFORMAT_LIBRARIES ${sdformat14_LIBRARIES})

# Install headers and library
install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)

# Gazebo Harmonic libs
# read https://gazebosim.org/docs/harmonic/release-features/#harmonic

find_package(gz-sim8 QUIET)
set(GZ_SIM_TARGET gz-sim8)

find_package(gz-math7 REQUIRED)
set(GZ_MATH_TARGET gz-math7)

find_package(gz-msgs10 QUIET)
set(GZ_MSGS_TARGET gz-msgs10)

find_package(gz-transport13 QUIET)
set(GZ_TRANSPORT_TARGET gz-transport13)

add_library(lion_quadcopter_plugin SHARED
  src/lionquadcopter.cpp
  include/sdrl_lionquadcopter/lionquadcopter.h
)

target_compile_features(lion_quadcopter_plugin PUBLIC cxx_std_17)

# Link ROS 2 dependencies
ament_target_dependencies(lion_quadcopter_plugin
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  std_srvs
)

target_include_directories(lion_quadcopter_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${SDFORMAT_INCLUDE_DIRS}
)

# Link Gazebo libs
target_link_libraries(lion_quadcopter_plugin
  ${GZ_SIM_TARGET}::${GZ_SIM_TARGET}
  ${GZ_MATH_TARGET}::${GZ_MATH_TARGET}
  ${GZ_MSGS_TARGET}::${GZ_MSGS_TARGET}
  ${GZ_TRANSPORT_TARGET}::${GZ_TRANSPORT_TARGET}
  ${SDFORMAT_LIBRARIES}
)

install(TARGETS lion_quadcopter_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install resources so model:// and package URIs resolve
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models)

ament_package()
