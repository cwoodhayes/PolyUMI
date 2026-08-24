cmake_minimum_required(VERSION 3.8)
project(polyumi_fr3_controllers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(Eigen3 REQUIRED)
# franka_hand_node (and the gated probes) talk to the hand directly. Franka is a plain CMake
# package (install/libfranka/lib/cmake/Franka), not an ament one, so it is linked by target rather
# than carried in DEPENDENCIES via ament_target_dependencies.
find_package(Franka REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(trajectory_msgs REQUIRED)

set(DEPENDENCIES
  controller_interface
  franka_msgs
  franka_semantic_components
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  realtime_tools
  tf2_eigen
  tf2_ros
  trajectory_msgs
)

# The maths (interpolator + control law) is a separate library from the ros2_control wrapper so the
# tests can link it without pulling in a hardware interface. That split is the whole reason the law
# is unit-testable at all.
add_library(polyumi_fr3_control_core SHARED
  src/pose_trajectory_interpolator.cpp
  src/cartesian_impedance_law.cpp
  src/gripper_trajectory_interpolator.cpp
)
target_include_directories(polyumi_fr3_control_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(polyumi_fr3_control_core PUBLIC Eigen3::Eigen)

add_library(polyumi_fr3_controllers SHARED
  src/cartesian_impedance_controller.cpp
)
target_include_directories(polyumi_fr3_controllers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(polyumi_fr3_controllers PUBLIC polyumi_fr3_control_core)
ament_target_dependencies(polyumi_fr3_controllers PUBLIC ${DEPENDENCIES})

pluginlib_export_plugin_description_file(controller_interface polyumi_fr3_controllers.xml)

# The production hand driver: owns the libfranka gripper connection outright, so franka_gripper
# must NOT be running (load_gripper:=false). See the file header for why the action server is gone.
add_executable(franka_hand_node src/franka_hand_node.cpp)
target_link_libraries(franka_hand_node polyumi_fr3_control_core Franka::Franka)
ament_target_dependencies(franka_hand_node rclcpp sensor_msgs trajectory_msgs)

# Firmware bringup probes: one-off characterisations of what a Move costs with no ROS, no action
# server and no shared executor in the way. Deliberately not nodes — the point is that nothing
# else shares their threads. Nothing in service loads them and they have already produced their
# answer (HandLimits), so they are off by default; turn them on to characterise a different hand.
#
#   colcon build --packages-select polyumi_fr3_controllers --cmake-args -DBUILD_HAND_PROBES=ON
option(BUILD_HAND_PROBES "Build the franka_hand_testing characterisation probes" OFF)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS franka_hand_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_HAND_PROBES)
  set(HAND_PROBES
    gripper_timing_probe
    simple_gripper_timing_probe
    csv_gripper_timing_probe
    stop_gripper_timing_probe
    speed_gripper_timing_probe
    maxspeed_gripper_timing_probe
    stroke_gripper_timing_probe
  )
  foreach(probe ${HAND_PROBES})
    add_executable(${probe} src/franka_hand_testing/${probe}.cpp)
    target_link_libraries(${probe} Franka::Franka)
  endforeach()
  install(TARGETS ${HAND_PROBES} DESTINATION lib/${PROJECT_NAME})
endif()

install(
  TARGETS polyumi_fr3_control_core polyumi_fr3_controllers
  EXPORT export_polyumi_fr3_controllers
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_pose_trajectory_interpolator test/test_pose_trajectory_interpolator.cpp)
  target_link_libraries(test_pose_trajectory_interpolator polyumi_fr3_control_core)

  ament_add_gtest(test_cartesian_impedance_law test/test_cartesian_impedance_law.cpp)
  target_link_libraries(test_cartesian_impedance_law polyumi_fr3_control_core)

  ament_add_gtest(test_gripper_trajectory_interpolator
    test/test_gripper_trajectory_interpolator.cpp)
  target_link_libraries(test_gripper_trajectory_interpolator polyumi_fr3_control_core)
endif()

ament_export_include_directories(include)
ament_export_libraries(polyumi_fr3_control_core polyumi_fr3_controllers)
ament_export_dependencies(${DEPENDENCIES})
ament_package()
