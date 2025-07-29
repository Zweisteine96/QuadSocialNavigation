cmake_minimum_required(VERSION 3.8)
project(social_pipeline_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find all dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(social_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

# NOTE: Find LibTorch
set(Torch_DIR ${CMAKE_SOURCE_DIR}/../../libtorch)
find_package(Torch REQUIRED)

# Node 1: The Publisher
add_executable(agent_publisher_node src/agent_publisher_node.cpp)
ament_target_dependencies(agent_publisher_node rclcpp social_msgs)

# Node 2: The Processor
add_executable(state_processor_node src/state_processor_node.cpp src/predictor.cpp)
ament_target_dependencies(state_processor_node rclcpp social_msgs nav_msgs)
#target_link_libraries(state_processor_node PRIVATE ${TORCH_LIBRARIES}) # link to LibTorch libraries

# Node 3: The Visualizer
add_executable(visualizer_node src/visualizer_node.cpp)
ament_target_dependencies(visualizer_node rclcpp social_msgs nav_msgs)
#target_link_libraries(visualizer_node PRIVATE sfml-graphics sfml-window sfml-system) # link to SFML libraries

# Specify Include Directories for our own headers
target_include_directories(state_processor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  #${ament_index_cpp_INCLUDE_DIRS}
  #${rclcpp_INCLUDE_DIRS}
  #${social_msgs_INCLUDE_DIRS}
  #${nav_msgs_INCLUDE_DIRS}
)

target_link_libraries(
  state_processor_node 
  ament_index_cpp::ament_index_cpp
  #rclcpp::rclcpp
  #social_msgs::social_msgs
  #nav_msgs::nav_msgs
  ${TORCH_LIBRARIES}
)

target_link_libraries(
  visualizer_node
  sfml-graphics 
  sfml-window 
  sfml-system
)

# Install the compiled programs
install(TARGETS
  agent_publisher_node
  state_processor_node
  visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# NOTE: Install the dataset file
install(DIRECTORY
  data/
  DESTINATION share/${PROJECT_NAME}/data
)

# NOTE: Install the model file
install(DIRECTORY
  model/
  DESTINATION share/${PROJECT_NAME}/model
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
