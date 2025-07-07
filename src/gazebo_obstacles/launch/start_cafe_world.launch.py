import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_gazebo_obstacles = get_package_share_directory('gazebo_obstacles')
    # Full path to the cafe world file you found in Step 1
    # IMPORTANT: Replace this path with the one you found on your system!
    world_path = os.path.join(pkg_gazebo_obstacles, 'worlds', 'cafe.world')

    # Launch Gazebo with the specified world
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        # Pass the world path as an argument to the included launch file
        launch_arguments={'world': world_path}.items()
    )

    return LaunchDescription([
        start_gazebo_cmd
    ])