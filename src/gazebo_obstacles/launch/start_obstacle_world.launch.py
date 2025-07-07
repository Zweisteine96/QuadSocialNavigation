import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_gazebo_obstacles = get_package_share_directory('gazebo_obstacles')

    # Start Gazebo server
    start_gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': os.path.join(pkg_gazebo_obstacles, 'worlds', 'obstacle_world.world')}.items()
    )

    # Start Gazebo client
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    # Spawn the moving cylinder
    spawn_cylinder_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'moving_cylinder',
            '-file', os.path.join(pkg_gazebo_obstacles, 'models', 'moving_cylinder', 'model.sdf'),
            '-x', '3.5',
            '-y', '-2.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Start the obstacle mover script
    obstacle_mover_node = Node(
        package='gazebo_obstacles',
        executable='obstacle_mover', # This matches the name in setup.py
        name='obstacle_mover',
        output='screen'
    )

    delayed_obstacle_mover_launch = TimerAction(
        period=5.0,
        actions=[obstacle_mover_node]
    )

    return LaunchDescription([
        #start_gazebo_server_cmd,
        #start_gazebo_client_cmd,
        start_gazebo_cmd,
        spawn_cylinder_node,
        #obstacle_mover_node,
        delayed_obstacle_mover_launch
    ])