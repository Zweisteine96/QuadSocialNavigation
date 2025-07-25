import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the share directory for your project and for nav2_bringup
    human_nav_project_dir = get_package_share_directory('human_nav_project')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # --- Use simple Python variables for file paths ---
    map_yaml_file = os.path.join(human_nav_project_dir, 'maps', 'turtlebot3_world.yaml')
    nav2_params_file = os.path.join(human_nav_project_dir, 'params', 'nav2_params.yaml')
    rviz_config_file = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # --- Launch Actions ---
    # 1. Launch Gazebo with your custom world and the robot
    start_world_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(human_nav_project_dir, 'launch', 'start_dynamic_world.launch.py'))
    )

    # 2. Launch the core Nav2 stack
    start_nav2_bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_yaml_file,
            'params_file': nav2_params_file,
        }.items(),
    )
    
    # 3. Launch RViz with a concrete path
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file], # Now this is a string, which is correct
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )


    # Create the launch description and add the actions in order
    ld = LaunchDescription()
    
    ld.add_action(start_world_cmd)
    ld.add_action(start_nav2_bringup_cmd)
    ld.add_action(start_rviz_cmd)

    return ld