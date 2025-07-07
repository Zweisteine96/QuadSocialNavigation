from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gazebo_obstacles'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    #packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'models', 'moving_cylinder'), glob('models/moving_cylinder/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chguo',
    maintainer_email='gcleon96@gmail.com',
    description='Trajectory Prediction with ROS2 Navigation',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_mover = gazebo_obstacles.scripts.obstacle_mover:main',
            'actor_tracker = gazebo_obstacles.scripts.actor_tracker:main',
            'trajectory_plotter = gazebo_obstacles.scripts.trajectory_plotter:main',
            'prediction_node = gazebo_obstacles.scripts.prediction_node:main',
            'cv_prediction_node = gazebo_obstacles.scripts.cv_prediction_node:main'
        ],
    },
)
