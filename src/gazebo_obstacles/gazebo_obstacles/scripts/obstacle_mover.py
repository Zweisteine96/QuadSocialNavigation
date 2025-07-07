#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion
import math

class ObstacleMover(Node):
    def __init__(self):
        super().__init__('obstacle_mover')
        self.model_name = 'moving_cylinder'
        
        # Create a client for the Gazebo service to set entity state
        self.client = self.create_client(SetEntityState, '/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /set_entity_state not available, waiting...')

        # Timer to call the update function periodically
        self.timer = self.create_timer(0.05, self.update_obstacle_pose) # 20 Hz update
        self.start_time = self.get_clock().now()
        
        # --- DEFINE YOUR TRAJECTORY PARAMETERS HERE ---
        self.trajectory_type = 'circle' # 'circle' or 'line'
        self.center_x = 2.0
        self.center_y = 0.0
        self.radius = 2.0
        self.angular_velocity = 0.4  # rad/s for circle
        self.linear_velocity = 0.5   # m/s for line

    def get_trajectory_pose(self, elapsed_seconds):
        """
        Calculates the desired pose of the obstacle at a given time.
        This is where you define your moving pattern.
        """
        if self.trajectory_type == 'circle':
            angle = self.angular_velocity * elapsed_seconds
            x = self.center_x + self.radius * math.cos(angle)
            y = self.center_y + self.radius * math.sin(angle)
        elif self.trajectory_type == 'line':
            # Moves back and forth on the Y-axis
            period = 8.0 # seconds to go from -2 to 2 and back
            cycle = math.fmod(elapsed_seconds, period)
            if cycle < period / 2.0:
                y = -2.0 + self.linear_velocity * 2 * cycle
            else:
                y = 2.0 - self.linear_velocity * 2 * (cycle - period / 2.0)
            x = 3.0
        else:
            # Default to staying still
            x, y = 3.5, -2.0

        # Create the pose message
        pose = Pose()
        pose.position = Point(x=x, y=y, z=0.0)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def update_obstacle_pose(self):
        elapsed_time = self.get_clock().now() - self.start_time
        elapsed_seconds = elapsed_time.nanoseconds / 1e9

        new_pose = self.get_trajectory_pose(elapsed_seconds)

        # Prepare the service request
        state_msg = EntityState()
        state_msg.name = self.model_name
        state_msg.pose = new_pose
        state_msg.reference_frame = 'world'

        req = SetEntityState.Request()
        req.state = state_msg

        # Call the service
        self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    obstacle_mover = ObstacleMover()
    rclpy.spin(obstacle_mover)
    obstacle_mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()