import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')
        
        # --- Configuration ---
        actor_to_track = 'actor1'
        topic_name = f'/model/{actor_to_track}/odom'
        
        # Data storage for the trajectory
        self.x_data = []
        self.y_data = []
        
        # --- Matplotlib Setup ---
        # Create the figure and axes for the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', label=f'{actor_to_track} Trajectory') # 'r-' is a red line
        
        # Configure plot aesthetics
        self.ax.set_title('Real-Time Actor Trajectory')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box') # Very important for correct aspect ratio
        
        # --- ROS Subscriber ---
        self.subscription = self.create_subscription(
            Odometry,
            topic_name,
            self.odom_callback,
            10)
        self.get_logger().info(f"Plotter started for '{actor_to_track}' on topic '{topic_name}'")

    def odom_callback(self, msg: Odometry):
        """This callback is executed every time a message is received."""
        self.get_logger().info('Received an odometry message!')
        # Extract position data and append it to our lists
        pos = msg.pose.pose.position
        self.x_data.append(pos.x)
        self.y_data.append(pos.y)

    def update_plot(self, frame):
        """This function is called by FuncAnimation to update the plot."""
        # Update the line data
        self.line.set_data(self.x_data, self.y_data)
        
        # Re-compute the plot limits to fit the new data
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Return the artists that were modified
        return self.line,

def main(args=None):
    rclpy.init(args=args)
    
    plotter_node = TrajectoryPlotter()
    
    # --- Threading to run ROS spin and Matplotlib show simultaneously ---
    # rclpy.spin() and plt.show() are both blocking functions.
    # We need to run the ROS node's spin in a separate thread.
    ros_thread = threading.Thread(target=rclpy.spin, args=(plotter_node,), daemon=True)
    ros_thread.start()
    
    # Create the animation. It will call update_plot every 100ms
    ani = FuncAnimation(plotter_node.fig, plotter_node.update_plot, interval=100, blit=True)
    
    # plt.show() will block the main thread until the plot window is closed.
    plt.show()
    
    # Cleanup
    plotter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()