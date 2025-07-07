import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class CVPredictionNode(Node):
    def __init__(self):
        super().__init__('cv_prediction_node')
        
        # --- Configuration ---
        actor_to_track = 'actor1'
        topic_name = f'/model/{actor_to_track}/odom'
        
        # --- Constant Velocity Model State ---
        # We only need to store the last known position and time
        self.last_pos = None
        self.last_time = None
        self.current_velocity = {'x': 0.0, 'y': 0.0}
        
        # --- Data Storage for Plotting ---
        self.measured_trajectory = {'x': [], 'y': []}
        self.predicted_trajectory = {'x': [], 'y': []}
        self.prediction_time_horizon = 2.0 # seconds into the future
        
        # --- Matplotlib Setup ---
        self.fig, self.ax = plt.subplots()
        self.line_measured, = self.ax.plot([], [], 'b.', label='Measured')
        self.line_predicted, = self.ax.plot([], [], 'c--', label='CV Predicted') # Cyan dashed line
        self.ax.set_title('Constant Velocity Trajectory Prediction')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        
        # --- ROS Subscriber (with QoS for Gazebo) ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            Odometry, topic_name, self.odom_callback, qos_profile)
        self.get_logger().info(f"CV Prediction node started for '{actor_to_track}'")

    def odom_callback(self, msg: Odometry):
        current_time = self.get_clock().now()
        current_pos = msg.pose.pose.position
        
        # --- Velocity Estimation ---
        # If we have a previous measurement, estimate the velocity
        if self.last_pos is not None and self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 1e-6: # Avoid division by zero
                self.current_velocity['x'] = (current_pos.x - self.last_pos.x) / dt
                self.current_velocity['y'] = (current_pos.y - self.last_pos.y) / dt

        # Update the last known state for the next callback
        self.last_pos = current_pos
        self.last_time = current_time
        
        # --- Store Data for Plotting ---
        self.measured_trajectory['x'].append(current_pos.x)
        self.measured_trajectory['y'].append(current_pos.y)
        
        # --- Predict Future Trajectory ---
        self.predict_future(current_pos)

    def predict_future(self, start_pos):
        """Use the current estimated velocity to project forward in time."""
        pred_traj_x = [start_pos.x]
        pred_traj_y = [start_pos.y]
        
        num_steps = 20
        dt = self.prediction_time_horizon / num_steps
        
        # Simple extrapolation loop
        for i in range(1, num_steps + 1):
            next_x = start_pos.x + self.current_velocity['x'] * (i * dt)
            next_y = start_pos.y + self.current_velocity['y'] * (i * dt)
            pred_traj_x.append(next_x)
            pred_traj_y.append(next_y)
            
        self.predicted_trajectory['x'] = pred_traj_x
        self.predicted_trajectory['y'] = pred_traj_y

    def update_plot(self, frame):
        """Update the plot for animation."""
        self.line_measured.set_data(self.measured_trajectory['x'], self.measured_trajectory['y'])
        self.line_predicted.set_data(self.predicted_trajectory['x'], self.predicted_trajectory['y'])
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line_measured, self.line_predicted,

def main(args=None):
    rclpy.init(args=args)
    cv_prediction_node = CVPredictionNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(cv_prediction_node,), daemon=True)
    ros_thread.start()
    ani = FuncAnimation(cv_prediction_node.fig, cv_prediction_node.update_plot, interval=100, blit=False)
    plt.show()
    cv_prediction_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()