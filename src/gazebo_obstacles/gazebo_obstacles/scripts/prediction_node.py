import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class PredictionNode(Node):
    r"""Linear Kalman Filter with a constant velocity model."""
    def __init__(self):
        super().__init__('prediction_node')
        
        # --- Configuration ---
        actor_to_track = 'actor1'
        topic_name = f'/model/{actor_to_track}/odom'
        
        # --- Kalman Filter State ---
        # State vector: [x_pos, y_pos, x_vel, y_vel]
        self.x = np.zeros((4, 1)) 
        # State Covariance Matrix: How uncertain we are about the state. Start high.
        self.P = np.eye(4) * 500. 
        # State Transition Matrix: Defines how the state evolves. Set up in callback.
        self.F = np.eye(4) 
        # Measurement Matrix: Maps state to measurement space. We only measure position.
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) 
        # Measurement Noise Covariance: How much we trust our sensor.
        self.R = np.eye(2) * 0.5 
        # Process Noise Covariance: How much we trust our "constant velocity" model.
        self.Q = np.eye(4) 
        
        self.last_time = None
        
        # --- Data Storage for Plotting ---
        self.measured_trajectory = {'x': [], 'y': []}
        self.filtered_trajectory = {'x': [], 'y': []}
        self.predicted_trajectory = {'x': [], 'y': []}
        self.prediction_time_horizon = 2.0 # seconds into the future
        
        # --- Matplotlib Setup ---
        self.fig, self.ax = plt.subplots()
        self.line_measured, = self.ax.plot([], [], 'b.', label='Measured')
        self.line_filtered, = self.ax.plot([], [], 'g-', label='Filtered (Kalman)', linewidth=2)
        self.line_predicted, = self.ax.plot([], [], 'm--', label='Predicted')
        self.ax.set_title('Kalman Filter Trajectory Prediction')
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
        self.get_logger().info(f"Prediction node started for '{actor_to_track}'")

    def odom_callback(self, msg: Odometry):
        current_time = self.get_clock().now()
        
        if self.last_time is None:
            self.last_time = current_time
            # Initialize state with the first measurement
            self.x[0] = msg.pose.pose.position.x
            self.x[1] = msg.pose.pose.position.y
            return

        # --- Time Step Calculation ---
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # --- Kalman Filter Predict Step ---
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Kalman Filter Update Step ---
        measurement_z = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y]])
        y = measurement_z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # --- Store Data for Plotting ---
        self.measured_trajectory['x'].append(measurement_z[0, 0])
        self.measured_trajectory['y'].append(measurement_z[1, 0])
        self.filtered_trajectory['x'].append(self.x[0, 0])
        self.filtered_trajectory['y'].append(self.x[1, 0])
        
        # --- Predict Future Trajectory ---
        self.predict_future()

    def predict_future(self):
        """Use the current filtered state to project forward in time."""
        pred_x = self.x.copy()
        pred_traj_x = [pred_x[0, 0]]
        pred_traj_y = [pred_x[1, 0]]
        
        # Simulate forward for the time horizon
        num_steps = 20
        dt = self.prediction_time_horizon / num_steps
        
        pred_F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
                           
        for _ in range(num_steps):
            pred_x = pred_F @ pred_x
            pred_traj_x.append(pred_x[0, 0])
            pred_traj_y.append(pred_x[1, 0])
            
        self.predicted_trajectory['x'] = pred_traj_x
        self.predicted_trajectory['y'] = pred_traj_y

    def update_plot(self, frame):
        """Update the plot for animation."""
        self.line_measured.set_data(self.measured_trajectory['x'], self.measured_trajectory['y'])
        self.line_filtered.set_data(self.filtered_trajectory['x'], self.filtered_trajectory['y'])
        self.line_predicted.set_data(self.predicted_trajectory['x'], self.predicted_trajectory['y'])
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line_measured, self.line_filtered, self.line_predicted,

def main(args=None):
    rclpy.init(args=args)
    prediction_node = PredictionNode()
    ros_thread = threading.Thread(target=rclpy.spin, args=(prediction_node,), daemon=True)
    ros_thread.start()
    ani = FuncAnimation(prediction_node.fig, prediction_node.update_plot, interval=100, blit=False)
    plt.show()
    prediction_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()