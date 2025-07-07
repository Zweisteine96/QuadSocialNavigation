import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class ActorTracker(Node):
    def __init__(self):
        super().__init__('actor_tracker')
        
        actor_to_track = 'actor1'  # Make sure this name matches your actor's name
        # The topic name is now '/odom' within the actor's namespace
        topic_name = f'/model/{actor_to_track}/odom' 
        
        self.subscription = self.create_subscription(
            # 2. CHANGE THE MESSAGE TYPE
            Odometry,
            topic_name,
            self.pose_callback,
            10)
        self.get_logger().info(f"Tracker started for {actor_to_track} on topic {topic_name}")

    def pose_callback(self, msg: Odometry):
        # 3. ACCESS THE POSE CORRECTLY
        # The pose data is now inside msg.pose.pose
        pos = msg.pose.pose.position
        
        self.get_logger().info(f'Actor Position: [x: {pos.x:.2f}, y: {pos.y:.2f}]')
        
        # BONUS: You can also access the velocity!
        vel = msg.twist.twist.linear
        self.get_logger().info(f'Actor Velocity: [x: {vel.x:.2f}, y: {vel.y:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    actor_tracker = ActorTracker()
    rclpy.spin(actor_tracker)
    actor_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()