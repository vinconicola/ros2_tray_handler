#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # Define the subscription topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10 # Queue size
        )
        
        self.bridge = CvBridge()
        
        # Setup output directory
        self.output_dir = os.path.expanduser('/home/nicola/ros2_ws/src/training/real_world/dataset2/images')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Throttling calculations for 8 FPS (1 second / 8 = 0.125 seconds)
        self.target_interval = 0.125  
        self.last_saved_time = 0.0
        self.image_counter = 0
        
        self.get_logger().info(f'Image Saver Node initialized. Saving 8 FPS to: {self.output_dir}')

    def image_callback(self, msg):
        current_time = time.time()
        
        # Check if enough time has passed since the last saved image
        if (current_time - self.last_saved_time) >= self.target_interval:
            try:
                # Convert ROS 2 Image message to an OpenCV image
                # Change "bgr8" to "mono8" if working with grayscale cameras
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                # Generate unique filename using timestamp and a sequential counter
                filename = os.path.join(self.output_dir, f'img_{self.image_counter:04d}_{int(current_time)}.png')
                
                # Save the image frame
                cv2.imwrite(filename, cv_image)
                
                # Update tracking variables
                self.last_saved_time = current_time
                self.image_counter += 1
                
                self.get_logger().info(f'Saved frame: {filename}')
                
            except Exception as e:
                self.get_logger().error(f'Failed to convert or save image frame: {str(e)}')

def main(argc=None):
    rclpy.init(args=argc)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()