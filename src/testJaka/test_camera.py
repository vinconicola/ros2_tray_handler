# Basic ROS 2 program to subscribe to real-time streaming video from a topic
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # Subscribe to the image topic (e.g., 'video_frames' or '/camera/image_raw')
        self.subscription = self.create_subscription(
            Image,
            '/camera/segmentation/labels_map',  # Replace with your topic name
            self.listener_callback,
            10
        )
        self.br = CvBridge()  # Bridge to convert ROS Image to OpenCV

    def listener_callback(self, data):
        #self.get_logger().info('Receiving video frame')
        try:
            # Convert ROS Image message to OpenCV image
            current_frame = self.br.imgmsg_to_cv2(data, "mono8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        print(current_frame)
        # Display the image using OpenCV
        cv2.imshow("Camera Feed", current_frame)
        cv2.waitKey(1)  # Required to refresh the window

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()