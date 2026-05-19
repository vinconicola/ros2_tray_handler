import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import open3d_conversions as o3c
import numpy as np

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('pcd_subs_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points', # Sostituisci con il tuo topic
            self.listener_callback,
            10)
        
        # Inizializza la finestra di visualizzazione di Open3D
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Open3D ROS2 Viewer")
        self.pcd = o3d.geometry.PointCloud()
        self.first_view = True
        # Create a timer to update the visualizer at ~30 FPS
        self.timer = self.create_timer(0.033, self.vis_callback)

    def listener_callback(self, msg):
        #print("msg")
        points = o3c.from_msg(msg)
        self.pcd =  points        
        
        
    def vis_callback(self):
        #print("render")
        if not self.first_view:
            self.vis.clear_geometries()
        
        if self.first_view:
            self.vis.add_geometry(self.pcd)
            self.first_view = False
        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        # Update geometry and render
        self.vis.poll_events()
        self.vis.update_renderer()
        


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.vis.destroy_window()
    rclpy.shutdown()

if __name__ == '__main__':
    main()