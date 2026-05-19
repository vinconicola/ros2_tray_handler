#!/usr/bin/env python3

import array
import sys

import cv2
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


class RealsenseRotatorNode(Node):
    def __init__(self) -> None:
        super().__init__('realsense_rotator')

        self.declare_parameter('input_image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_image_topic', '/camera/image')
        self.declare_parameter('input_depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('output_depth_topic', '/camera/depth_image')
        self.declare_parameter('input_camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_camera_info_topic', '/camera/camera_info')
        self.declare_parameter('input_points_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('output_points_topic', '/camera/points')
        self.declare_parameter('output_frame_id', 'camera_link')
        self.declare_parameter('restamp_with_node_clock', True)

        self.bridge = CvBridge()
        input_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        input_depth_topic = self.get_parameter('input_depth_topic').get_parameter_value().string_value
        output_depth_topic = self.get_parameter('output_depth_topic').get_parameter_value().string_value
        input_camera_info_topic = self.get_parameter('input_camera_info_topic').get_parameter_value().string_value
        output_camera_info_topic = self.get_parameter('output_camera_info_topic').get_parameter_value().string_value
        input_points_topic = self.get_parameter('input_points_topic').get_parameter_value().string_value
        output_points_topic = self.get_parameter('output_points_topic').get_parameter_value().string_value
        self.output_frame_id = self.get_parameter('output_frame_id').get_parameter_value().string_value
        self.restamp_with_node_clock = (
            self.get_parameter('restamp_with_node_clock').get_parameter_value().bool_value
        )

        self.image_pub = self.create_publisher(Image, output_image_topic, output_qos)
        self.depth_pub = self.create_publisher(Image, output_depth_topic, output_qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, output_camera_info_topic, output_qos)
        self.points_pub = self.create_publisher(PointCloud2, output_points_topic, output_qos)

        self.image_sub = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            input_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            input_depth_topic,
            self.depth_callback,
            input_qos,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            input_camera_info_topic,
            self.camera_info_callback,
            input_qos,
        )
        self.points_sub = self.create_subscription(
            PointCloud2,
            input_points_topic,
            self.points_callback,
            input_qos,
        )

        self.get_logger().info(
            f'Rotating image {input_image_topic} -> {output_image_topic} and '
            f'depth {input_depth_topic} -> {output_depth_topic}, '
            f'camera info {input_camera_info_topic} -> {output_camera_info_topic}, '
            f'point cloud {input_points_topic} -> {output_points_topic} '
            f'with output frame {self.output_frame_id}; '
            f'restamp_with_node_clock={self.restamp_with_node_clock}'
        )

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
        out_msg = self.bridge.cv2_to_imgmsg(rotated, encoding=msg.encoding)
        out_msg.header = msg.header
        if self.restamp_with_node_clock:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.output_frame_id
        self.image_pub.publish(out_msg)

    def depth_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
        out_msg = self.bridge.cv2_to_imgmsg(rotated, encoding=msg.encoding)
        out_msg.header = msg.header
        if self.restamp_with_node_clock:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.output_frame_id
        self.depth_pub.publish(out_msg)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        out_msg = CameraInfo()
        out_msg.header = msg.header
        if self.restamp_with_node_clock:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.output_frame_id
        out_msg.height = msg.height
        out_msg.width = msg.width
        out_msg.distortion_model = msg.distortion_model
        out_msg.d = list(msg.d)
        out_msg.k = list(msg.k)
        out_msg.r = list(msg.r)
        out_msg.p = list(msg.p)
        out_msg.binning_x = msg.binning_x
        out_msg.binning_y = msg.binning_y
        out_msg.roi = msg.roi

        if out_msg.width > 0 and out_msg.height > 0:
            out_msg.k[2] = (out_msg.width - 1) - out_msg.k[2]
            out_msg.k[5] = (out_msg.height - 1) - out_msg.k[5]
            out_msg.p[2] = (out_msg.width - 1) - out_msg.p[2]
            out_msg.p[6] = (out_msg.height - 1) - out_msg.p[6]

        self.camera_info_pub.publish(out_msg)

    def points_callback(self, msg: PointCloud2) -> None:
        reshape_cloud = msg.height > 1
        points = pc2.read_points(msg, skip_nans=False)

        if reshape_cloud:
            points = points.reshape((msg.height, msg.width))

        if not reshape_cloud:
            self.get_logger().warn(
                'Received unorganized point cloud; rotating XYZ only, image-to-cloud alignment may be incorrect.'
            )

        rotated = points.copy()

        if 'x' not in rotated.dtype.names or 'y' not in rotated.dtype.names:
            self.get_logger().warn('Point cloud is missing x/y fields; forwarding without rotation.')
        else:
            rotated['x'] *= -1.0
            rotated['y'] *= -1.0

        if (
            msg.header.frame_id == 'camera_depth_optical_frame'
            and self.output_frame_id == 'camera_link'
        ):
            x_optical = rotated['x'].copy()
            y_optical = rotated['y'].copy()
            z_optical = rotated['z'].copy()

            # Convert points from ROS optical frame axes to camera_link axes:
            # optical: x right, y down, z forward
            # link:    x forward, y left, z up
            rotated['x'] = z_optical
            rotated['y'] = -x_optical
            rotated['z'] = -y_optical

        if bool(sys.byteorder != 'little') != bool(msg.is_bigendian):
            rotated = rotated.byteswap().view(rotated.dtype.newbyteorder())

        out_msg = PointCloud2()
        out_msg.header = msg.header
        if self.restamp_with_node_clock:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.output_frame_id
        out_msg.height = msg.height
        out_msg.width = msg.width
        out_msg.fields = msg.fields
        out_msg.is_bigendian = msg.is_bigendian
        out_msg.point_step = msg.point_step
        out_msg.row_step = msg.row_step
        out_msg.is_dense = msg.is_dense

        buffer = array.array('B')
        buffer.frombytes(rotated.tobytes())
        out_msg.data = buffer

        self.points_pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealsenseRotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
