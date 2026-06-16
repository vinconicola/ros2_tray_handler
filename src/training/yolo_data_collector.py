#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json
from pathlib import Path
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Label map: pixel value -> class id (0-indexed for YOLO, 0=tray, 1=rack)
LABEL_MAP = {
    100: 0,  # tray
    101: 1,  # rack
}
CLASS_NAMES = ['tray', 'rack']

class YoloDataCollector(Node):
    def __init__(self):
        super().__init__('yolo_data_collector')

        self.declare_parameter('output_dir', 'data/yolo_dataset')
        self.declare_parameter('save_interval', 5)  # save every N frames
        self.declare_parameter('max_frames', 2000)

        self.output_dir = self.get_parameter('output_dir').value
        self.save_interval = self.get_parameter('save_interval').value
        self.max_frames = self.get_parameter('max_frames').value

        # Create dataset directories
        for split in ['train', 'val']:
            Path(f'{self.output_dir}/images/{split}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.output_dir}/labels/{split}').mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.saved_count = 0

        # Synchronized subscribers
        self.rgb_sub   = Subscriber(self, Image, '/camera/image')
        self.label_sub = Subscriber(self, Image, '/camera/segmentation/labels_map')

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.label_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.callback)
        self.get_logger().info(f'Collecting data to {self.output_dir}')
        self.get_logger().info(f'Saving every {self.save_interval} frames, max {self.max_frames}')

    def labels_to_yolo(self, label, img_h, img_w):
        annotations = []
        for pixel_val, class_id in LABEL_MAP.items():
            # label is already the R channel (single value per pixel)
            mask = (label == pixel_val).astype(np.uint8)  # already 2D
            if mask.sum() < 100:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 200:
                    continue
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 3:
                    continue
                points = approx.reshape(-1, 2).astype(float)
                points[:, 0] /= img_w
                points[:, 1] /= img_h
                coords = ' '.join([f'{x:.6f} {y:.6f}' for x, y in points])
                annotations.append(f'{class_id} {coords}')

        return annotations

    def callback(self, rgb_msg, label_msg):
        if self.saved_count >= self.max_frames:
            self.get_logger().info(f'Reached {self.max_frames} frames, stopping.')
            rclpy.shutdown()
            return

        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return

        # Convert images
        rgb   = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        label = self.bridge.imgmsg_to_cv2(label_msg, 'rgb8')
        label = np.array(label)[:, :, 0]  # R channel only → 2D array
        img_h, img_w = rgb.shape[:2]

        # Check if frame has any useful labels
        has_tray = (label == 100).sum() > 100
        has_rack = (label == 101).sum() > 100
        if not has_tray and not has_rack:
            return  # skip background-only frames

        # Generate YOLO annotations
        annotations = self.labels_to_yolo(label, img_h, img_w)
        if not annotations:
            return

        # 80/20 train/val split
        split = 'val' if self.saved_count % 5 == 0 else 'train'
        frame_id = f'{self.saved_count:06d}'

        # Save image
        img_path = f'{self.output_dir}/images/{split}/{frame_id}.jpg'
        cv2.imwrite(img_path, rgb)

        # Save label
        lbl_path = f'{self.output_dir}/labels/{split}/{frame_id}.txt'
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(annotations))

        self.saved_count += 1
        if self.saved_count % 50 == 0:
            self.get_logger().info(f'Saved {self.saved_count}/{self.max_frames} frames')

    def save_yaml(self):
        yaml_content = f"""path: {self.output_dir}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
        with open(f'{self.output_dir}/dataset.yaml', 'w') as f:
            f.write(yaml_content)
        self.get_logger().info(f'Saved dataset.yaml')


def main():
    rclpy.init()
    node = YoloDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_yaml()
        node.get_logger().info(f'Done. Saved {node.saved_count} frames.')
        node.destroy_node()

if __name__ == '__main__':
    main()