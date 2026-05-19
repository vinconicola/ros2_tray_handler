import sys
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')  # ROS1 path if available

# Try reading directly with rosbags
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_types_from_msg, register_types, Stores
import cv2
import numpy as np
import os

out_dir = '/home/nicola/ros2_ws/data/real_world/images'
os.makedirs(out_dir, exist_ok=True)

bag_files = [
    '/home/nicola/ros2_ws/data/bag/20260401_093235.bag',
    '/home/nicola/ros2_ws/data/bag/20260401_094240.bag',
    '/home/nicola/ros2_ws/data/bag/20260401_094433.bag',
]

saved = 0
for bag_path in bag_files:
    print(f'Processing {bag_path}')
    try:
        with Reader(bag_path) as reader:
            # List available topics
            for conn in reader.connections:
                print(f'  Topic: {conn.topic} Type: {conn.msgtype}')
    except Exception as e:
        print(f'  Error: {e}')