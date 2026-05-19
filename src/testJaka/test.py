#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2

import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]

def main(args=None):
    rclpy.init(args=args)  # ← This must be called first

    node = Node('moveit2_example')  # Now safe to create the node

    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name="base_link",
        end_effector_name="Link_06",
        group_name="jaka_s12"
    )

    #moveit2.move_to_configuration(joint_positions=[0.0, 2.36, 0.0, 1.57, 0.0, 0.0])   
    
    
    moveit2.move_to_pose(
        position=[-0.50, 0.0, 1.0],
        quat_xyzw=euler_to_quaternion(1.57, 0, -1.57),
        frame_id="world",
        tolerance_position=0.001,         # Position tolerance
        tolerance_orientation=0.001,
        cartesian=False 
    )

    moveit2.wait_until_executed()

    return

if __name__ == '__main__':
    main()