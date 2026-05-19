#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
import time

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

    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    moveit2 = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name="base_link",
        end_effector_name="Link_06",
        group_name="jaka_s12"
    )

    
    states = [
        [0., 1.57, 0.0, 3.14, 3.50, -1.57],
        [0., 1.57, 1., 3.8, 1.57, 0.],
        [0., 1.64, 1.88, 2.74, 1.57, 0.],
        [0., 0.57, 2.48, 3.22, 1.57, 0.],
        [0., 1.2, 2.41, 2.46, 1.57, 0.],
        [0., 1.5, 2.63, 1.81, 1.57, 0.]
    ]
    #states = [[0., 1.57, 0.0, 3.14, 3.50, -1.57], [0., 0.57, 2.48, 3.22, 1.57, 0.], [0., 1.2, 2.41, 2.46, 1.57, 0.]]

    for state in states:
        for i in range(0, 628, 20):
            state[0] = float(i/100)
            print(states)
            moveit2.move_to_configuration(joint_positions=state)
            moveit2.wait_until_executed()

    # moveit2.move_to_configuration(joint_positions=[0., 1.57, 1., 3.8, 1.57, 0.])
    # moveit2.wait_until_executed()
    # moveit2.move_to_configuration(joint_positions=[0., 1.64, 1.88, 2.74, 1.57, 0.])
    # moveit2.wait_until_executed()
    # moveit2.move_to_configuration(joint_positions=[0., 0.57, 2.48, 3.22, 1.57, 0.])
    # moveit2.wait_until_executed()
    # moveit2.move_to_configuration(joint_positions=[0., 1.2, 2.41, 2.46, 1.57, 0.])
    # moveit2.wait_until_executed()
    # moveit2.move_to_configuration(joint_positions=[0., 1.5, 2.63, 1.81, 1.57, 0.])
    # moveit2.wait_until_executed()

    print(moveit2.joint_state.position[1])

    return

if __name__ == '__main__':
    main()