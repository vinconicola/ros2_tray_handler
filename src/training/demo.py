#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import rclpy.duration
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_matrix, quaternion_from_matrix
import tf2_ros
import numpy as np
import time
DEBUG = True
BASE = [0.0, 2.5831, -1.8500, -0.7505, -1.5708, 0.0]
GRIPPER_OPEN = [0.07]
GRIPPER_CLOSED = [0.0]


TRAY_SLOT_BASE_HEIGHT = 0.15   # meters from ground — lowest slot
TRAY_SLOT_OFFSET      = 0.11   # meters between slots
NUM_SLOTS             = 10
APPROACH_DISTANCE     = 0.70   # meters in front of rack
APPROACH_HEIGHT       = 0.65   # meters from ground
TRAY_Y_TOLERANCE      = 0.02   # meters — tray must be within 2cm of rack Y center

# How long to wait at each scan position for TF detections
SCAN_WAIT    = 2   # seconds
# Minimum distance between two rack detections to count as different rack
RACK_DEDUP_DIST = 0.5  # meters


class TransformListenerNode(Node):
    def __init__(self):
        super().__init__('transform_listener')
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )
            return transform
        except tf2_ros.LookupException as e:
            self.get_logger().error(f'Transform lookup failed: {e}')
            return None

    def get_all_rack_transforms(self, max_racks=10):
        racks = []

        # frames = self.tf_buffer.all_frames_as_string()
        # self.get_logger().info(f'Known frames: {frames}')
        
        for i in range(max_racks):
            try:
                tf = self.tf_buffer.lookup_transform(
                    'world',
                    f'rack_{i}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                pos = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ])
                racks.append({
                    'id': i,
                    'frame': f'rack_{i}',
                    'position': pos,
                    'transform': tf
                })
                self.get_logger().info(f'Found rack_{i} at [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]')
            except Exception as e:
                self.get_logger().debug(f'rack_{i} not available: {e}')
                continue  # don't break — keep checking
        return racks

    def is_new_rack(self, position, known_racks, threshold=RACK_DEDUP_DIST):
        """Check if this position is far enough from all known racks (X,Y only)."""
        for rack in known_racks:
            dist = np.linalg.norm(position[:2] - rack['position'][:2])
            if dist < threshold:
                return False
        return True


def find_rack(jaka, tf_node):
    """
    Rotate the robot through a scan arc, collecting rack TF detections.
    Returns a list of unique rack positions in world frame.
    """
    tf_node.get_logger().info('Starting rack scan...')

    # Move to base position first
    jaka.move_to_configuration(BASE)
    jaka.wait_until_executed()

    found_racks = []
    state = list(BASE)
    step = 104
    if DEBUG: step = 627
    for angle_int in range(0, 628-step, step):
        angle = float(angle_int) / 100.0
        state[0] = angle
        tf_node.get_logger().info(f'Scanning at joint_0 = {angle:.2f} rad')

        jaka.move_to_configuration(joint_positions=state)
        jaka.wait_until_executed()

        # Wait for inference node to publish TFs
        time.sleep(SCAN_WAIT)

        # Spin longer to ensure TF buffer is populated
        end_time = time.time() + 2.0
        while time.time() < end_time:
            rclpy.spin_once(tf_node, timeout_sec=0.2)

        # Collect all currently visible rack TFs
        visible_racks = tf_node.get_all_rack_transforms()
        tf_node.get_logger().info(f'  Visible racks at this position: {len(visible_racks)}')

        for rack in visible_racks:
            pos = rack['position']
            if tf_node.is_new_rack(pos, found_racks):
                found_racks.append(rack)
                tf_node.get_logger().info(
                    f'  New rack found: {rack["frame"]} at '
                    f'[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]'
                )
            else:
                tf_node.get_logger().info(
                    f'  Duplicate rack skipped: {rack["frame"]}'
                )

    # Return to base
    jaka.move_to_configuration(BASE)
    jaka.wait_until_executed()

    tf_node.get_logger().info(
        f'Scan complete. Found {len(found_racks)} unique racks:'
    )
    for i, rack in enumerate(found_racks):
        pos = rack['position']
        tf_node.get_logger().info(
            f'  rack_{i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]'
        )

    return found_racks

def refine_rack_positions(jaka, tf_node, rough_racks):
    """
    For each rough rack position, rotate joint_1 to face it directly,
    then capture a refined TF from the front view.
    """
    refined_racks = []
    state = list(BASE)

    for rack in rough_racks:
        pos = rack['position']
        tf_node.get_logger().info(
            f'Refining {rack["frame"]} at [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]'
        )

        # ── Step 1: compute joint_1 angle to face the rack ────────────────────
        # rack position is in world frame, base_link is at world origin
        # joint_1 rotates around Z axis of base_link
        # angle = atan2(rack_y, rack_x) in base_link frame
        try:
            # Get base_link → world transform to convert rack pos to base_link frame
            tf_base = tf_node.tf_buffer.lookup_transform(
                'base_link', 'world',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            from tf_transformations import quaternion_matrix
            q = tf_base.transform.rotation
            mat = quaternion_matrix([q.x, q.y, q.z, q.w])
            rack_world = np.array([pos[0], pos[1], pos[2], 1.0])
            rack_base  = mat @ rack_world
            rack_base[0] += tf_base.transform.translation.x
            rack_base[1] += tf_base.transform.translation.y
            rack_base[2] += tf_base.transform.translation.z
        except Exception as e:
            tf_node.get_logger().warn(f'base_link transform failed, using world pos: {e}')
            rack_base = np.array([pos[0], pos[1], pos[2], 1.0])

        # Angle to face rack from joint_1
        angle = np.arctan2(rack_base[1], rack_base[0])
        tf_node.get_logger().info(f'  Facing angle: {np.degrees(angle):.1f} deg')

        # ── Step 2: move to face the rack ─────────────────────────────────────
        state[0] = float(angle)
        jaka.move_to_configuration(joint_positions=state)
        jaka.wait_until_executed()

        # Wait for inference node to stabilize
        time.sleep(3.0)
        end_time = time.time() + 1.0
        while time.time() < end_time:
            rclpy.spin_once(tf_node, timeout_sec=0.1)

        # ── Step 3: capture refined TF ────────────────────────────────────────
        # Find the rack_N closest to the rough position
        best_tf   = None
        best_dist = float('inf')

        for i in range(10):
            try:
                tf = tf_node.tf_buffer.lookup_transform(
                    'world', f'rack_{i}',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                rp = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                ])
                dist = np.linalg.norm(rp - pos[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_tf   = tf
            except Exception:
                continue

        if best_tf is None or best_dist > 1.0:
            tf_node.get_logger().warn(
                f'  Could not find refined TF for {rack["frame"]} (best_dist={best_dist:.2f})'
            )
            # Keep rough position as fallback
            refined_racks.append(rack)
            continue

        refined_pos = np.array([
            best_tf.transform.translation.x,
            best_tf.transform.translation.y,
            best_tf.transform.translation.z,
        ])

        tf_node.get_logger().info(
            f'  Refined position: [{refined_pos[0]:.3f},{refined_pos[1]:.3f},{refined_pos[2]:.3f}]'
            f'  (delta={best_dist:.3f}m from rough)'
        )

        refined_racks.append({
            'id':        rack['id'],
            'frame':     rack['frame'],
            'position':  refined_pos,
            'transform': best_tf,
        })

    # Return to base
    jaka.move_to_configuration(BASE)
    jaka.wait_until_executed()

    tf_node.get_logger().info(f'Refinement complete. {len(refined_racks)} racks refined.')
    return refined_racks

def face_rack_slot(jaka, tf_node, rack, slot_index):
    log = tf_node.get_logger()
    
    # 1. Get Rack Position in World
    pos_w = np.array([
        rack['transform'].transform.translation.x,
        rack['transform'].transform.translation.y,
        rack['transform'].transform.translation.z
    ])
    
    # 2. Get Rack Normal (X-axis of the rack frame)
    from tf_transformations import quaternion_matrix, quaternion_from_matrix
    q_w = rack['transform'].transform.rotation
    mat_w = quaternion_matrix([q_w.x, q_w.y, q_w.z, q_w.w])
    
    # This is the vector pointing OUT of the rack face
    rack_normal = mat_w[:3, 0] 
    rack_normal /= np.linalg.norm(rack_normal)

    # 3. Calculate Slot Position (The "Goal")
    # Height Z is calculated based on slot index
    target_z_w = TRAY_SLOT_BASE_HEIGHT + (slot_index * TRAY_SLOT_OFFSET)
    slot_pos_w = np.array([pos_w[0], pos_w[1], target_z_w])

    # 4. Calculate Approach Position (The "Start of Linear Path")
    # FIX: We move ALONG the normal (toward the robot) for the approach
    # If the rack is at X=0.76 and normal is [1, 0, 0], 
    # the approach should be at X=0.76 + 0.20 = 0.96 (if robot is at X > 1.0)
    # OR X=0.76 - 0.20 = 0.56 (if robot is at X=0)
    
    # We'll use a 20cm offset. 
    # We check the robot's position relative to the rack to ensure we stay on the "outside"
    approach_pos_w = slot_pos_w + (rack_normal * 0.20) 

    # --- 5. Orientation (TCP Z points into the rack, Gripper Roll Adjusted) ---
    z_axis = -rack_normal # Keep pointing into the rack
    world_up = np.array([0, 0, 1])
    
    # OLD LOGIC (Resulted in -90 or +90 degree offset):
    # y_axis = np.cross(world_up, z_axis)
    
    # NEW LOGIC: 
    # To rotate 90 degrees around Z, we make the X-axis horizontal instead of the Y-axis
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rot_mat = np.eye(4)
    rot_mat[:3, 0] = x_axis
    rot_mat[:3, 1] = y_axis
    rot_mat[:3, 2] = z_axis
    target_quat_w = quaternion_from_matrix(rot_mat)

    # --- LOGS FOR VERIFICATION ---
    log.info(f"VERIFICATION: Target X={slot_pos_w[0]:.3f}, Approach X={approach_pos_w[0]:.3f}")
    
    # 6. EXECUTE
    # Move to approach
    jaka.move_to_pose(position=approach_pos_w, quat_xyzw=target_quat_w, frame_id='world')
    
    if not jaka.wait_until_executed():
        log.error("Approach failed. Check if Approach X is between Robot and Rack.")
        return False

    # Linear Insertion
    jaka.move_to_pose(position=slot_pos_w, quat_xyzw=target_quat_w, frame_id='world', cartesian=True)
    return jaka.wait_until_executed()

def main(args=None):
    rclpy.init(args=args)

    node        = Node('jacka')
    nodeG       = Node('gripper')
    tf_node     = TransformListenerNode()

    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    jaka = MoveIt2(
        node=node,
        joint_names=joint_names,
        base_link_name='base_link',
        end_effector_name='tcp',
        group_name='jaka_s12'
    )
    gripper = MoveIt2(
        node=nodeG,
        joint_names=['joint_7'],
        base_link_name='Gripper01',
        end_effector_name='Gripper02',
        group_name='gripper'
    )

    
    
    rough_racks   = find_rack(jaka, tf_node)
    refined_racks = refine_rack_positions(jaka, tf_node, rough_racks)

    # tf_node.get_logger().info('Final rack positions:')
    # for rack in refined_racks:
    #     pos = rack['position']
    #     tf_node.get_logger().info(
    #         f'  {rack["frame"]}: [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]'
    #     )

    # for rack in refined_racks:
    #     occupancy = get_rack_tray_occupancy(jaka, tf_node, rack)
    #     rack['occupancy'] = occupancy
    #     tf_node.get_logger().info(
    #         f'{rack["frame"]} occupancy: {occupancy}'
    #     )
    # Test if the robot can move 10cm up from where it is right now
    # test_quat_corrected = [0.5, 0.5, 0.5, 0.5] 

    # jaka.move_to_pose(
    #     position=[0.5, 0.0, 0.4], 
    #     quat_xyzw=test_quat_corrected, 
    #     frame_id='base_link'
        
        
    # )
    # jaka.wait_until_executed()
    face_rack_slot(jaka, tf_node, refined_racks[0], 8)

    tf_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()