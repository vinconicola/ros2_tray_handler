#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger  # <--- Added this
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from ultralytics import YOLO
import cv2
from scipy import ndimage
from rclpy.time import Time


IMG_W = 640
IMG_H = 480

RACK_WIDTH = 0.46  # meters from center of left and right arm

CLASS_NAMES  = {0: 'tray', 1: 'rack'}
CLASS_COLORS = {
    0: (255, 180, 100),  # tray  → orange
    1: (180, 100, 255),  # rack  → purple
}

MODEL_PATH = '/home/nicola/ros2_ws/src/yolo_inference/weights/best.pt'

# Filtering parameters
MIN_CLUSTER_POINTS = 100    # ignore tiny detections
MAX_DEPTH          = 8.0    # ignore points further than 8m
MIN_DEPTH          = 0.3    # ignore points closer than 0.3m


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference')

        self.bridge  = CvBridge()
        self.model   = YOLO(MODEL_PATH)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # --- Snapshot Logic ---
        self.detect_requested = False 
        self.srv = self.create_service(Trigger, '~/trigger_detection', self.handle_trigger)
        # ----------------------

        self.rgb_sub = Subscriber(self, Image, '/camera/image')
        self.depth_sub = Subscriber(self, Image, '/camera/depth_image')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/camera_info')
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.callback)

        self.labeled_pub = self.create_publisher(PointCloud2, '/labeled_points', 10)
        self.vis_pub     = self.create_publisher(Image, '/yolo/visualization', 10)
        
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.world_frame = 'world'
        
        self.get_logger().info('YOLO Node ready in SNAPSHOT mode. Call ~/trigger_detection to scan.')

    def handle_trigger(self, request, response):
        """Service callback to enable detection for the next available frame."""
        self.detect_requested = True
        self.get_logger().info('Detection triggered! Waiting for next sync frame...')
        response.success = True
        response.message = "YOLO scan scheduled for next frame."
        return response

    def transform_points_to_world(self, points_xyz, source_frame, stamp):
        for lookup_time in [Time(), stamp]:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    source_frame,
                    lookup_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                break
            except Exception as e:
                last_error = e
        else:
            self.get_logger().warn(f'TF lookup failed: {last_error}', throttle_duration_sec=5.0)
            return None

        q  = t.transform.rotation
        tr = t.transform.translation

        x, y, z, w = q.x, q.y, q.z, q.w
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-6:
            self.get_logger().warn('Degenerate quaternion in TF, skipping frame')
            return None
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tr.x, tr.y, tr.z]

        pts = points_xyz.astype(np.float64)

        # Replace NaN/inf with 0 just for the transform — invalid points stay
        # identifiable downstream because their XYZ will be (tr.x, tr.y, tr.z),
        # which won't pass any distance filter
        valid_mask = np.isfinite(pts).all(axis=1)
        pts[~valid_mask] = 0.0

        ones  = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pts_w = (T @ pts_h.T).T[:, :3]

        # Restore NaN for originally invalid points so filters still reject them
        pts_w[~valid_mask] = np.nan

        return pts_w.astype(np.float32)

    def filter_cluster(self, points, std_threshold=1, apply_statistical=True):
        if len(points) == 0:
            return points

        # Depth filter
        # depths = points[:, 0]
        # mask = (depths > MIN_DEPTH) & (depths < MAX_DEPTH)
        # points = points[mask]
        # if len(points) < MIN_CLUSTER_POINTS:
        #     return np.array([])

        # Floor filter — remove bottom 5% of Z values
        z_floor = np.percentile(points[:, 2], 5)
        points = points[points[:, 2] > z_floor]
        if len(points) < MIN_CLUSTER_POINTS:
            return np.array([])

        if not apply_statistical:
            return points

        # Statistical outlier filter
        mean = points.mean(axis=0)
        std  = points.std(axis=0)
        inliers = np.all(np.abs(points - mean) < std_threshold * std, axis=1)
        return points[inliers]

    def get_surface_with_normal(self, points, label='object'):
        """Get front vertical face center with horizontal normal for each stacked tray layer."""
        if len(points) < 50:
            return []

        # ── Closest dense cluster in XY distance (world coords) ──────────────────
        xy_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

        hist, bin_edges = np.histogram(xy_dist, bins=20)
        threshold = hist.max() * 0.05
        first_bin = next((i for i, h in enumerate(hist) if h > threshold), None)
        if first_bin is None:
            return []

        cluster_end = first_bin
        for i in range(first_bin, len(hist)):
            if hist[i] > threshold:
                cluster_end = i
            else:
                if all(h <= threshold for h in hist[i:i+3]):
                    break

        dist_min = bin_edges[first_bin]
        dist_max = bin_edges[cluster_end + 1]
        cluster  = points[(xy_dist >= dist_min) & (xy_dist <= dist_max)]

        if len(cluster) < 30:
            return []

        # ── Split into individual trays by Z (stacked, min 10cm apart) ───────────
        z_vals      = cluster[:, 2]
        z_hist, z_edges = np.histogram(z_vals, bins=40)
        z_bin_width = z_edges[1] - z_edges[0]

        min_separation_bins = max(1, int(0.10 / z_bin_width))
        peak_bins = []
        for i, h in enumerate(z_hist):
            if h < z_hist.max() * 0.1:
                continue
            if not peak_bins or (i - peak_bins[-1]) >= min_separation_bins:
                peak_bins.append(i)
            elif z_hist[i] > z_hist[peak_bins[-1]]:
                peak_bins[-1] = i

        if not peak_bins:
            return []

        results = []
        for pb in peak_bins:
            z_peak = (z_edges[pb] + z_edges[pb + 1]) / 2.0
            layer  = cluster[np.abs(cluster[:, 2] - z_peak) < 2 * z_bin_width]

            if len(layer) < 20:
                continue

            layer_xy_dist   = np.sqrt(layer[:, 0]**2 + layer[:, 1]**2)
            front_threshold = np.percentile(layer_xy_dist, 20)
            front_points    = layer[layer_xy_dist <= front_threshold]

            if len(front_points) < 10:
                continue

            centroid     = layer.mean(axis=0).copy()
            centroid[:2] = front_points[:, :2].mean(axis=0)

            centered = front_points - front_points.mean(axis=0)
            cov      = np.cov(centered.T)
            _, eigenvectors = np.linalg.eigh(cov)
            normal   = eigenvectors[:, 0].copy()
            normal[2] = 0.0
            norm_len  = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue
            normal = normal / norm_len

            if np.dot(normal, centroid) > 0:
                normal = -normal

            results.append((centroid, normal))

        return results

    def get_rack_opening(self, points, label='rack'):
        """Find rack center and normal. Works for any rack orientation around the robot."""
        if len(points) < 50:
            return None, None

        # ── Step 1: Remove floor points ──────────────────────────────────────────
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        cluster = points[points[:, 2] >= z_min + (z_max - z_min) * 0.30]
        if len(cluster) < 30:
            return None, None

        # ── Step 2: Keep only the nearest 40% in XY distance from robot ──────────
        #This isolates the front face and arm tips regardless of rack orientation
        xy_dist = np.linalg.norm(cluster[:, :2], axis=1)
        xy_min  = xy_dist.min()
        xy_max  = xy_dist.max()
        cluster = cluster[xy_dist <= xy_min + (xy_max - xy_min) * 0.4]
        if len(cluster) < 20:
            return None, None

        # ── Step 3: PCA in XY to find the rack face direction ────────────────────
        pts_2d   = cluster[:, :2]
        centered = pts_2d - pts_2d.mean(axis=0)
        cov      = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)

        vec_large = vecs[:, np.argmax(vals)].copy()  # largest variance
        vec_small = vecs[:, np.argmin(vals)].copy()  # smallest variance

        # depth_vec should point FROM rack TOWARD robot — check both candidates
        # The vector pointing toward robot has a negative dot with rack_mean_xy
        rack_mean_xy = pts_2d.mean(axis=0)
        rack_dir     = -rack_mean_xy / (np.linalg.norm(rack_mean_xy) + 1e-6)  # unit vector toward robot

        # Check which eigenvector is closer to pointing toward the robot (<40°)
        cos_thresh = np.cos(np.radians(40))

        dot_large = abs(np.dot(vec_large, rack_dir))
        dot_small = abs(np.dot(vec_small, rack_dir))

        if dot_small >= cos_thresh or dot_small > dot_large:
            # Smallest variance vector points toward robot — normal use case
            depth_vec = vec_small
            face_vec  = vec_large
        else:
            # Largest variance vector is closer to robot direction — side view case
            self.get_logger().warn('PCA: swapping face/depth vectors — rack may be viewed from side')
            depth_vec = vec_large
            face_vec  = vec_small

        # Ensure depth_vec points toward robot
        if np.dot(depth_vec, rack_dir) < 0:
            depth_vec = -depth_vec

        # face_vec handedness: ensure face_vec x depth_vec points up (world Z+)
        cross_z = face_vec[0] * depth_vec[1] - face_vec[1] * depth_vec[0]
        if cross_z < 0:
            face_vec = -face_vec

        # ── Step 4: Project all points onto (face_vec, depth_vec) local axes ─────
        face_proj  = cluster[:, :2] @ face_vec   # position along rack width
        depth_proj = cluster[:, :2] @ depth_vec  # distance toward robot

        # ── Step 5: Isolate front face — keep only the nearest 20% in depth ──────
        depth_threshold = np.percentile(depth_proj, 80)  # high depth_proj = closer to robot
        front_mask   = depth_proj >= depth_threshold
        front_points = cluster[front_mask]
        front_face   = face_proj[front_mask]

        if len(front_points) < 10:
            return None, None

        # ── Step 6: Split arms by face projection (left/right along rack width) ───
        face_mid  = (front_face.min() + front_face.max()) / 2.0
        left_mask  = front_face <  face_mid
        right_mask = front_face >= face_mid

        left_arm  = front_points[left_mask]
        right_arm = front_points[right_mask]

        if len(left_arm) < 5 or len(right_arm) < 5:
            return None, None

        # ── Step 7: Find the true tip of each arm ────────────────────────────────
        # Tip = closest to robot = highest depth_proj value
        left_depth  = depth_proj[front_mask][left_mask]
        right_depth = depth_proj[front_mask][right_mask]

        left_tip_thresh  = np.percentile(left_depth,  80)
        right_tip_thresh = np.percentile(right_depth, 80)

        left_tip_pts  = left_arm[left_depth   >= left_tip_thresh]
        right_tip_pts = right_arm[right_depth >= right_tip_thresh]

        if len(left_tip_pts) < 2 or len(right_tip_pts) < 2:
            return None, None

        left_tip  = left_tip_pts.mean(axis=0)
        right_tip = right_tip_pts.mean(axis=0)

        # ── Step 8: Centroid in rack local frame ──────────────────────────────────
        left_tip  = left_tip_pts.mean(axis=0)
        right_tip = right_tip_pts.mean(axis=0)

        # Project both tips onto face_vec to get their position along rack width
        left_face_proj  = np.dot(left_tip[:2],  face_vec)
        right_face_proj = np.dot(right_tip[:2], face_vec)

        # Project both tips onto depth_vec to get their distance toward robot
        left_depth_proj  = np.dot(left_tip[:2],  depth_vec)
        right_depth_proj = np.dot(right_tip[:2], depth_vec)

        # Center along face = midpoint of the two arms along rack width
        center_face_proj  = (left_face_proj + right_face_proj) / 2.0

        # Center in depth = closest tip (the front face of the rack)
        center_depth_proj = max(left_depth_proj, right_depth_proj)

        # Reconstruct world XY from local rack frame
        center_xy = center_face_proj * face_vec + center_depth_proj * depth_vec

        x_center = center_xy[0]
        y_center = center_xy[1]
        z_center = cluster[:, 2].mean()
        centroid = np.array([x_center, y_center, z_center])

        # ── Step 9: Normal = depth_vec (already points toward robot) ─────────────
        normal = np.array([depth_vec[0], depth_vec[1], 0.0])

        return centroid, normal

    def is_duplicate(self, new_pos, existing_positions, threshold=0.25):
        """Check if the new position is within 'threshold' meters of any existing detection."""
        for pos in existing_positions:
            dist = np.linalg.norm(new_pos - pos)
            if dist < threshold:
                return True
        return False

    def publish_tf_with_normal(self, position, normal, frame_name, parent_frame, stamp, static=False):
        """Publish TF with orientation aligned to surface normal."""
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id  = frame_name

        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])

        # Compute rotation from X axis to -normal direction
        x_axis = np.array([1.0, 0.0, 0.0])
        target = normal
        target = target / np.linalg.norm(target)

        v = np.cross(x_axis, target)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, target)

        if s < 1e-6:
            if c > 0:
                q = np.array([0.0, 0.0, 0.0, 1.0])  # already aligned
            else:
                q = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg around Z
        else:
            vx = np.array([
                [ 0,    -v[2],  v[1]],
                [ v[2],  0,    -v[0]],
                [-v[1],  v[0],  0   ]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

            # Rotation matrix → quaternion (Shepperd's method)
            trace = R[0,0] + R[1,1] + R[2,2]
            if trace > 0:
                sq = 0.5 / np.sqrt(trace + 1.0)
                w  =  0.25 / sq
                x  = (R[2,1] - R[1,2]) * sq
                y  = (R[0,2] - R[2,0]) * sq
                z  = (R[1,0] - R[0,1]) * sq
            elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                sq = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w  = (R[2,1] - R[1,2]) / sq
                x  =  0.25 * sq
                y  = (R[0,1] + R[1,0]) / sq
                z  = (R[0,2] + R[2,0]) / sq
            elif R[1,1] > R[2,2]:
                sq = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w  = (R[0,2] - R[2,0]) / sq
                x  = (R[0,1] + R[1,0]) / sq
                y  =  0.25 * sq
                z  = (R[1,2] + R[2,1]) / sq
            else:
                sq = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w  = (R[1,0] - R[0,1]) / sq
                x  = (R[0,2] + R[2,0]) / sq
                y  = (R[1,2] + R[2,1]) / sq
                z  =  0.25 * sq

            q = np.array([x, y, z, w])

        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        if static:
            self.static_tf_broadcaster.sendTransform(t)
        else:
            self.tf_broadcaster.sendTransform(t)

    def get_instances(self, points_xyz, label_mask, cls_id, mask_binary, std_threshold=1.0):
        """Split a class mask into individual instances using connected components."""
        # Erode mask to remove border pixels that bleed onto background
        # Fill small holes first (Closing)
        kernel = np.ones((3, 3), np.uint8)
        mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        
        # Then erode to clean the edges
        mask_eroded = cv2.erode(mask_closed, kernel, iterations=1)
        
        labeled_array, num_features = ndimage.label(mask_eroded)
        instances = []
        for i in range(1, num_features + 1):
            instance_pixels = (labeled_array == i).reshape(-1)
            instance_points = points_xyz[instance_pixels]
            valid = np.isfinite(instance_points).all(axis=1)
            instance_points = instance_points[valid]
            instance_points = self.filter_cluster(instance_points, std_threshold=1.5)
            if len(instance_points) >= MIN_CLUSTER_POINTS:
                instances.append(instance_points)
        instances.sort(key=lambda p: p[:, 0].mean())
        return instances

    def build_points_from_depth(self, depth_msg, camera_info_msg, image_shape):
        img_h, img_w = image_shape[:2]
        if camera_info_msg.width != img_w or camera_info_msg.height != img_h:
            self.get_logger().warn(
                f'Camera info/image size mismatch: info={camera_info_msg.width}x{camera_info_msg.height}, '
                f'image={img_w}x{img_h}. Skipping snapshot.',
                throttle_duration_sec=5.0
            )
            return None, None

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        if depth.shape[:2] != (img_h, img_w):
            self.get_logger().warn(
                f'Depth/image size mismatch: depth={depth.shape[1]}x{depth.shape[0]}, '
                f'image={img_w}x{img_h}. Skipping snapshot.',
                throttle_duration_sec=5.0
            )
            return None, None

        depth = depth.astype(np.float32)
        if depth_msg.encoding == '16UC1':
            depth *= 0.001

        fx = camera_info_msg.k[0]
        fy = camera_info_msg.k[4]
        cx = camera_info_msg.k[2]
        cy = camera_info_msg.k[5]
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn('Camera intrinsics are invalid (fx/fy == 0). Skipping snapshot.')
            return None, None

        u_coords, v_coords = np.meshgrid(
            np.arange(img_w, dtype=np.float32),
            np.arange(img_h, dtype=np.float32)
        )
        z = depth
        x_optical = (u_coords - cx) * z / fx
        y_optical = (v_coords - cy) * z / fy

        # Depth images are projected in ROS optical-frame coordinates:
        # x right, y down, z forward. The rest of this stack expects camera_link:
        # x forward, y left, z up.
        x_link = z
        y_link = -x_optical
        z_link = -y_optical

        xyz = np.stack((x_link, y_link, z_link), axis=-1).astype(np.float32)
        valid_mask = np.isfinite(z) & (z > MIN_DEPTH) & (z < MAX_DEPTH)
        xyz[~valid_mask] = np.nan
        return xyz, valid_mask

    def callback(self, rgb_msg, depth_msg, camera_info_msg):

        if not self.detect_requested:
            return

        # 2. Reset the flag immediately so we don't process more than one frame
        self.detect_requested = False
        
        self.get_logger().info('Processing snapshot...')

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        img_h, img_w = rgb.shape[:2]

        arr, visible_mask = self.build_points_from_depth(depth_msg, camera_info_msg, rgb.shape)
        if arr is None:
            return

        arr_world = self.transform_points_to_world(
            arr.reshape(-1, 3),
            depth_msg.header.frame_id,
            depth_msg.header.stamp
        )
        if arr_world is None:
            return
        arr_world_img = arr_world.reshape((img_h, img_w, 3))
        arr_world_flat = arr_world_img.reshape(-1, 3)
        visible_mask &= np.isfinite(arr_world_img).all(axis=2)


        # Run YOLOv8
        results = self.model(rgb, verbose=False)[0]

        label_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        vis_img    = rgb.copy()

        tray_masks = []
        rack_masks = []

        if results.masks is not None:
            masks   = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()

            for mask, cls, conf in zip(masks, classes, confs):
                if conf < 0.5:
                    continue
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h))
                binary = (mask > 0.5).astype(np.uint8)
                label_mask[binary == 1] = cls + 1

                # Store masks per class
                if cls == 0:
                    tray_masks.append(binary)
                else:
                    rack_masks.append(binary)

                # Visualization
                color   = CLASS_COLORS.get(cls, (255,255,255))
                colored = np.zeros_like(vis_img)
                colored[binary == 1] = color
                vis_img = cv2.addWeighted(vis_img, 1.0, colored, 0.4, 0)

        found_rack_positions = []
        
        # ── Trays ──
        tray_count = 0
        all_found_centroids = []

        for mask_idx, mask in enumerate(tray_masks):
            object_mask = mask.astype(bool) & visible_mask
            pts_tray = arr_world_img[object_mask]

            if len(pts_tray) < MIN_CLUSTER_POINTS:
                self.get_logger().info(f'  Mask {mask_idx}: skipped — too few points')
                continue

            detections = self.get_surface_with_normal(pts_tray, f'tray_{mask_idx}')

            if not detections:
                self.get_logger().info(f'  Mask {mask_idx}: skipped — no centroid')
                continue

            for centroid, normal in detections:
                is_duplicate = any(
                    np.linalg.norm(centroid - prev_c) < 0.05
                    for prev_c in all_found_centroids
                )
                if not is_duplicate:
                    frame_id = f'tray_{tray_count}'
                    self.publish_tf_with_normal(
                        centroid, normal, frame_id,
                        self.world_frame,
                        depth_msg.header.stamp
                    )
                    all_found_centroids.append(centroid)
                    tray_count += 1

        # ── Racks ──
        rack_count = 0
        if rack_masks:
            merged_rack_mask = np.maximum.reduce(rack_masks)
            visible_label_mask = label_mask.copy()
            visible_label_mask[~visible_mask] = 0
            visible_rack_mask = merged_rack_mask.astype(bool) & visible_mask
            instances = self.get_instances(
                arr_world_flat,
                visible_label_mask,
                1,
                visible_rack_mask.astype(np.uint8)
            )

            # Merge instances that belong to the same physical rack
            # by checking if their centroids are within RACK_DEDUP_DIST in XY
            merged_instances = []
            used = [False] * len(instances)

            for i, inst_a in enumerate(instances):
                if used[i]:
                    continue
                group = list(inst_a)
                centroid_a = inst_a.mean(axis=0)
                used[i] = True

                for j, inst_b in enumerate(instances):
                    if used[j] or i == j:
                        continue
                    centroid_b = inst_b.mean(axis=0)
                    dist_xy = np.linalg.norm(centroid_a[:2] - centroid_b[:2])
                    if dist_xy < 1:  # merge instances within 1.5m XY
                        group.append(inst_b)
                        used[j] = True

                merged_instances.append(np.vstack(group) if len(group) > 1 else inst_a)

            for pts_cluster in merged_instances:

                centroid, normal = self.get_rack_opening(pts_cluster, 'rack')
                if centroid is not None:
                    if not self.is_duplicate(centroid, found_rack_positions, threshold=0.5):
                        frame = f'rack_{rack_count}'
                        self.publish_tf_with_normal(centroid, normal, frame,
                                                    self.world_frame,
                                                    depth_msg.header.stamp)
                        found_rack_positions.append(centroid)
                        rack_count += 1
                                        
        # ── Labeled point cloud ────────────────────────────────────────────────
        label_flat   = label_mask.reshape(-1)
        point_labels = label_flat

        n_tray = (point_labels == 1).sum()
        n_rack = (point_labels == 2).sum()
        self.get_logger().info(f'Points — tray: {n_tray}, rack: {n_rack} | TFs — trays: {tray_count}, racks: {rack_count}')

        labeled_cloud = self.build_labeled_cloud(depth_msg, arr_world_flat, point_labels)
        self.labeled_pub.publish(labeled_cloud)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'bgr8')
        vis_msg.header = rgb_msg.header
        self.vis_pub.publish(vis_msg)

    def build_labeled_cloud(self, original_msg, points_xyz, labels):
        header = Header()
        header.stamp    = original_msg.header.stamp
        header.frame_id = self.world_frame

        points_with_labels = []
        for pt, lbl in zip(points_xyz, labels):
            if np.isfinite(pt[0]) and np.isfinite(pt[1]) and np.isfinite(pt[2]):
                points_with_labels.append([pt[0], pt[1], pt[2], float(lbl)])

        cloud = pc2.create_cloud(
            header,
            [
                pc2.PointField(name='x',     offset=0,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y',     offset=4,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z',     offset=8,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='label', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
            ],
            points_with_labels
        )
        self.get_logger().info('Snapshot processing complete.')
        return cloud


def main():
    rclpy.init()
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
