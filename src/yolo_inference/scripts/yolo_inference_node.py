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
import scipy.signal
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray

SIM = False

IMG_W = 640
IMG_H = 480

CLASS_NAMES  = {0: 'tray', 1: 'rack'}
CLASS_COLORS = {
    0: (255, 180, 100),  # tray  → orange
    1: (180, 100, 255),  # rack  → purple
}

if (not SIM):
    MODEL_PATH = '/home/nicola/ros2_ws/src/yolo_inference/weights/best.pt'
else:
    MODEL_PATH = '/home/nicola/ros2_ws/src/yolo_inference/weights/best_sim.pt'

# Filtering parameters
MIN_CLUSTER_POINTS = 100    # ignore tiny detections
MAX_DEPTH          = 3.0    # ignore points further than 3m
MIN_DEPTH          = 0.1    # ignore points closer than 0.1m
DETECTION_TTL      = 5.0    # seconds; matches marker lifetime in RViz


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
        self.obb_pub = self.create_publisher(MarkerArray, '/yolo/obb_markers', 10)

        
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.world_frame = 'world'

        self.last_tray_count = 0
        self.last_rack_count = 0
        
        self.get_logger().info('YOLO Node ready in SNAPSHOT mode. Call "ros2 service call /yolo_inference/trigger_detection std_srvs/srv/Trigger "{}"" to scan.')
    
    def clear_detection_markers(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.obb_pub.publish(marker_array)

    def make_obb_marker(self, obb_center, eigenvectors, half_extents, 
                    marker_id, frame_id, stamp, cls_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp    = stamp
        m.ns              = CLASS_NAMES.get(cls_id, 'object')
        m.id              = marker_id
        m.type            = Marker.CUBE
        m.action          = Marker.ADD
        m.lifetime        = rclpy.duration.Duration(seconds=DETECTION_TTL).to_msg()

        m.pose.position.x = float(obb_center[0])
        m.pose.position.y = float(obb_center[1])
        m.pose.position.z = float(obb_center[2])

        # Convert eigenvectors rotation matrix → quaternion
        R = eigenvectors.copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x =  0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y =  0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z =  0.25 * s

        m.pose.orientation.x = float(x)
        m.pose.orientation.y = float(y)
        m.pose.orientation.z = float(z)
        m.pose.orientation.w = float(w)

        # Full box size = 2 * half_extents
        m.scale.x = float(half_extents[0] * 2)
        m.scale.y = float(half_extents[1] * 2)
        m.scale.z = float(half_extents[2] * 2)

        # Class color with transparency
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        m.color.r = color[2] / 255.0  # CLASS_COLORS is BGR
        m.color.g = color[1] / 255.0
        m.color.b = color[0] / 255.0
        m.color.a = 0.35

        return m   

    def handle_trigger(self, request, response):
        """Service callback to enable detection for the next available frame."""
        self.clear_detection_markers()
        self.detect_requested = True
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
    
    def get_surface_with_normal(self, points, label='tray'):
        """
        Find front vertical face center and inward normal for each stacked tray layer.
        Uses a full 3D OBB (PCA on all 3 axes) per layer to extract geometry.
        Returns a list of (centroid_3d, normal_3d) tuples, one per detected tray layer.
        """
        if len(points) < 50:
            return []

        # ── Step 1: Detect tray layers by Z histogram peaks ───────────────────────
        z_vals          = points[:, 2]
        n_bins          = max(20, min(60, len(points) // 10))
        z_hist, z_edges = np.histogram(z_vals, bins=n_bins)
        z_bin_width     = z_edges[1] - z_edges[0]
        z_centers       = (z_edges[:-1] + z_edges[1:]) / 2.0

        min_sep_bins = max(1, int(0.10 / z_bin_width))

        peak_indices, _ = scipy.signal.find_peaks(
            z_hist,
            height=z_hist.max() * 0.1,
            distance=min_sep_bins,
        )

        if len(peak_indices) == 0:
            return []

        results = []

        for idx, peak_idx in enumerate(peak_indices):

            # ── Step 2: Extract layer using valley boundaries ──────────────────────
            z_lo = z_edges[0] if idx == 0 else \
                z_centers[peak_indices[idx - 1]:peak_idx + 1][
                    np.argmin(z_hist[peak_indices[idx - 1]:peak_idx + 1])
                ]
            z_hi = z_edges[-1] if idx == len(peak_indices) - 1 else \
                z_centers[peak_idx:peak_indices[idx + 1] + 1][
                    np.argmin(z_hist[peak_idx:peak_indices[idx + 1] + 1])
                ]

            layer = points[(points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)]
            if len(layer) < 20:
                continue

            # Trim top/bottom 10% in Z to remove fringe points
            z_lo_layer = np.percentile(layer[:, 2], 10)
            z_hi_layer = np.percentile(layer[:, 2], 90)
            layer = layer[(layer[:, 2] >= z_lo_layer) & (layer[:, 2] <= z_hi_layer)]
            if len(layer) < 20:
                continue

            xy_dist = np.linalg.norm(layer[:, :2], axis=1)
            dist_threshold = np.percentile(xy_dist, 98)
            layer = layer[xy_dist <= dist_threshold]

            if len(layer) < 20:
                continue

            # ── Step 3: First-pass PCA to find rough depth axis ───────────────────────
            centroid_3d = layer.mean(axis=0)
            centered    = layer - centroid_3d
            cov         = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            order        = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order]

            # Rough rack_dir to identify depth axis
            rack_dir_2d = -centroid_3d[:2] / (np.linalg.norm(centroid_3d[:2]) + 1e-6)
            dots = np.array([abs(np.dot(eigenvectors[:2, i], rack_dir_2d)) for i in range(3)])
            depth_axis_idx_rough = int(np.argmax(dots))

            rough_depth_vec = eigenvectors[:, depth_axis_idx_rough].copy()
            if np.dot(rough_depth_vec[:2], rack_dir_2d) < 0:
                rough_depth_vec = -rough_depth_vec

            # ── Step 4: Trim far 5% along rough depth axis, recompute PCA ─────────────
            depth_proj     = layer @ rough_depth_vec
            depth_threshold = np.percentile(depth_proj, 95)
            layer_trimmed  = layer[depth_proj <= depth_threshold]

            if len(layer_trimmed) < 20:
                layer_trimmed = layer  # fallback: use original

            # Second-pass PCA on trimmed layer
            centroid_3d = layer_trimmed.mean(axis=0)
            centered    = layer_trimmed - centroid_3d
            cov         = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            order        = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order]

            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 2] *= -1

            projected = centered @ eigenvectors

            # ── Step 5: Identify depth axis from refined PCA ──────────────────────────
            rack_dir_2d    = -centroid_3d[:2] / (np.linalg.norm(centroid_3d[:2]) + 1e-6)
            dots           = np.array([abs(np.dot(eigenvectors[:2, i], rack_dir_2d)) for i in range(3)])
            depth_axis_idx = int(np.argmax(dots))

            depth_vec = eigenvectors[:, depth_axis_idx].copy()
            if np.dot(depth_vec[:2], rack_dir_2d) < 0:
                depth_vec = -depth_vec

            # ── Step 5: Front-face centroid ────────────────────────────────────────
            proj_min = np.percentile(projected, 3,  axis=0)
            proj_max = np.percentile(projected, 97, axis=0)

            # Mask points inside the 90% range on ALL axes simultaneously
            inlier_mask = np.all(
                (projected >= proj_min) & (projected <= proj_max),
                axis=1
            )
            projected_trimmed = projected[inlier_mask]

            if len(projected_trimmed) < 5:
                projected_trimmed = projected  # fallback: use all points

            obb_center = centroid_3d + eigenvectors @ (
                (projected_trimmed.max(axis=0) + projected_trimmed.min(axis=0)) / 2.0
            )

            half_extents = (projected_trimmed.max(axis=0) - projected_trimmed.min(axis=0)) / 2.0

            face_centroid = obb_center + depth_vec * half_extents[depth_axis_idx]

            # ── Step 6: Normal ─────────────────────────────────────────────────────
            normal = np.array([depth_vec[0], depth_vec[1], 0.0])
            normal /= np.linalg.norm(normal) + 1e-6

            results.append((face_centroid, normal, eigenvectors, half_extents, obb_center))

            self.get_logger().debug(
                f'[{label} layer {idx}] '
                f'half_extents=({half_extents[0]:.3f}, {half_extents[1]:.3f}, {half_extents[2]:.3f})  '
                f'depth_axis={depth_axis_idx}  centroid={face_centroid.round(3)}'
            )

        return results

    def get_rack_opening(self, points, label='rack'):
        """
        Find rack front-face center and inward normal from a U-shaped rack point cloud.
        The rack opening faces the robot. Returns (centroid_3d, normal_3d) or (None, None).
        """
        if len(points) < 50:
            return None, None

        # ── Step 1: Remove bottom 20% in Z (base of U fills the arm gap) ─────────
        z_threshold = np.percentile(points[:, 2], 20)
        points = points[points[:, 2] >= z_threshold]
        if len(points) < 30:
            return None, None

        # ── Step 2: Keep only the nearest 40% in XY distance ─────────────────────
        xy_dist = np.linalg.norm(points[:, :2], axis=1)
        xy_min  = xy_dist.min()
        xy_max  = xy_dist.max()
        cluster = points[xy_dist <= xy_min + (xy_max - xy_min) * 0.4]
        if len(cluster) < 20:
            return None, None

        # ── Step 3: PCA in XY to find rack face orientation ───────────────────────
        pts_2d      = cluster[:, :2]
        centroid_2d = pts_2d.mean(axis=0)
        centered    = pts_2d - centroid_2d
        cov         = np.cov(centered.T)
        vals, vecs  = np.linalg.eigh(cov)

        depth_vec = vecs[:, np.argmin(vals)].copy()
        face_vec  = vecs[:, np.argmax(vals)].copy()

        # depth_vec must point FROM rack TOWARD robot
        rack_dir = -centroid_2d / (np.linalg.norm(centroid_2d) + 1e-6)
        if np.dot(depth_vec, rack_dir) < 0:
            depth_vec = -depth_vec

        # Detect side-view degenerate case
        if np.dot(depth_vec, rack_dir) < 0.5:
            self.get_logger().warn(
                f'PCA: depth_vec misaligned (dot={np.dot(depth_vec, rack_dir):.2f})'
                ' — swapping face/depth vectors'
            )
            depth_vec, face_vec = face_vec, depth_vec
            if np.dot(depth_vec, rack_dir) < 0:
                depth_vec = -depth_vec

        # ── Step 4: Split into left/right arms using face_vec projection ──────────
        face_proj = cluster[:, :2] @ face_vec

        hist, bin_edges = np.histogram(face_proj, bins=30)
        bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Look for the gap only in the middle 60% to avoid edge effects
        mid_mask    = (bin_centers > np.percentile(face_proj, 20)) & \
                    (bin_centers < np.percentile(face_proj, 80))
        gap_center  = bin_centers[mid_mask][np.argmin(hist[mid_mask])]

        left_cluster  = cluster[face_proj <  gap_center]
        right_cluster = cluster[face_proj >= gap_center]

        if len(left_cluster) < 10 or len(right_cluster) < 10:
            return None, None

        # ── Step 5: For each arm, find the tip (front 20% in depth) ───────────────
        def arm_tip_center(arm):
            d_proj     = arm[:, :2] @ depth_vec
            threshold  = np.percentile(d_proj, 80)
            tip_points = arm[d_proj >= threshold]
            return tip_points[:, :2].mean(axis=0) if len(tip_points) >= 2 else None

        left_tip  = arm_tip_center(left_cluster)
        right_tip = arm_tip_center(right_cluster)

        if left_tip is None or right_tip is None:
            return None, None

        # ── Step 6: Opening center = midpoint of outer edges ──────────────────────
        left_face_proj  = left_cluster[:,  :2] @ face_vec
        right_face_proj = right_cluster[:, :2] @ face_vec

        # Outer edge = points furthest from the gap (bottom 10% by face projection)
        left_outer_pts  = left_cluster[left_face_proj  <= np.percentile(left_face_proj,  10)]
        right_outer_pts = right_cluster[right_face_proj >= np.percentile(right_face_proj, 90)]

        left_outer  = left_outer_pts[:,  :2].mean(axis=0)
        right_outer = right_outer_pts[:, :2].mean(axis=0)

        center_xy = (left_outer + right_outer) / 2.0
        z_center  = cluster[:, 2].mean()
        centroid  = np.array([center_xy[0], center_xy[1], z_center])

        # ── Step 7: Normal points toward robot ────────────────────────────────────
        arm_vec = right_tip - left_tip                          # vector along rack width
        arm_vec = arm_vec / (np.linalg.norm(arm_vec) + 1e-6)   # normalize

        # Normal = 90° rotation of arm_vec in XY
        normal_candidates = np.array([ arm_vec[1], -arm_vec[0]])  # rotate +90°

        # Pick the direction pointing toward robot
        if np.dot(normal_candidates, rack_dir) < 0:
            normal_candidates = -normal_candidates

        normal = np.array([normal_candidates[0], normal_candidates[1], 0.0])

        return centroid, normal


    def is_duplicate(self, new_pos, existing_positions, threshold=0.25):
        """Check if the new position is within 'threshold' meters of any existing detection."""
        for pos in existing_positions:
            dist = np.linalg.norm(new_pos - pos)
            if dist < threshold:
                return True
        return False

    def publish_tf_with_normal(self, position, normal, frame_name, parent_frame, stamp, static=False):
        """Publish TF with Z axis aligned to surface normal."""
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

        # D435 640x480 — computed from actual FOV 69.4° x 42.5°
        if (not SIM):
            fx = 605 #615
            fy = 605 #615

            cx = 309  #331 image upside down
            cy = 231  #249
        else:
            fx = 462.1  #sim values
            fy = 455

            cx = 320.0
            cy = 240.0


        u_coords, v_coords = np.meshgrid(
            np.arange(img_w, dtype=np.float32),
            np.arange(img_h, dtype=np.float32)
        )
        z = depth
        x_optical = (u_coords - cx) * z / fx
        y_optical = (v_coords - cy) * z / fy


        xyz = np.stack((x_optical, y_optical, z), axis=-1).astype(np.float32)
        valid_mask = np.isfinite(z) & (z > MIN_DEPTH) & (z < MAX_DEPTH)
        xyz[~valid_mask] = np.nan
        return xyz, valid_mask

    def callback(self, rgb_msg, depth_msg, camera_info_msg):

        if not self.detect_requested:
            return

        # 2. Reset the flag immediately so we don't process more than one frame
        self.detect_requested = False
        
        self.get_logger().info('Processing snapshot...')
        detection_stamp = self.get_clock().now().to_msg()

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        img_h, img_w = rgb.shape[:2]

        arr, visible_mask = self.build_points_from_depth(depth_msg, camera_info_msg, rgb.shape)
        if arr is None:
            return
        # #debug
        # arr_flat_debug = arr.reshape(-1, 3)
        # valid_pts = arr_flat_debug[np.isfinite(arr_flat_debug).all(axis=1)]
        # self.get_logger().info(
        #     f'RAW optical points\n'
        #     f'  x (right):   min={valid_pts[:,0].min():.3f}  max={valid_pts[:,0].max():.3f}  mean={valid_pts[:,0].mean():.3f}\n'
        #     f'  y (down):    min={valid_pts[:,1].min():.3f}  max={valid_pts[:,1].max():.3f}  mean={valid_pts[:,1].mean():.3f}\n'
        #     f'  z (forward): min={valid_pts[:,2].min():.3f}  max={valid_pts[:,2].max():.3f}  mean={valid_pts[:,2].mean():.3f}'
        # )
        # self.get_logger().info(
        #     f'Intrinsics — fx={camera_info_msg.k[0]:.2f}  fy={camera_info_msg.k[4]:.2f}  '
        #     f'cx={camera_info_msg.k[2]:.2f}  cy={camera_info_msg.k[5]:.2f}  '
        #     f'encoding={depth_msg.encoding}'
        # )
        # #

        arr_world = self.transform_points_to_world(
            arr.reshape(-1, 3),
            'camera_optical_frame',
            depth_msg.header.stamp
        )
        if arr_world is None:
            return
        arr_world_img = arr_world.reshape((img_h, img_w, 3))
        arr_world_flat = arr_world_img.reshape(-1, 3)
        visible_mask &= np.isfinite(arr_world_img).all(axis=2)
        # # debug
        # valid_w = arr_world_flat[np.isfinite(arr_world_flat).all(axis=1)]
        # self.get_logger().info(
        #     f'WORLD points\n'
        #     f'  x: min={valid_w[:,0].min():.3f}  max={valid_w[:,0].max():.3f}  mean={valid_w[:,0].mean():.3f}\n'
        #     f'  y: min={valid_w[:,1].min():.3f}  max={valid_w[:,1].max():.3f}  mean={valid_w[:,1].mean():.3f}\n'
        #     f'  z: min={valid_w[:,2].min():.3f}  max={valid_w[:,2].max():.3f}  mean={valid_w[:,2].mean():.3f}'
        # )
        # #

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
        marker_array = MarkerArray()
        marker_id    = 0

        if tray_masks:
            # Merge all tray masks and collect all points at once
            merged_tray_mask = np.maximum.reduce(tray_masks).astype(bool) & visible_mask
            pts_all_trays = arr_world_img[merged_tray_mask]

            if len(pts_all_trays) >= MIN_CLUSTER_POINTS:
                detections = self.get_surface_with_normal(pts_all_trays, 'tray')

                for centroid, normal, eigenvectors, half_extents, obb_center in detections:
                    frame_id = f'tray_{tray_count}'
                    self.publish_tf_with_normal(
                        centroid, normal, frame_id,
                        self.world_frame, detection_stamp
                    )
                    all_found_centroids.append(centroid)
                    

                    marker_array.markers.append(
                        self.make_obb_marker(
                            obb_center,
                            eigenvectors, half_extents,
                            marker_id, self.world_frame,
                            detection_stamp, cls_id=0
                        )
                    )
                    marker_id  += 1
                    tray_count += 1

        self.obb_pub.publish(marker_array)

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
                                                    detection_stamp)
                        found_rack_positions.append(centroid)
                        rack_count += 1
                                        
        # ── Labeled point cloud ────────────────────────────────────────────────
        label_flat   = label_mask.reshape(-1)
        point_labels = label_flat

        n_tray = (point_labels == 1).sum()
        n_rack = (point_labels == 2).sum()
        self.get_logger().info(f'Points — tray: {n_tray}, rack: {n_rack} | TFs — trays: {tray_count}, racks: {rack_count}')

        arr_flat = arr.reshape(-1, 3)
        labeled_cloud = self.build_labeled_cloud(depth_msg, arr_flat, point_labels, 'camera_optical_frame')
        self.labeled_pub.publish(labeled_cloud)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, 'bgr8')
        vis_msg.header = rgb_msg.header
        self.vis_pub.publish(vis_msg)
        self.last_tray_count = tray_count
        self.last_rack_count = rack_count

    def build_labeled_cloud(self, original_msg, points_xyz, labels, frame_id=None):
        header = Header()
        header.stamp    = original_msg.header.stamp
        header.frame_id = frame_id if frame_id else self.world_frame

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
        #self.get_logger().info(f'Publishing cloud with {len(points_with_labels)} points, frame: {header.frame_id}')
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
