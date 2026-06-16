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
from sklearn.cluster import KMeans
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
    MODEL_PATH = 'src/yolo_inference/weights/best.pt'
else:
    MODEL_PATH = 'src/yolo_inference/weights/best_sim.pt'

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
        Approach: split into Z-layers, project each layer to XY, fit a minimum-area
        rotated rectangle (cv2.minAreaRect) to the XY footprint, then lift back to 3D
        using the layer's Z extent for height.
        Returns a list of (face_centroid, normal, eigenvectors, half_extents, obb_center) tuples.
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

            # Trim top/bottom 10% in Z to remove fringe points (floor/ceiling of layer)
            z_lo_layer = np.percentile(layer[:, 2], 10)
            z_hi_layer = np.percentile(layer[:, 2], 90)
            layer = layer[(layer[:, 2] >= z_lo_layer) & (layer[:, 2] <= z_hi_layer)]
            if len(layer) < 20:
                continue

            # ── Step 3: Fit a 2D rotated rectangle to the XY footprint ─────────────
            xy = layer[:, :2].astype(np.float32)

            # Remove gross outliers before computing the hull, so a few stray
            # points don't blow up the rectangle. Use median + percentile radius.
            center_xy = np.median(xy, axis=0)
            radii = np.linalg.norm(xy - center_xy, axis=1)
            radius_thresh = np.percentile(radii, 97)
            xy_clean = xy[radii <= radius_thresh]

            if len(xy_clean) < 10:
                xy_clean = xy

            rect = cv2.minAreaRect(xy_clean)  # ((cx, cy), (w, h), angle_deg)
            (rect_cx, rect_cy), (rect_w, rect_h), angle_deg = rect

            if rect_w < 1e-6 or rect_h < 1e-6:
                continue

            angle_rad = np.deg2rad(angle_deg)

            # cv2.minAreaRect angle convention: rotation of the "width" axis from +x
            axis_w = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            axis_h = np.array([-np.sin(angle_rad), np.cos(angle_rad), 0.0])

            # ── Step 4: Decide which rectangle axis is "depth" (toward camera) ─────
            rect_center_2d = np.array([rect_cx, rect_cy])
            toward_camera_2d = -rect_center_2d / (np.linalg.norm(rect_center_2d) + 1e-6)

            align_w = abs(np.dot(axis_w[:2], toward_camera_2d))
            align_h = abs(np.dot(axis_h[:2], toward_camera_2d))

            if align_w >= align_h:
                depth_vec_2d, depth_extent = axis_w, rect_w
                width_vec_2d, width_extent = axis_h, rect_h
            else:
                depth_vec_2d, depth_extent = axis_h, rect_h
                width_vec_2d, width_extent = axis_w, rect_w

            # Orient depth axis to point toward the camera (front face faces sensor)
            if np.dot(depth_vec_2d[:2], toward_camera_2d) < 0:
                depth_vec_2d = -depth_vec_2d

            depth_vec = np.array([depth_vec_2d[0], depth_vec_2d[1], 0.0])
            depth_vec /= np.linalg.norm(depth_vec) + 1e-6

            width_vec = np.array([width_vec_2d[0], width_vec_2d[1], 0.0])
            width_vec /= np.linalg.norm(width_vec) + 1e-6

            height_vec = np.array([0.0, 0.0, 1.0])

            eigenvectors = np.column_stack([depth_vec, width_vec, height_vec])

            # ── Step 5: Lift back to 3D ─────────────────────────────────────────────
            z_center = (layer[:, 2].min() + layer[:, 2].max()) / 2.0
            z_half   = (layer[:, 2].max() - layer[:, 2].min()) / 2.0

            obb_center = np.array([rect_cx, rect_cy, z_center])
            half_extents = np.array([depth_extent / 2.0, width_extent / 2.0, z_half])

            # Front face centroid: OBB center pushed forward along depth axis
            face_centroid = obb_center + depth_vec * half_extents[0]

            # ── Step 6: Normal ───────────────────────────────────────────────────────
            normal = depth_vec.copy()  # already XY-only, normalized, z=0

            results.append((face_centroid, normal, eigenvectors, half_extents, obb_center))

            self.get_logger().debug(
                f'[{label} layer {idx}] '
                f'half_extents=({half_extents[0]:.3f}, {half_extents[1]:.3f}, {half_extents[2]:.3f})  '
                f'centroid={face_centroid.round(3)}  '
                f'rect=({rect_w:.3f}x{rect_h:.3f} @ {angle_deg:.1f}°)'
            )

        return results

    def get_rack_opening(self, points, label='rack'):
        """
        Simple, Data-Driven Pose Estimation for a front-view U-rack with NO back wall.
        Splits points into Left/Right arms using 2D K-Means spatial clustering, 
        then applies independent local statistical filters.
        """
        if len(points) < MIN_CLUSTER_POINTS:
            return None, None

        # 1. Project points onto the 2D horizontal plane (Discard Z)
        points_2d = points[:, :2].astype(np.float64)

        # 2. Split the raw points into two clusters based on BOTH X and Y using K-Means
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(points_2d)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Identify which cluster is Left and which is Right based on their X-centers
        if cluster_centers[0, 0] < cluster_centers[1, 0]:
            left_mask = (labels == 0)
            right_mask = (labels == 1)
        else:
            left_mask = (labels == 1)
            right_mask = (labels == 0)

        left_arm_raw = points_2d[left_mask]
        right_arm_raw = points_2d[right_mask]

        # 3. STRICT CHECK: Ensure both sides have enough data before filtering
        if len(left_arm_raw) < (MIN_CLUSTER_POINTS // 3) or len(right_arm_raw) < (MIN_CLUSTER_POINTS // 3):
            self.get_logger().debug("Rack tracking skipped: Points are heavily concentrated on only one arm.")
            return None, None

        # =========================================================================
        # LOCAL STATISTICAL FILTERING
        # =========================================================================
        # Clean Left Arm: Filter points based on distance to the left arm's own center
        left_median = np.median(left_arm_raw, axis=0)
        left_distances = np.linalg.norm(left_arm_raw - left_median, axis=1)
        left_cleaned = left_arm_raw[left_distances < np.percentile(left_distances, 96)]

        # Clean Right Arm: Filter points based on distance to the right arm's own center
        right_median = np.median(right_arm_raw, axis=0)
        right_distances = np.linalg.norm(right_arm_raw - right_median, axis=1)
        right_cleaned = right_arm_raw[right_distances < np.percentile(right_distances, 96)]
        # =========================================================================

        # Ensure we still have enough valid data after the local filtering
        if len(left_cleaned) < 15 or len(right_cleaned) < 15:
            return None, None

        # 4. Extract the exact 'Front Tips' facing the camera
        left_tip_y = np.percentile(left_cleaned[:, 1], 2)
        left_tip_x = np.median(left_cleaned[left_cleaned[:, 1] < np.percentile(left_cleaned[:, 1], 10), 0])
        left_tip = np.array([left_tip_x, left_tip_y])

        right_tip_y = np.percentile(right_cleaned[:, 1], 2)
        right_tip_x = np.median(right_cleaned[right_cleaned[:, 1] < np.percentile(right_cleaned[:, 1], 10), 0])
        right_tip = np.array([right_tip_x, right_tip_y])

        # 5. Centroid is the clean mathematical midpoint of the opening gap
        center_2d = (left_tip + right_tip) / 2.0

        # 6. Compute Yaw Orientation from the line connecting the two tips
        width_vector = right_tip - left_tip
        width_vector /= np.linalg.norm(width_vector) + 1e-6

        # The normal vector is strictly perpendicular to the connecting line of the tips
        normal_2d = np.array([-width_vector[1], width_vector[0]])
        
        # Ensure normal vector points back toward the camera frame (negative Y direction)
        if normal_2d[1] > 0:
            normal_2d = -normal_2d
        normal_2d /= np.linalg.norm(normal_2d) + 1e-6

        # 7. Package into standard 3D outputs expected by your node's publishers
        centroid = np.array([center_2d[0], center_2d[1], np.mean(points[:, 2])])
        normal = np.array([normal_2d[0], normal_2d[1], 0.0])

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
            instance_points = self.filter_cluster(instance_points, std_threshold=1)
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
