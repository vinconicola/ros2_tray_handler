#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d
import ros2_numpy
import sensor_msgs_py.point_cloud2 as pc2
from numpy.lib import recfunctions
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
                [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]


def to_msg(open3d_cloud, frame_id=None, stamp=None):
    header = Header()
    if stamp is not None:
        header.stamp = stamp
    if frame_id is not None:
        header.frame_id = frame_id

    o3d_asarray = np.asarray(open3d_cloud.points)

    o3d_x = o3d_asarray[:, 0]
    o3d_y = o3d_asarray[:, 1]
    o3d_z = o3d_asarray[:, 2]

    cloud_data = np.core.records.fromarrays([o3d_x, o3d_y, o3d_z], names='x,y,z')

    if not open3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        color_array = np.array(np.floor(np.asarray(open3d_cloud.colors) * 255), dtype=np.uint8)

        o3d_r = color_array[:, 0]
        o3d_g = color_array[:, 1]
        o3d_b = color_array[:, 2]

        cloud_data = np.lib.recfunctions.append_fields(cloud_data, ['r', 'g', 'b'], [o3d_r, o3d_g, o3d_b])

        cloud_data = ros2_numpy.point_cloud2.merge_rgb_fields(cloud_data)

    return pc2.create_cloud(header, fields, cloud_data)


def from_msg(ros_cloud):
    field_names = [f.name for f in ros_cloud.fields]
    has_rgb = 'rgb' in field_names or 'rgba' in field_names

    points_list = list(pc2.read_points(ros_cloud,
                                        field_names=("x", "y", "z"),
                                        skip_nans=True))

    open3d_cloud = open3d.geometry.PointCloud()
    if len(points_list) == 0:
        return open3d_cloud

    points = np.column_stack([
        np.array([p[0] for p in points_list], dtype=np.float64),
        np.array([p[1] for p in points_list], dtype=np.float64),
        np.array([p[2] for p in points_list], dtype=np.float64),
    ])
    
    # Filter out inf values
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    
    if len(points) == 0:
        return open3d_cloud

    open3d_cloud.points = open3d.utility.Vector3dVector(points)

    if has_rgb:
        rgb_list = list(pc2.read_points(ros_cloud,
                                         field_names=("rgb",),
                                         skip_nans=True))
        rgb_floats = np.array([r[0] for r in rgb_list], dtype=np.float32)
        # Apply same mask
        rgb_floats = rgb_floats[mask]
        rgb_int = rgb_floats.view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.float64) / 255.0
        g = ((rgb_int >> 8)  & 0xFF).astype(np.float64) / 255.0
        b = ( rgb_int        & 0xFF).astype(np.float64) / 255.0
        colors = np.stack([r, g, b], axis=1)
        open3d_cloud.colors = open3d.utility.Vector3dVector(colors)

    return open3d_cloud
