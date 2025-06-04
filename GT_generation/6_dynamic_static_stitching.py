#!/usr/bin/env python

import os
import sys

import open3d as o3d
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


device = o3d.core.Device("CPU:0")
dtype_f32 = o3d.core.float32
dtype_u8 = o3d.core.uint8
dtype_u16 = o3d.core.uint16
dtype_u32 = o3d.core.uint32

used_subname = sys.argv[1]
label_path_header = sys.argv[2]
path_header = sys.argv[3]

voxel_size = 0.01

# @profile
def main():
    for root, dirs, files in os.walk(path_header):
        for sub_name in dirs:
            if sub_name != used_subname:
                continue
            print("sub_name",sub_name)
            static_lidar_dir = os.path.join(path_header, sub_name, "static_map")
            dynamic_lidar_dir = os.path.join(path_header, sub_name, "tracking")
            stitched_lidar_dir = os.path.join(path_header, sub_name, "stitched")
            label_dir = os.path.join(label_path_header, sub_name, "info_label")
            labels = sorted([file for file in os.listdir(label_dir)])
            os.makedirs(stitched_lidar_dir, exist_ok=True)
            time = 0
            static_pcd_files = sorted([file for file in os.listdir(static_lidar_dir)])
            for static_pcd_file in tqdm(static_pcd_files):
                time += 1
                if time > 100000:
                    break
                if not static_pcd_file.endswith('.pcd'):
                    continue
                static_pcd = o3d.io.read_point_cloud(os.path.join(static_lidar_dir, static_pcd_file))
                static_points = static_pcd.points

                frame = static_pcd_file[-9:-4]
                label_path = os.path.join(label_dir, labels[int(frame)-1])
                with open(label_path, 'r') as file:
                    label_lines = file.readlines()

                for label_line in label_lines[1:]:
                    label_split = label_line.split(",")
                    tracking_id = label_split[1]
                    tracking_id_dir = os.path.join(dynamic_lidar_dir, str(tracking_id), "enhanced")
                    tracking_pcd = o3d.io.read_point_cloud(os.path.join(tracking_id_dir, str(frame+".pcd")))
                    tracking_points = tracking_pcd.points
                    static_points = np.vstack((static_points, tracking_points))
                    
                stitched_pcd_path = os.path.join(stitched_lidar_dir, static_pcd_file)
                stitched_pcd = o3d.geometry.PointCloud()
                stitched_pcd.points = o3d.utility.Vector3dVector(static_points)
                stitched_pcd = stitched_pcd.voxel_down_sample(voxel_size)
                o3d.io.write_point_cloud(stitched_pcd_path, stitched_pcd, write_ascii=True)
            break
        break

main()