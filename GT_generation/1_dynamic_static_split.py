#!/usr/bin/env python

import os
import sys

import open3d as o3d
import numpy as np
import pickle
from tqdm import tqdm

used_subname = sys.argv[1]
lidar_path_header = sys.argv[2]
label_path_header = sys.argv[2]
split_lidar_path_header = sys.argv[3]

device = o3d.core.Device("CPU:0")
dtype_f32 = o3d.core.float32
dtype_u8 = o3d.core.uint8
dtype_u16 = o3d.core.uint16
dtype_u32 = o3d.core.uint32

for root, dirs, files in os.walk(lidar_path_header):
    for sub_name in dirs:
        if sub_name != used_subname:
            continue
        print("sub_name",sub_name)
        lidar_dir = os.path.join(lidar_path_header, sub_name, "os1-128")
        low_lidar_dir = os.path.join(lidar_path_header, sub_name, "os2-64")
        label_dir = os.path.join(label_path_header, sub_name, "info_label")
        dyn_pcd_dir = os.path.join(split_lidar_path_header, sub_name, "dynamic")
        sta_pcd_dir = os.path.join(split_lidar_path_header, sub_name, "static")
        tracking_dir = os.path.join(split_lidar_path_header, sub_name, "tracking")
        os.makedirs(dyn_pcd_dir, exist_ok=True)
        os.makedirs(sta_pcd_dir, exist_ok=True)
        os.makedirs(tracking_dir, exist_ok=True)
        labels = sorted([file for file in os.listdir(label_dir)])
        for label in tqdm(labels):
            label_path = os.path.join(label_dir, label)
            with open(label_path, 'r') as file:
                label_lines = file.readlines()
            num = label_lines[0][56:61]
            low_lidar_name = "os2-64_" + num + ".pcd"
            low_lidar_path = os.path.join(low_lidar_dir, low_lidar_name)
            dyn_pcd_path = os.path.join(dyn_pcd_dir, num+".pcd")
            sta_pcd_path = os.path.join(sta_pcd_dir, num+".pcd")
            low_lidar = o3d.t.io.read_point_cloud(low_lidar_path)

            sta_pcd = low_lidar
            dyn_points_mask = np.zeros_like((low_lidar.point["positions"].numpy()[:,0]))

            points = low_lidar.point["positions"].numpy()
            for label_line in label_lines[1:]:
                label_split = label_line.split(",")

                tracking_id = label_split[1]
                tracking_id_dir = os.path.join(tracking_dir,tracking_id)
                os.makedirs(tracking_id_dir, exist_ok=True)
                tracking_pcd_path = os.path.join(tracking_id_dir, num+".pcd")
                    
                center_x = float(label_split[3])
                center_y = float(label_split[4])
                center_z = float(label_split[5])
                position = np.array([center_x, center_y, center_z])
                heading = np.deg2rad(float(label_split[6]))
                c, s = np.cos(-heading), np.sin(-heading)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                local_points = R.dot((points - position).T).T

                l_2 = float(label_split[7])
                w_2 = float(label_split[8])
                h_2 = float(label_split[9])
                half_size = np.array([[l_2, w_2, h_2]])

                mask = np.all((-half_size <= local_points) & (local_points <= half_size), axis=-1)
                dyn_points_mask += mask

                tracking_points_mask = mask.astype(int)
                tracking_pcd = o3d.t.geometry.PointCloud(device)
                if tracking_points_mask.sum() != 0:
                    tracking_points = low_lidar.point["positions"][tracking_points_mask == 1].numpy()
                    tracking_points[np.abs(tracking_points) < 1e-10] = 0.0
                    tracking_pcd.point["positions"] = o3d.core.Tensor(tracking_points, dtype_f32, device)
                    tracking_pcd.point["intensity"] = o3d.core.Tensor(low_lidar.point["intensity"][tracking_points_mask == 1].numpy(), dtype_f32, device)
                    tracking_pcd.point["t"] = o3d.core.Tensor(low_lidar.point["t"][tracking_points_mask == 1].numpy(), dtype_u32, device)
                    tracking_pcd.point["reflectivity"] = o3d.core.Tensor(low_lidar.point["reflectivity"][tracking_points_mask == 1].numpy(), dtype_u16, device)
                    tracking_pcd.point["ring"] = o3d.core.Tensor(low_lidar.point["ring"][tracking_points_mask == 1].numpy(), dtype_u8, device)
                    tracking_pcd.point["ambient"] = o3d.core.Tensor(low_lidar.point["ambient"][tracking_points_mask == 1].numpy(), dtype_u16, device)
                    tracking_pcd.point["range"] = o3d.core.Tensor(low_lidar.point["range"][tracking_points_mask == 1].numpy(), dtype_u32, device)
                    o3d.t.io.write_point_cloud(tracking_pcd_path, tracking_pcd, write_ascii=True)

                elif tracking_points_mask.sum() == 0:
                    tracking_pcd.point["positions"] = o3d.core.Tensor(position.reshape(1, 3), dtype_f32, device)
                    tracking_pcd.point["intensity"] = o3d.core.Tensor(np.zeros((1,1)), dtype_f32, device)
                    tracking_pcd.point["t"] = o3d.core.Tensor(np.zeros((1,1)), dtype_u32, device)
                    tracking_pcd.point["reflectivity"] = o3d.core.Tensor(np.zeros((1,1)), dtype_u16, device)
                    tracking_pcd.point["ring"] = o3d.core.Tensor(np.zeros((1,1)), dtype_u8, device)
                    tracking_pcd.point["ambient"] = o3d.core.Tensor(np.zeros((1,1)), dtype_u16, device)
                    tracking_pcd.point["range"] = o3d.core.Tensor(np.zeros((1,1)), dtype_u32, device)
                    o3d.t.io.write_point_cloud(tracking_pcd_path, tracking_pcd, write_ascii=True)

            sta_points_mask = 1-dyn_points_mask
            sta_points_mask = sta_points_mask.astype(int)
            sta_pcd = o3d.t.geometry.PointCloud(device)
            sta_points = low_lidar.point["positions"][sta_points_mask == 1].numpy()
            sta_points[np.abs(sta_points) < 1e-10] = 0.0
            sta_pcd.point["positions"] = o3d.core.Tensor(sta_points, dtype_f32, device)
            sta_pcd.point["intensity"] = o3d.core.Tensor(low_lidar.point["intensity"][sta_points_mask == 1].numpy(), dtype_f32, device)
            sta_pcd.point["t"] = o3d.core.Tensor(low_lidar.point["t"][sta_points_mask == 1].numpy(), dtype_u32, device)               # sta_pcd.points = o3d.utility.Vector3dVector(sta_points)
            sta_pcd.point["reflectivity"] = o3d.core.Tensor(low_lidar.point["reflectivity"][sta_points_mask == 1].numpy(), dtype_u16, device)
            sta_pcd.point["ring"] = o3d.core.Tensor(low_lidar.point["ring"][sta_points_mask == 1].numpy(), dtype_u8, device)
            sta_pcd.point["ambient"] = o3d.core.Tensor(low_lidar.point["ambient"][sta_points_mask == 1].numpy(), dtype_u16, device)
            sta_pcd.point["range"] = o3d.core.Tensor(low_lidar.point["range"][sta_points_mask == 1].numpy(), dtype_u32, device)
            o3d.t.io.write_point_cloud(sta_pcd_path, sta_pcd, write_ascii=True)
    break