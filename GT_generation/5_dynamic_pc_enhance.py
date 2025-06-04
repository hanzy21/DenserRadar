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

before_frames = 10
after_frames = 10

used_subname = sys.argv[1]
label_path_header = sys.argv[2]
tracking_path_header = sys.argv[3]

for root, dirs, files in os.walk(label_path_header):
    for sub_name in dirs:
        if sub_name != used_subname:
            continue
        print("sub_name",sub_name)
        label_dir = os.path.join(label_path_header, sub_name, "info_label")
        labels = sorted([file for file in os.listdir(label_dir)])
        tracking_dir = os.path.join(tracking_path_header, sub_name, "tracking")
        tracking_frames = {}

        odom_file = os.path.join(tracking_path_header, sub_name, "lidar_pose.txt")
        with open(odom_file, 'r') as file:
            odom_lines = file.readlines()
        for root, dirs, files in os.walk(tracking_dir):
            for tracking_id in dirs:
                print(tracking_id)
                tracking_frames[tracking_id] = []
                tracking_id_dir = os.path.join(tracking_dir,tracking_id)
                tracking_odom_file = os.path.join(tracking_id_dir, "tracking_odom.txt")
                with open(tracking_odom_file, 'w'):
                    pass
                enhanced_pcd_dir = os.path.join(tracking_id_dir,"enhanced")
                os.makedirs(enhanced_pcd_dir, exist_ok=True)
                tracking_time = 0
                tracking_odom = odom_lines[0]
                tracking_pcds = sorted([file for file in os.listdir(tracking_id_dir)])
                for tracking_pcd in tqdm(tracking_pcds):
                    if tracking_pcd.endswith('.pcd'):
                        tracking_time += 1
                        num = tracking_pcd[-9:-4]
                        tracking_frames[tracking_id].append(num)
                        odom = odom_lines[int(num)-1].split(",")
                        odom = [float(odo) for odo in odom]
                        time = odom[0]
                        label_path = os.path.join(label_dir, labels[int(num)-1])
                        with open(label_path, 'r') as file:
                            label_lines = file.readlines()
                            exist = False
                            for label_line in label_lines[1:]:
                                label_split = label_line.split(",")
                                if tracking_id == label_split[1]:
                                    exist = True
                                    tracking_odom = [float(label_split[i]) for i in [3,4,5,6]] 
                            assert exist
                        local_t = np.array(tracking_odom[0:3])
                        local_r = R.from_euler('z', np.deg2rad(float(tracking_odom[3]))).as_quat()
                        local_odom = list(np.hstack((time, local_t, local_r)))

                        with open(tracking_odom_file,"a") as file:
                            file.write(','.join(map(str, local_odom)))
                            file.write("\n")

                tracking_time = 0
                with open(tracking_odom_file, 'r') as file:
                    tracking_odom_lines = file.readlines()
                for tracking_pcd in tqdm(os.listdir(tracking_id_dir)):
                    if tracking_pcd.endswith('.pcd'):
                        tracking_time += 1
                        num = tracking_pcd[-9:-4]
                        if tracking_time < before_frames+1:
                            start_time = 0
                        else:
                            start_time = tracking_time - before_frames - 1

                        if tracking_time > len(tracking_frames[tracking_id]) - after_frames:
                            end_time = len(tracking_frames[tracking_id])-1
                        else:
                            end_time = tracking_time + after_frames - 1

                        start_frame = tracking_frames[tracking_id][start_time]
                        tracking_frame = tracking_frames[tracking_id][tracking_time-1]
                        end_frame = tracking_frames[tracking_id][end_time]

                        tracking_odom = tracking_odom_lines[tracking_time-1].split(',')
                        tracking_odom = [float(odo) for odo in tracking_odom]
                        tracking_t = np.array(tracking_odom[1:4])
                        tracking_r = R.from_quat(tracking_odom[4:8])
                        tracking_pcd = o3d.io.read_point_cloud(os.path.join(tracking_id_dir, tracking_frame+".pcd"))
                        tracking_points = tracking_pcd.points
                        for current_time in range(start_time, end_time+1):
                            if current_time == tracking_time-1:
                                continue
                            current_odom = tracking_odom_lines[current_time].split(',')
                            current_odom = [float(odo) for odo in current_odom]
                            current_t = np.array(current_odom[1:4])
                            current_r = R.from_quat(current_odom[4:8])

                            current_frame = tracking_frames[tracking_id][current_time]
                            current_pcd = o3d.io.read_point_cloud(os.path.join(tracking_id_dir, current_frame+".pcd"))
                            current_points = np.array(current_pcd.points)

                            current_points = current_r.inv().apply(current_points - current_t) 
                            
                            current_points = tracking_r.apply(current_points) + tracking_t 

                            tracking_points = np.vstack((tracking_points, current_points))

                        enhanced_tracking_pcd_path = os.path.join(enhanced_pcd_dir,tracking_frame+".pcd")
                        enhanced_tracking_pcd = o3d.geometry.PointCloud()
                        enhanced_tracking_pcd.points = o3d.utility.Vector3dVector(tracking_points)
                        o3d.io.write_point_cloud(enhanced_tracking_pcd_path, enhanced_tracking_pcd, write_ascii=True)
            break
