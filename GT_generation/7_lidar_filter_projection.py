#!/usr/bin/env python


# Library
import sys
import os
import numpy as np
import torch
import open3d as o3d
from scipy.io import loadmat
import torch.nn.functional as F
from tqdm import tqdm


from projection_utils import *

used_subname = sys.argv[1]
raw_path_header = sys.argv[2]
save_path_header = sys.argv[3]

class LidarReg():
    def __init__(self):
        super().__init__()
        self.initGlobalVariables()

    def initGlobalVariables(self):
        pass

    def getCalibration(self):
        self.calib_lr = np.array([-2.54, 0.3, 0.7])
            
    def loadLidar(self):
        pcd = o3d.io.read_point_cloud(self.lidar_path)
        points = np.asarray(pcd.points)
        points = points[(points[:, 0] != 0) & (points[:, 1] != 0) & (points[:, 2] != 0)]
        self.lidar_points = points + self.calib_lr
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.lidar_points)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
        
    def loadRawData(self):
        arrDREA = loadmat(self.raw_path)
        arrDREA = arrDREA["arrDREA"].astype(np.float32)
        arrDREA = np.log(arrDREA)
        arrDREA[0,:,:,:] = 0
        arrDREA[:,:,:ground_elevation_index, :] = 0
        max_value = np.max(arrDREA, axis=0)
        max_indices = np.argmax(arrDREA, axis=0)
        max_doppler = max_indices*0.06-1.932591218305504
        max_value = torch.tensor(max_value)
        result = np.stack((max_value, max_doppler))
        torch.save(result[:,:,ground_elevation_index:, :].astype(np.float32), "{}/tensor_{}.pt".format(self.save_tensor_path, self.rawname))
        k = int(0.3 * max_value.numel())
        top_values, _ = torch.topk(max_value.reshape(-1), k, sorted=True)
        threshold = top_values[-1]
        self.rawdata = torch.where(max_value > threshold, torch.tensor(1.0), torch.tensor(0.0))

    def cubeGeneration(self):
        self.label = np.zeros([len_range, len_elevation, len_azimuth])
        rea_points = xyz2rea(self.lidar_points)
        rea_points_roi = rea_points[
        (rea_points[:, 0] >= min_range)     & (rea_points[:, 0] < max_range) &
        (rea_points[:, 1] >= min_elevation) & (rea_points[:, 1] < max_elevation) &
        (rea_points[:, 2] >= min_azimuth)   & (rea_points[:, 2] < max_azimuth)
        ]
        self.xyz_points_roi = rea2xyz(rea_points_roi)
        
        indexs_roi = rea2tensorInd(rea_points_roi)
        filtered_indexs_roi = indexs_roi[
        (indexs_roi[:, 0] >= 0) & (indexs_roi[:, 0] < len_range) &
        (indexs_roi[:, 1] >= 0) & (indexs_roi[:, 1] < len_elevation) &
        (indexs_roi[:, 2] >= 0) & (indexs_roi[:, 2] < len_azimuth)
        ]
        self.label[filtered_indexs_roi[:, 0], filtered_indexs_roi[:, 1], filtered_indexs_roi[:, 2]] = 1
        
    def labelFilter(self):
        R, E, A = self.rawdata.size()
        self.rawdata_upsampled = F.interpolate(self.rawdata.unsqueeze(0).unsqueeze(0), 
                                        size=(R*2, E*2, A*2), 
                                        mode='nearest').squeeze(0).squeeze(0)
        self.filtered_cube = self.rawdata_upsampled*self.label
        self.filtered_cube = self.filtered_cube[:,ground_elevation_index*2:, :]
        
        self.filtered_cube = torch.nonzero(self.filtered_cube)
        self.filtered_points = tensorInd2rea(self.filtered_cube)
        self.filtered_points = rea2xyz(self.filtered_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.filtered_points)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

        torch.save(self.filtered_cube, "{}/label_{}.pt".format(self.save_label_path, self.rawname))
        
    def reverse(self):
        self.label = torch.tensor(self.label)
        self.lidar_points = torch.nonzero(self.label)
        self.lidar_points = tensorInd2rea(self.lidar_points)
        self.lidar_points = rea2xyz(self.lidar_points)
    
            
    def run(self):

        self.getCalibration()
        for root, dirs, files in os.walk(save_path_header):
            for sub_name in dirs:
                if sub_name != used_subname:
                    continue
                print("sub_name",sub_name)
                self.lid_dirs = os.path.join(save_path_header, sub_name, "stitched")
                self.raw_dirs = os.path.join(raw_path_header, sub_name, "radar_tesseract")
                self.label_dirs = os.path.join(raw_path_header, sub_name, "info_label")
                self.save_label_path = os.path.join(save_path_header, sub_name, sub_name, "labels")
                self.save_tensor_path = os.path.join(save_path_header, sub_name, sub_name, "tensors")
                os.makedirs(self.save_label_path, exist_ok=True)
                os.makedirs(self.save_tensor_path, exist_ok=True)
                print(os.path.join(root,sub_name))
                labels = sorted([file for file in os.listdir(self.label_dirs)])
                num = 0
                for label in tqdm(labels):
                    label_file = os.path.join(self.label_dirs, label)
                    with open(label_file, "r") as file:
                        info = file.readline()
                        self.rawname = info[50:55]
                        self.lidname = info[56:61]
                    self.lidar_path = os.path.join(self.lid_dirs, self.lidname+".pcd")
                    self.raw_path = os.path.join(self.raw_dirs, "tesseract_"+self.rawname+".mat")
                    self.loadLidar()
                    self.loadRawData()
                    self.cubeGeneration()
                    self.labelFilter()
            break


if __name__ == '__main__':
    depth_map = LidarReg()
    depth_map.run()