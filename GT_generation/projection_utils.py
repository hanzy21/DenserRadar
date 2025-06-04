import numpy as np
import math
from scipy.spatial.transform import Rotation as R

up_rate = 2

init_len_range = 256
init_len_elevation = 37
init_len_azimuth = 107
init_len_doppler = 64

ground_elevation_index = 17

len_range = init_len_range*up_rate
len_elevation = init_len_elevation*up_rate
len_azimuth = init_len_azimuth*up_rate

init_res_range = 0.4629
init_res_elevation = 1
init_res_azimuth = 1

res_range = init_res_range/up_rate
res_elevation = init_res_elevation/up_rate
res_azimuth = init_res_azimuth/up_rate
points_num_fps = 100000

max_range = len_range*res_range
min_range = 0
max_elevation = int(len_elevation*res_elevation/2)
min_elevation = -max_elevation
max_azimuth = int(len_azimuth*res_azimuth/2)
min_azimuth = -max_azimuth


def xyz2rea(points):
    if points.ndim == 1:
        points = points.reshape(1,-1)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    r = np.sqrt(np.sum(points**2, axis=1))#math.sqrt(x**2 + y**2 + z**2)  # Radius
    # print("z/np.sqrt(x**2+y**2)",z/np.sqrt(x**2+y**2))
    e = np.degrees(np.arctan2(z, np.sqrt(x**2+y**2)))  # Elevation angle 
    a = np.degrees(np.arctan2(-y, x))  # Azimuth angle   #角度的计算抄自kradar官方m文件
    return np.vstack([r, e, a]).T

def rea2xyz(points):
    if points.ndim == 1:
        points = points.reshape(1,-1)
    r = points[:,0]
    e = np.radians(points[:,1])
    a = np.radians(points[:,2])
    x = r*np.cos(e)*np.cos(a)
    y = -r*np.cos(e)*np.sin(a)
    z = r*np.sin(e)
    return np.vstack([x, y, z]).T

def rea2tensorInd(points):
    r = points[:,0]
    e = points[:,1]
    a = points[:,2]
    index_r = (r/res_range).astype(np.int64)
    index_e = (e/res_elevation+(len_elevation-1)/2).astype(np.int64)
    index_a = (a/res_azimuth+(len_azimuth-1)/2).astype(np.int64)
    return np.vstack([index_r, index_e, index_a]).T

def tensorInd2rea(indexs):
    index_r = indexs[:,0]
    index_e = indexs[:,1]
    index_a = indexs[:,2]
    r = index_r*res_range
    e = (index_e-(len_elevation-1)/2)*res_elevation
    a = (index_a-(len_azimuth-1)/2)*res_azimuth
    return np.vstack([r, e, a]).T



def index2coord_trans(non_zero_indices):
    index_r = non_zero_indices[:,0]
    index_e = non_zero_indices[:,1]
    index_a = non_zero_indices[:,2]
    range_start = 0
    range_step  = res_range
    elevation_start = -int(20-37/2)
    elevation_step  = res_elevation
    azimuth_start = -int(107/res_azimuth/2)
    azimuth_step  = res_azimuth
    start = np.array([range_start, elevation_start, azimuth_start])
    step  = np.array([range_step,  elevation_step,  azimuth_step])
    # transformed_indices = start + non_zero_indices*step
    
    r = start[0] + index_r*step[0]
    e = start[1] + index_e*step[1]
    a = start[2] + index_a*step[2]
    return np.vstack([r, e, a]).T