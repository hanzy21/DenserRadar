
from easydict import EasyDict as edict
import torch
import yaml
import numpy as np
from scipy.spatial import cKDTree


def get_config():
    with open("./config/base.yaml","r") as file:
        return edict(yaml.safe_load(file))


def collate_fn(batch):
    cfgs = get_config()
    min_point = cfgs.max_labelpoints
    for x in batch:
        min_point = min(min_point, x[0].shape[0])
    label = torch.zeros([len(batch), min_point, 3])
    rawdata = batch[0][1].unsqueeze(0).repeat(len(batch), 1, 1, 1, 1)
    for i in range(len(batch)):
        if cfgs.fps:
            label[i,:,:] = farthest_point_sample(batch[i][0], min_point)
        else:
            label[i,:,:] = batch[i][0][:min_point,:]
        rawdata[i,:,:] = batch[i][1]
    return label, rawdata


def farthest_point_sample(point, npoint):
    point = point.numpy()
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return torch.tensor(point).cuda()


def index2coord(selected_values, up_rate):
    cfgs = get_config()
    mask = selected_values != 0
    non_zero_indices = mask.nonzero(as_tuple=False).float()
    range_start = 0
    range_step  = cfgs.res_range/up_rate
    elevation_start = -int(cfgs.size_elevation/cfgs.res_elevation/2)
    elevation_step  = cfgs.res_elevation/up_rate
    azimuth_start = -int(cfgs.size_azimuth/cfgs.res_azimuth/2)
    azimuth_step  = cfgs.res_azimuth/up_rate
    start = torch.tensor([range_start, elevation_start, azimuth_start]).cuda()
    step  = torch.tensor([range_step,  elevation_step,  azimuth_step]).cuda()
    transformed_indices = start + non_zero_indices*step
    coords = transformed_indices
    return coords


def xyz2rea(points):
    if points.ndim == 1:
        points = points.reshape(1,-1)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    r = np.sqrt(np.sum(points**2, axis=1))
    e = np.degrees(np.arctan2(z, np.sqrt(x**2+y**2)))
    a = np.degrees(np.arctan2(-y, x))
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


def index2coord_trans(non_zero_indices, up_rate):
    cfgs = get_config()
    range_start = 0
    range_step  = cfgs.res_range/up_rate
    elevation_start = -int((cfgs.size_elevation_all-cfgs.size_elevation)/cfgs.res_elevation/2)
    elevation_step  = cfgs.res_elevation/up_rate
    azimuth_start = -int(cfgs.size_azimuth/cfgs.res_azimuth/2)
    azimuth_step  = cfgs.res_azimuth/up_rate
    start = torch.tensor([range_start, elevation_start, azimuth_start]).cuda()
    step  = torch.tensor([range_step,  elevation_step,  azimuth_step]).cuda()
    transformed_indices = start + non_zero_indices*step
    return transformed_indices


def metric(generated_points, label_points):
    cfgs = get_config()
    g_tree = cKDTree(generated_points)
    l_tree = cKDTree(label_points)
    
    clutter = 0
    for i, point in enumerate(generated_points):
        if len(l_tree.query_ball_point(point, cfgs.delta1)) == 0:
            clutter += 1
    
    density = 0
    for i, point in enumerate(label_points):
        if len(g_tree.query_ball_point(point, cfgs.delta2)) > 0:
            density += 1

    density /= label_points.shape[0]
    clutter /= generated_points.shape[0]
    accuracy = 1 - clutter
    return accuracy, density