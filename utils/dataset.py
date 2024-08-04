import os
import torch
from tqdm import tqdm

from utils.common_utils import *
from torch.utils.data import Dataset


class KRadar(Dataset):
    def __init__(self):
        self.cfgs = get_config()
        rawdata_name = self.cfgs.rawdata_folder
        label_name = self.cfgs.cube_folder
        self.label_list = []
        self.rawdata_list = []
        
        for segment in os.listdir(self.cfgs.data_root):
            segment_path = os.path.join(self.cfgs.data_root, segment)
            if os.path.isdir(segment_path):
                rawdata_path = os.path.join(segment_path, rawdata_name)
                label_path = os.path.join(segment_path, label_name)
                
                labels = sorted([file for file in os.listdir(label_path)])
                rawdatas = sorted([file for file in os.listdir(rawdata_path)])
                
                for name in labels:
                    if os.path.isfile(os.path.join(label_path, name)):
                        self.label_list.append(os.path.join(segment, label_name, name))
                        
                for name in rawdatas:
                    if os.path.isfile(os.path.join(rawdata_path, name)):
                        self.rawdata_list.append(os.path.join(segment, rawdata_name, name))
                        
        assert len(self.label_list) == len(self.rawdata_list)
        
        self.i_mean = 0
        self.i_std = 0
        for single_rawdata in tqdm(self.rawdata_list):
            raw_data = torch.tensor(torch.load(os.path.join(self.cfgs.data_root, single_rawdata)))
            print(raw_data.shape)
            
            self.i_mean += torch.mean(raw_data)
            self.i_std += torch.mean(raw_data**2)
        
        self.i_mean /= len(self.label_list)
        self.i_std = torch.sqrt(self.i_std/len(self.label_list)-self.i_mean**2)
        
    def __len__(self):
        return len(self.rawdata_list)
    
    def __getitem__(self, index):
        return self.get_single_sample(index)
    
    def get_single_sample(self, index):
        assert index < len(self.label_list)
        label = torch.load(os.path.join(self.cfgs.data_root, self.label_list[index]))
        label = label.unsqueeze(0)
        rawdata = torch.tensor(torch.load(os.path.join(self.cfgs.data_root, self.rawdata_list[index])))
        rawdata = (rawdata-self.i_mean)/self.i_std
        
        return label, rawdata, self.label_list[index]
    
        
        
