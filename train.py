
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.init as init
import torch.utils.data

from utils.loss import *
from utils.dataset import KRadar
from utils.common_utils import *
from model.core import *
import matplotlib.pyplot as plt

import open3d as o3d
from os import path
import random


class Trainer:
    def __init__(self):
        self.cfgs = get_config()
        self.data = KRadar()
        self.data_split()
        self.min_eval_loss = 100000
        self.model = UNet_final(67,1)
        self.criterion = WholeLoss()
        self.model = torch.nn.DataParallel(self.model.cuda())
        total = sum([param.nelement() for param in self.model.parameters()])
        print('Params: %.2fM' % (total / 1e6))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfgs.base_lr, weight_decay=self.cfgs.weight_decay)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                init.xavier_uniform_(param)
            else:
                init.constant_(param, 0)


    def main(self):
        
        self.train_loss = []
        self.train_points = []
        self.eval_loss = []
        self.eval_points = []
        
        self.d_loss_list = []
        self.g_loss_list = []
        self.g_loss_vanilla_list = []
        
        if self.cfgs.load_checkpoint and path.exists('Model.pkl'):
            print("LOAD MODEL")
            self.model.load_state_dict(torch.load('./Model.pkl'))
        
        for i in tqdm(range(1,self.cfgs.epoch)):
            self.train(i)
                
            if i % self.cfgs.epochs_per_val == 0:
                self.eval(i)
            if i % self.cfgs.epochs_per_trans == 0:
                self.val(i)
                self.transform(i)
    
    
    def train(self, epoch):
        self.model.train()
        Loss = []
        points = []
        num = 0
        for label, rawdata, name in tqdm(self.train_loader):
            plot = random.random()
            self.model.train()
            num += 1
            label = label.float().cuda()
            rawdata = rawdata.float().cuda()
            out = self.model(rawdata)
            loss = self.criterion(out, label)
            
            Loss.append(loss.item())
            self.train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.train_loss.append(np.mean(np.array(Loss)))
        print("\n Train Loss: {:.4}".format(np.mean(np.array(Loss))))
        
        plt.figure()
        plt.plot(self.train_loss, label="Train_loss")
        plt.savefig("loss/train_loss.jpg")
        plt.clf()
        plt.close('all')
        
        
    def eval(self, epoch):
        self.model.eval()
        Loss = []
        for label, rawdata, name in self.test_loader:
            label = label.float().cuda()
            rawdata = rawdata.float().cuda()
            out = self.model(rawdata)
            loss = self.criterion(out, label)
            Loss.append(loss.item())
            self.print_grad_norms(self.model)
           
        self.eval_loss.append(np.mean(np.array(Loss)))
        print("Eval  Loss: {:.4} \n".format(np.mean(np.array(Loss))))
        
        plt.plot(self.eval_loss, label="Eval_loss")
        plt.savefig("loss/eval_loss.jpg")
        plt.clf()
        plt.close('all')
        
        if self.eval_loss[-1] < self.min_eval_loss:
            torch.save(self.model.state_dict(), 'Model.pkl')
            print("Save Model!")
            self.min_eval_loss = self.eval_loss[-1]
        
        
    def transform(self, epoch):
        self.model.eval()
        num = 0
        for label, rawdata, name in self.train_loader:
            name = name[0][-8:-3]
            label = label.float().cuda()
            rawdata = rawdata.float().cuda()
            out = self.model(rawdata)
            out = torch.sigmoid(out[0])
            
            for thresh in self.cfgs.transform_thresh:
                for i in range(out.shape[0]):
                    num += 1
                    selected = torch.where(out[i,0,:,:,:] > thresh, out[i,0,:,:,:], 0)
                    selected = torch.nonzero(selected)
                    if selected.shape[0] > 100000 or selected.shape[0] < 10:
                        continue
                    transformed_points = index2coord_trans(selected, self.cfgs.up_rate).cpu()
                    transformed_points = rea2xyz(transformed_points)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(transformed_points)
                    o3d.io.write_point_cloud("./thresh_{}_name_{}.pcd".format(thresh, name), pcd)
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
                    o3d.visualization.draw_geometries([pcd, coordinate_frame])
        print("save transform")
        
        
    def data_split(self):
        num_train = (int)(self.cfgs.train_data*self.data.__len__())
        num_test = self.data.__len__() - num_train
        dataset_train, dataset_test = torch.utils.data.random_split(self.data, [num_train, num_test])
        print('train', len(dataset_train))
        print('test', len(dataset_test))
        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.cfgs.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.cfgs.batch_size, shuffle=True)
        
    
    def print_grad_norms(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None and 'weight' in name:
                norm = parameter.grad.norm()
                print(f"Grad norm of {name}: {norm}")
                
                
    def val(self, epoch):
        self.model.eval()
        for label, rawdata, _ in tqdm(self.test_loader):
            label = label.float().cuda()
            rawdata = rawdata.float().cuda()
            out = self.model(rawdata)
            out = torch.sigmoid(out[0])
            for i in range(out.shape[0]):
                label_points = torch.nonzero(label[i,0,:,:,:])
                label_points = index2coord_trans(label_points, self.cfgs.up_rate).cpu()
                label_points = rea2xyz(label_points)
                for j in range(1000):
                    thresh = j*0.001
                    generated_points = torch.where(out[i,0,:,:,:] > thresh, out[i,0,:,:,:], 0)
                    generated_points = torch.nonzero(generated_points)
                    generated_points = index2coord_trans(generated_points, self.cfgs.up_rate).cpu()
                    generated_points = rea2xyz(generated_points)
                    accuracy, density = metric(generated_points, label_points)
                    density = int(density*1000)
                    results[0,density] += 1
                    results[1,density] += accuracy
                    results[2,density] += generated_points.shape[0]
        non_zero_indices = np.nonzero(results[0,:])[0]
        results = results[:, non_zero_indices]
        results[1,:] /= results[0,:]
        results[2,:] /= results[0,:]
        results[0,:] = np.array(non_zero_indices)
        np.save("my_result.npy",results)
        
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()