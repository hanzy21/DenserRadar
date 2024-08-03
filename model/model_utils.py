import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common_utils import *

    
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.W_x = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(F_g, F_int, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
            nn.ReLU(),
            nn.Conv3d(F_int, F_int, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        self.ReLU = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.ReLU(g1 + x1)
        psi = self.psi(psi)
        out = g * psi
        return out
    
    
class InConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv3D, self).__init__()
        self.cfgs = get_config()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=2)
            )

    def forward(self, x):
        return self.conv(x)
    
    
class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, up_rate):
        super(OutConv3D, self).__init__()
        self.cfgs = get_config()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=up_rate, stride=up_rate),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=1, dilation=2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv3D5(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, drop=0.2):
        self.cfgs = get_config()
        super(DownConv3D5, self).__init__()
        hidden_channels = hidden_channels or out_channels
        self.conv_3d = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
            nn.Dropout(drop, inplace=True),
            nn.ReLU(),
            
            nn.Conv3d(hidden_channels, out_channels, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
            nn.Dropout(drop, inplace=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_3d(x)
    

class UpConv3D5(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, drop=0.2):
        super(UpConv3D5, self).__init__()
        hidden_channels = hidden_channels or out_channels
        self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv_3d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
            nn.Dropout(drop, inplace=True),
            nn.ReLU(),
            
            nn.Conv3d(hidden_channels, out_channels, kernel_size=(5,3,3), stride=1, padding=(2,1,1), bias=True),
            nn.Dropout(drop, inplace=True),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffR = x2.size()[2] - x1.size()[2]
        diffE = x2.size()[3] - x1.size()[3]
        diffA = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffA // 2, diffA - diffA // 2,
                        diffE // 2, diffE - diffE // 2,
                        diffR // 2, diffR - diffR // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_3d(x)
        return x