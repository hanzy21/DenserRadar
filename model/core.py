import torch.nn as nn
from .model_utils import *
from utils.common_utils import *
    

class UNet_final(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_final, self).__init__()
        self.cfgs = get_config()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.up_rate = self.cfgs.up_rate

        self.inc1 = InConv3D(n_channels, 32)
        self.inc2 = InConv3D(32, 64)
        self.down1 = DownConv3D5(64, 128, drop = self.cfgs.drop_rate)
        self.down2 = DownConv3D5(128, 256, drop = self.cfgs.drop_rate)
        self.down3 = DownConv3D5(256, 512, drop = self.cfgs.drop_rate)
        self.up3 = UpConv3D5(512, 256, drop = self.cfgs.drop_rate)
        self.up2 = UpConv3D5(256, 128, drop = self.cfgs.drop_rate)
        self.up1 = UpConv3D5(128, 64, drop = self.cfgs.drop_rate)
        self.outc4 = OutConv3D(512, n_classes, self.up_rate)
        self.outc3 = OutConv3D(256, n_classes, self.up_rate)
        self.outc2 = OutConv3D(128, n_classes, self.up_rate)
        self.outc1 = OutConv3D(64, n_classes, self.up_rate)
        
        self.att2 = Attention_block(F_g = 128, F_l = 256, F_int = 64)
        self.att3 = Attention_block(F_g = 256, F_l = 512, F_int = 128)


    def forward(self, x):
        B, D, R, E, A = x.size()
        loc_r = torch.linspace(-1.0, 1.0, R).cuda().view(1, 1, R, 1, 1).repeat(B, 1, 1, E, A)
        loc_e = torch.linspace(-1.0, 1.0, E).cuda().view(1, 1, 1, E, 1).repeat(B, 1, R, 1, A)
        loc_a = torch.linspace(-1.0, 1.0, A).cuda().view(1, 1, 1, 1, A).repeat(B, 1, R, E, 1)
        x = torch.cat((loc_r, loc_e, loc_a, x), 1)
        x1 = self.inc1(x)
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.att3(x4, x3)
        x = self.up3(x, x3)
        logits3 = self.outc3(x)
        x = self.att2(x, x2)
        x = self.up2(x, x2)
        logits2 = self.outc2(x)
        x = self.up1(x, x1)
        logits1 = self.outc1(x)
        
        return [logits1, logits2, logits3]
    
