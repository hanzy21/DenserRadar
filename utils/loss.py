import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        sigmoid_p = torch.sigmoid(inputs)
        p_t = targets * sigmoid_p + (1 - targets) * (1 - sigmoid_p)
        alpha_t = self.alpha
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = -alpha_t*(1-p_t)**self.gamma * torch.log(p_t+1e-8)
        return torch.mean(F_loss)
        
        
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, inputs, targets):
        targets = targets.unsqueeze(1)
        iflat = inputs.view(-1).float()
        tflat = targets.view(-1).float()
        intersection = (iflat * tflat).sum()
        loss = 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))
        return loss


class WholeLoss(nn.Module):
    def __init__(self):
        super(WholeLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        
    def forward(self, inputs, target):
        loss = 0
        for i in range(len(inputs)):
            input = inputs[i]
            B, C, R, E, A = input.shape
            target_d = F.interpolate(target, size=(R, E, A), 
                                    mode='trilinear', align_corners=False)
            loss += 1/(2**i)*(self.dice(input, target_d)+700*self.focal(input, target_d))
        return loss