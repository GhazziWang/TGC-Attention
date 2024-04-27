# Define the loss function
import torch.nn as nn
import torch
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true)**2))
    
def MSE_loss(output, target):
    return F.mse_loss(output, target)

def MAE_loss(output, target):
    return torch.mean(torch.abs(output - target))

def r2_score(output, target):
    mean_y_true = torch.mean(target)
    ss_tot = torch.sum((target - mean_y_true)**2)
    ss_res = torch.sum((target - output)**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2