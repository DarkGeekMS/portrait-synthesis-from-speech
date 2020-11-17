import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

import numpy as np

class FocalLoss(nn.Module):
    """
    Binary focal loss implementation
    """
    def __init__(self, alpha=1, gamma=2, reduce=True):
        # focal loss initialization
        super(FocalLoss, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        # forward pass
        # calculate BCE loss
        bce_loss = self.criterion(inputs, targets)
        # calculate Pt
        pt = torch.exp(-1 * bce_loss)
        # calculate focal loss (alpha * (1 - Pt)^gamma * BCE)
        f_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        # reduce dimensions using mean
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
