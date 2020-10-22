import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

# pixel-wise distance loss
class PixelwiseDistanceLoss(nn.Module):
    """Simple pixel-wise l2 distance loss for face images and feature maps"""
    def __init__(self, reduction='mean'):
        super(PixelwiseDistanceLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    def forward(self, output, target):
        return self.mse_loss(output, target)

# KL-divergence loss
class KLDLoss(nn.Module):
    """KL-Divergence loss with log softmax"""
    def __init__(self, reduction='batchmean'):
        super(KLDLoss, self).__init__()
        self.kld_loss_fn = nn.KLDivLoss(reduction=reduction)
    def forward(self, output, target):
        return self.kld_loss_fn(F.log_softmax(output, dim=1), F.softmax(target, dim=1))

# latent space vector loss
class LatentLoss(nn.Module):
    """
    Custom loss for latent space vectors.
    Can be one or a combination of the following losses :
        - L1Loss (l1).
        - MSELoss (mse).
        - KLDivLoss (kl).
        - Cosine Similarity (cosine).
    """
    def __init__(self, losses_list, reduction='mean'):
        super(LatentLoss, self).__init__()
        self.losses_fn = list()
        if 'l1' in losses_list:
            self.losses_fn.append(nn.L1Loss(reduction=reduction))
        if 'mse' in losses_list:
            self.losses_fn.append(nn.MSELoss(reduction=reduction))
        if 'kl' in losses_list:
            if reduction == 'mean':
                self.losses_fn.append(KLDLoss(reduction='batchmean'))
            else:
                self.losses_fn.append(KLDLoss(reduction=reduction))
        if 'cosine' in losses_list:
            self.losses_fn.append(nn.CosineSimilarity(dim=1))

    def forward(self, output, target):
        total_loss = self.losses_fn[0](output, target)
        for idx in range(1, len(self.losses_fn)):
            total_loss += self.losses_fn[idx](output, target)
        return total_loss
