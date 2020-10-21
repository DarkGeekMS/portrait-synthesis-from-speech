import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

# reconstruction loss
class ReconstructionLoss(nn.Module):
    """Simple reconstruction loss for face images"""
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    def forward(self, recon_x, x):
        d = (0,1) if len(recon_x.shape) == 2 else (0,1,2)
        return torch.sum(self.mse_loss(recon_x, x), dim=d)

# latent space vector loss
class LatentLoss(nn.Module):
    """
    Custom loss for latent space vectors.
    One or a combination of :
        - L1Loss (l1).
        - MSELoss (mse).
        - KLDivLoss (kl).
        - Cosine Similarity (cosine).
    """
    def __init__(self, loss_list, reduction='mean'):
        super(LatentLoss, self).__init__()
        self.losses = list()
        if 'l1' in loss_list:
            self.losses.append(nn.L1Loss(reduction=reduction))
        if 'mse' in loss_list:
            self.losses.append(nn.MSELoss(reduction=reduction))
        if 'kl' in loss_list:
            self.losses.append(nn.KLDivLoss(reduction=reduction))
        if 'cosine' in loss_list:
            self.losses.append(nn.CosineSimilarity(dim=1))
    def forward(self, output, target):
        total_loss = self.losses[0](output, target)
        for idx in range(1, len(self.losses)):
            total_loss += self.losses[idx](output, target)
        return total_loss
