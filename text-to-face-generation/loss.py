import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

# reconstruction loss
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    def forward(self, recon_x, x):
        d = (0,1) if len(recon_x.shape) == 2 else (0,1,2)
        return torch.sum(self.mse_loss(recon_x, x), dim=d)

# latent space vector loss
class LatentLoss(nn.Module):
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
        total_loss = torch.tensor([0.0], requires_grad=True)
        for loss in self.losses:
            total_loss += loss(output, target)
        return total_loss
