"""
Perceptual path length (PPL) : measures the difference between consecutive images (their VGG16 embeddings).
Drastic changes mean that multiple features have changed together and that they might be entangled.
"""

import torch
import lpips

class PerceptualSimilarity:
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    ...

    Attributes
    ----------
    net : string
        name of feature extraction network [vgg | alex | squeeze]
    """
    def __init__(self, net='vgg'):
        # initialize metric function
        self.loss_fn = lpips.LPIPS(net=net)

    def calculate_lpips(self, img1, img2):
        # compute LPIPS score between two input images
        # normalize input images between -1 and 1
        img1 = (img1-127.5) / 127.5
        img2 = (img2-127.5) / 127.5
        # convert images to torch tensors
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        # transpose image tensors
        img1 = img1.permute(2, 0, 1).float()
        img2 = img2.permute(2, 0, 1).float()
        # return LPIPS score between two images
        return self.loss_fn(img1, img2)
