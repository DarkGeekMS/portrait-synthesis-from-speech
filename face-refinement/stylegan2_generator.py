import torch
import dnnlib
import numpy as np

import legacy

class StyleGAN2Generator:
    """StyleGAN2 generator class, used for face generation"""
    def __init__(
        self, network_pkl, truncation_psi=1.0, noise_mode='const',
        use_projector=False
    ):
        # initialize network and other class attributes
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self.use_projector = use_projector

    def map_latent_vector(self, latent_vector):
        # map unextended latent vector (z) to extended latent vector (w)
        # define empty label tensor
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        # convert latent vector to tensor
        latent_vector = torch.from_numpy(latent_vector).to(self.device)
        # project latent vector
        extended_latent_vector = self.G.mapping(
            latent_vector, label, truncation_psi=self.truncation_psi
        )
        return extended_latent_vector.cpu().detach().numpy()

    def generate_images(self, latent_vector):
        # generate face image from given latent vector
        # define empty label tensor
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        # convert latent vector to tensor
        latent_vector = torch.from_numpy(latent_vector).to(self.device)
        # generate face image
        if self.use_projector:
            images = self.G(
                latent_vector, label,
                truncation_psi=self.truncation_psi,
                noise_mode=self.noise_mode
            )
        else:
            images = self.G.synthesis(
                latent_vector, noise_mode=self.noise_mode
            )
        images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
        images[images < -1.0] = -1.0
        images[images > 1.0] = 1.0
        images = (images + 1.0) * 127.5
        images = images.astype(np.uint8)
        return images
