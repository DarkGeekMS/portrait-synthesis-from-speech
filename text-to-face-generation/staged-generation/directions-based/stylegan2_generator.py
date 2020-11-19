import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys

import pretrained_networks

#----------------------------------------------------------------------------

class StyleGAN2Generator(object):
    """StyleGAN2 generator class, used for face generation"""
    def __init__(self, network_pkl, truncation_psi=1.0, use_projector=False):
        # initialize network and other class attributes
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = pretrained_networks.load_networks(network_pkl)
        self.truncation_psi = truncation_psi
        self.use_projector = use_projector

    def map_latent_vector(self, latent_vector):
        # map unextended latent vector (z) to extended latent vector (w)
        # initialize random state
        noise_seed = np.random.randint(10000)
        rnd = np.random.RandomState(noise_seed)
        # feed input noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        # define generator arguments
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if self.truncation_psi is not None:
            Gs_kwargs.truncation_psi = self.truncation_psi
        # project latent vector
        extended_latent_vector = self.Gs.components.mapping.run(latent_vector, None)
        return extended_latent_vector

    def generate_images(self, latent_vector):
        # generate face image from given latent vector
        # initialize random state
        noise_seed = np.random.randint(10000)
        rnd = np.random.RandomState(noise_seed)
        # feed input noise
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        # define generator arguments
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if self.truncation_psi is not None:
            Gs_kwargs.truncation_psi = self.truncation_psi
        # generate face image
        if self.use_projector == True:
            images = self.Gs.run(latent_vector, None, **Gs_kwargs)
        else:
            images = self.Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
        return images
