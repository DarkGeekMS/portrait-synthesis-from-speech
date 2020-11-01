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
    """StyleGAN2 generator class, used for NLP module training"""
    def __init__(self, network_pkl, truncation_psi=1.0, result_dir='results/nlp-training'):
        # initialize network and other class attributes
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = pretrained_networks.load_networks(network_pkl)
        self.truncation_psi = truncation_psi
        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

    def generate_images(self, latent_vector):
        # generate face image from given latent vector
        noise_seed = np.random.randint(10000)
        noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if self.truncation_psi is not None:
            Gs_kwargs.truncation_psi = self.truncation_psi

        rnd = np.random.RandomState(noise_seed)
        z = latent_vector # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
        return images
