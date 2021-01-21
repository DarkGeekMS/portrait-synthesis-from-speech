import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys
import torch
import tqdm

import pretrained_networks

from stylegan2_modules import Generator

#----------------------------------------------------------------------------

class StyleGAN2GeneratorTF(object):
    """StyleGAN2 generator class, used for NLP module training"""
    def __init__(self, network_pkl, truncation_psi=1.0, result_dir='results/nlp-training', feed_extended=False):
        # initialize network and other class attributes
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = pretrained_networks.load_networks(network_pkl)
        self.truncation_psi = truncation_psi
        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.feed_extended = feed_extended

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
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

        if(not self.feed_extended):
            z = latent_vector # [minibatch, component]
            images = self.Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]

        else:
            w = latent_vector # [minibatch, n_vectors, component]
            images = self.Gs.components.synthesis.run(w, **Gs_kwargs) # [minibatch, height, width, channel]
        
        return images

class StyleGAN2GeneratorPT(object):
    def __init__(self, checkpoint, truncation_psi, result_dir='results/nlp-training'):
         # initialize network and other class attributes
        print('Loading networks from "%s"...' % checkpoint)
        self.truncation_psi = truncation_psi
        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.generator = Generator(size=1024, style_dim=512, n_mlp= 8, channel_multiplier=2).cuda()

        checkpoint = torch.load(checkpoint)

        self.generator.load_state_dict(checkpoint["g_ema"])

    
    def generate(self, inp, feed_extended=False):

        if self.truncation_psi < 1:
            with torch.no_grad():
                mean_latent = self.generator.mean_latent(4096)
        else:
            mean_latent = None

        with torch.no_grad():
            self.generator.eval()

            gen_img , _ = self.generator(
                [inp], truncation=self.truncation_psi, truncation_latent=mean_latent
            )

            return gen_img



        

    
