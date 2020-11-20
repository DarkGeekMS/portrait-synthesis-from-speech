from stylegan2_generator import StyleGAN2Generator
from text_processing.bert.inference import BERTMultiLabelClassifier
from latent_manipulation import manipulate_latent

import torch
import torch.nn as nn
from numba import cuda
import numpy as np
import skimage.io as io
import skimage.transform as transform
import argparse
import pickle
import json
import os

def generate_faces(config):
    pass

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-cfg', '--config', type=str, help='path to config file of pipeline parameters', default='configs/generation_config.json')

    args = argparser.parse_args()

    # read JSON config
    with open(args.config) as f:
        config = json.load(f)

    # folders creation
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    # run face generation function
    generate_faces(config)
