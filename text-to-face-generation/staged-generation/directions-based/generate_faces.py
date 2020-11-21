from stylegan2_generator import StyleGAN2Generator
from text_processing.bert.inference import BERTMultiLabelClassifier
from seed_generation import generate_seed
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
    # generate face images from text descriptions using directions-based latent navigation
    # read text descriptions
    sent_list = []
    with open(config['text_desc']) as f:
        for line in f:
            line = line.strip()
            sent_list.append(line)
    # initialize BERT multi-label classifier model
    bert_model = BERTMultiLabelClassifier()
    # get text logits for each sentence
    text_output = []
    for sent in sent_list:
        text_output.append(bert_model.predict(sent))
    # de-allocate BERT model
    del bert_model
    # free CUDA GPU memory
    device = cuda.get_current_device()
    device.reset()
    # initialize StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(config['stylegan_pkl'], truncation_psi=config['truncation_psi'], use_projector=False)
    # read pre-defined feature directions
    feature_directions = np.load(config['directions_npy'])
    # generate a random seed of extended latent vector and corresponding logits
    latent_vector, image_logits = generate_seed(feature_directions, stylegan2_generator, config['seed'])
    # loop over each text description to generate the corresponding face image
    for idx, text_logits in enumerate(text_output):
        # manipulate latent space to get the target latent vector
        target_latent = manipulate_latent(latent_vector, image_logits, text_logits, feature_directions)
        target_latent = np.expand_dims(target_latent, axis=0)
        # generate the required face image
        face_image = stylegan2_generator.generate_images(target_latent)
        # save the generated face image
        io.imsave(f'results/{idx}.png', face_image[0])
    # de-allocate StyleGAN2 generator
    del stylegan2_generator
    # free CUDA GPU memory
    device = cuda.get_current_device()
    device.reset()

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
