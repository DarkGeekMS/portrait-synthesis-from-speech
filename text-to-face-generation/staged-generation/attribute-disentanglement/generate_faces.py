from stylegan2_generator import StyleGAN2Generator
from mobilenetv2.model import MobileNet
from bert.inference import BERTMultiLabelClassifier
from navigation.latent_manipulation import get_feature_axes, manipulate_latent, get_target_latent_vector

import torch
import torch.nn as nn
import numpy as np
import skimage.io as io
import skimage.transform as transform
import argparse
import pickle
import json
import os

def generate_single_pair(num_samples, mobilenet_weights, stylegan_pkl, n_classes=32, truncation_psi=0.5, initial_seed=10000):
    # generate a single random latent vector - image logits pair
    # generate a random latent vector
    z = np.random.RandomState(initial_seed).randn(1, 512)
    # initialize StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(stylegan_pkl, truncation_psi=truncation_psi)
    # generate random face
    random_face = stylegan2_generator.generate_images(z)[0]
    # resize output face image
    random_face_resized = transform.resize(random_face, (768, 768))
    random_face_resized = np.expand_dims(random_face_resized, axis=0)
    # deallocate StyleGAN2 generator
    del stylegan2_generator
    # initialize MobileNetv2 model
    mobilenet_model = MobileNet(n_classes, pretrained=False)
    mobilenet_model.load_state_dict(torch.load(mobilenet_weights))
    mobilenet_model.eval()
    mobilenet_model.cuda()
    # get logits of faces images
    image_logits = mobilenet_model(torch.div(torch.from_numpy(random_face_resized).cuda().permute(0, 3, 1, 2), 255.0))
    image_logits = image_logits.cpu()
    # deallocate MobileNetv2 model
    del mobilenet_model
    # return random latent vector - image logits pair
    return z[0], image_logits[0]

def extract_feature_axes(num_samples, mobilenet_weights, stylegan_pkl, n_classes=32, truncation_psi=0.5, initial_seed=10000):
    # generate random face samples and get the feature axes matrix (512 X n_classes)
    # generate random latent vectors (num_samples X 512)
    random_latent = []
    for i in range(num_samples):
        seed = np.random.randint((i + 1) * initial_seed)
        z = np.random.RandomState(seed).randn(1, 512)
        random_latent.append(z)
    random_latent = np.concatenate(random_latent, axis=0)
    # initialize StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(stylegan_pkl, truncation_psi=truncation_psi)
    # generate random faces
    random_faces = stylegan2_generator.generate_images(random_latent)
    # resize all output face images
    random_faces_resized = []
    for face in random_faces:
        face_resized = transform.resize(face, (768, 768))
        random_faces_resized.append(face_resized)
    random_faces_resized = np.stack(random_faces_resized, axis=0)
    # deallocate StyleGAN2 generator
    del stylegan2_generator
    # initialize MobileNetv2 model
    mobilenet_model = MobileNet(n_classes, pretrained=False)
    mobilenet_model.load_state_dict(torch.load(mobilenet_weights))
    mobilenet_model.eval()
    mobilenet_model.cuda()
    # get logits of faces images
    image_logits = mobilenet_model(torch.div(torch.from_numpy(random_faces_resized).cuda().permute(0, 3, 1, 2), 255.0))
    image_logits = image_logits.cpu()
    # fit feature axes matrix using multilabel logistic regression
    feature_axes_matrix = get_feature_axes(random_latent, image_logits)
    # deallocate MobileNetv2 model
    del mobilenet_model
    # return random latent vectors and corresponding logits, along with feature axes matrix
    return random_latent, image_logits, feature_axes_matrix

def generate_faces_from_text(text_desc, random_latent, image_logits, feature_axes_matrix, stylegan_pkl, truncation_psi=0.5):
    # generate face images from given text descriptions, random logits and feature axes matrix
    # define BERT model
    bert_model = BERTMultiLabelClassifier()
    # process text descriptions using BERT
    text_output = []
    for sent in text_desc:
        text_output.append(bert_model.predict(sent))
    # deallocate BERT model
    del bert_model
    # define StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(stylegan_pkl, truncation_psi=truncation_psi)
    # loop over each text description
    for idx, text_logits in enumerate(text_output):
        # manipulate random latent vector to get target latent vector
        target_latent = manipulate_latent(random_latent, image_logits, text_logits, feature_axes_matrix)
        target_latent = np.expand_dims(target_latent, axis=0)
        # generate and save final face image
        face_image = stylegan2_generator.generate_images(target_latent)
        io.imsave(f'results/{idx}.png', face_image[0])
    # deallocate StyleGAN2 generator
    del stylegan2_generator

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-cfg', '--config', type=str, help='path to config file of pipeline parameters', default='configs/config.json')

    args = argparser.parse_args()

    # read JSON config
    with open(args.config) as f:
        config = json.load(f)

    # folders creation
    if not os.path.isdir('results'):
        os.mkdir('results')

    # operation selection
    if config['operation'] == 'complete':
        # run complete pipeline
        # get feature axes matrix
        random_latent, image_logits, feature_axes_matrix = extract_feature_axes(config['num_samples'], config['mobilenet_weights'],
                                                                                config['stylegan_pkl'], config['n_classes'],
                                                                                config['truncation_psi'], config['seed'])
        # read text descriptions
        sent_list = []
        with open(config['text_desc']) as f:
            for line in f:
                line = line.strip()
                sent_list.append(line)
        # get random latent vector - image logits pair
        idx = np.random.randint(random_latent.shape[0])
        # perform text-to-face generation
        generate_faces_from_text(sent_list, random_latent[idx], image_logits[idx], feature_axes_matrix, config['stylegan_pkl'], config['truncation_psi'])
    
    elif config['operation'] == 'fit':
        # run and save feature axes matrix fitting
        # get feature axes matrix
        random_latent, image_logits, feature_axes_matrix = extract_feature_axes(config['num_samples'], config['mobilenet_weights'],
                                                                                config['stylegan_pkl'], config['n_classes'],
                                                                                config['truncation_psi'], config['seed'])
        # save features axes matrix as pickle
        with open(os.path.join(config['axes_matrix_dir'], 'axes_mat.pkl'), 'wb') as f:
            pickle.dump(feature_axes_matrix, f)
    
    elif config['operation'] == 'generate':
        # run latent manipulation and text-to-face generation
        # read feature axes matrix pickle
        with open(os.path.join(config['axes_matrix_dir'], 'axes_mat.pkl'), 'rb') as f:
            feature_axes_matrix = pickle.load(f)
        # read text descriptions
        sent_list = []
        with open(config['text_desc']) as f:
            for line in f:
                line = line.strip()
                sent_list.append(line)
        # generate a single random latent vector - image logits pair
        random_latent, image_logits = generate_single_pair(config['num_samples'], config['mobilenet_weights'],
                                                        config['stylegan_pkl'], config['n_classes'],
                                                        config['truncation_psi'], config['seed'])        
        # perform text-to-face generation
        generate_faces_from_text(sent_list, random_latent, image_logits, feature_axes_matrix, config['stylegan_pkl'], config['truncation_psi'])
