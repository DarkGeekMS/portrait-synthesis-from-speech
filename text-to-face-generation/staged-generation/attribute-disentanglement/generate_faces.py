from stylegan2_generator import StyleGAN2Generator
from mobilenetv2.model import MobileNet
from bert.inference import bertMultiLabelClassifier
from navigation.latent_manipulation import get_feature_axes, manipulate_latent, get_target_latent_vector

import torch
import torch.nn as nn
import numpy as np
import skimage.transform as transform
import argparse

def extract_feature_axes(num_samples, mobilenet_weights, stylegan_pkl, n_classes=32, truncation_psi=0.5, seed=10000):
    # generate random face samples and get the feature axes matrix (512 X n_classes)
    # generate random latent vectors (num_samples X 512)
    random_latent = []
    for i in range(num_samples):
        seed = np.random.randint((i + 1) * seed)
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
    # return random latent vectors and corresponding logits, along with feature axes matrix
    return random_latent, image_logits, feature_axes_matrix

def generate_faces_from_text(text_desc, random_latent, image_logits, stylegan_pkl, truncation_psi=0.5):
    pass

if __name__ == "__main__":
    pass
