import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm
import numpy as np
import argparse

from dataset import FaceDataset
from model import FaceClassifier

# evaluation parameters
backbone = 'mobilenetv2'
n_classes = 32
img_size = 224
batch_size = 64
num_workers = 4

def evaluate(faces_root, pickle_file, weights_file):
    # perform networks initialization and evaluation
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define multi-label face classifier model
    print('Loading Face Classifier model ...')
    model = FaceClassifier(n_classes, backbone=backbone)
    model.load_state_dict(torch.load(weights_file))
    model.eval()
    model.to(device)

    # define image transforms
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # define faces dataset and dataloader
    print('Building dataset ...')
    eval_dataset = FaceDataset(faces_root, pickle_file, preprocess)
    data_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)

    # main evaluation loop
    print(f'Evaluating on {len(eval_dataset)} samples')
    correct = 0 # correct labels count
    for images, labels in tqdm(data_loader):
        # data to device
        images = images.to(device)
        labels = labels.to(device)
        # forward pass
        output = model(images)
        # correct labels calculation
        result = output > 0.5
        correct += (result == labels).sum().item()
    # calculate total accuracy
    total_acc = correct / (len(eval_dataset) * n_classes)
    print('Average accuracy: {:.3f}%'.format(total_acc * 100))

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-fr', '--faces_root', type=str, help='path to faces root directory', required=True)
    argparser.add_argument('-pkl', '--pickle_file', type=str, help='path to labels pickle file', required=True)
    argparser.add_argument('-wf', '--weights_file', type=str, help='path to model weights file', required=True)

    args = argparser.parse_args()

    # call main training driver
    evaluate(args.faces_root, args.pickle_file, args.weights_file)
