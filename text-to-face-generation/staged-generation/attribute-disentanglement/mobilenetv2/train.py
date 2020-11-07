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
from model import MobileNet

# train parameters
n_classes = 32
img_size = 768
num_epoch = 200
batch_size = 32
initial_lr = 1e-3
save_interval = 10
num_workers = 4

def train(faces_root, pickle_file):
    # perform networks initialization and training
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define tensorboard log writer
    writer = SummaryWriter(logdir='results/logs', comment='training log')

    # define MobileNet model
    print('Loading MobileNet pretrained model ...')
    model = MobileNet(n_classes, pretrained=True)
    model.train()
    model.to(device)

    # define image transforms
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # define faces dataset and dataloader
    print('Building dataset ...')
    train_dataset = FaceDataset(faces_root, pickle_file, preprocess)
    train_db, val_db = torch.utils.data.random_split(train_dataset, [len(train_dataset)-2000, 2000])
    train_loader = torch.utils.data.DataLoader(dataset=train_db,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_db,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)

    # define binary cross-entropy loss
    criterion = nn.BCEWithLogitsLoss()

    # define ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.75*num_epoch), int(0.9*num_epoch)], gamma=0.1)

    # freeze MobileNet feature extractor (for first half of epochs)
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # define number of training and validation steps
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    # main training loop
    print(f'Training on {len(train_db)} samples / Validation on {len(val_db)} samples')
    for epoch in range(num_epoch):
        # unfreeze MobileNet feature extractor after first half of epochs
        if epoch == 0.5*num_epoch:
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
        i = 0
        train_losses = []
        for images, labels in tqdm(train_loader):
            i += 1
            # data to device
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            output = model(images)
            # loss calculation
            loss = criterion(output, labels)
            train_losses.append(loss.item())
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training loss to log writer
            writer.add_scalar('train_loss', loss.item(), epoch*train_steps+i)
        # calculate average training loss
        total_train_loss = np.mean(np.array(train_losses))
        print("epoch:{:2d} training loss:{:.3f}".format(epoch, total_train_loss))
        # model validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            j = 0
            for images, labels in tqdm(val_loader):
                j += 1
                # data to device
                images = images.to(device)
                labels = labels.to(device)
                # forward pass
                output = model(images)
                # loss calculation
                loss = F.binary_cross_entropy(output, labels)
                val_losses.append(loss.item())
                # validation loss to log writer
                writer.add_scalar('val_loss', loss.item(), epoch*val_steps+j)
        # calculate average validation loss
        total_val_loss = np.mean(np.array(val_losses))
        print("epoch:{:2d} validation loss:{:.3f}".format(epoch, total_val_loss))
        # model back to train mode
        model.train()
        # LR scheduler step
        scheduler.step()
        # save model state dictionary every epoch interval
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'results/models/epoch_{epoch}_ckpt.pth')
    # save final model state dictionary
    print('Finalizing training process ...')
    torch.save(model.state_dict(), 'results/models/final_ckpt.pth')
    writer.close()

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-fr', '--faces_root', type=str, help='path to faces root directory', required=True)
    argparser.add_argument('-pkl', '--pickle_file', type=str, help='path to labels pickle file', required=True)

    args = argparser.parse_args()

    # create required folders
    if not os.path.isdir('results/'):
        os.mkdir('results/')
        os.mkdir('results/models')
        os.mkdir('results/logs')

    # call main training driver
    train(args.faces_root, args.pickle_file)
