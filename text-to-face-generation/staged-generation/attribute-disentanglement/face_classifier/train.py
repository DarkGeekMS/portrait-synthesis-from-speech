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

# train parameters
n_classes = 32
img_size = 224
num_epoch = 100
batch_size = 48
num_valid = 20000
initial_lr = 1e-3
num_workers = 4

def train(faces_root, pickle_file):
    # perform networks initialization and training
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define tensorboard log writer
    writer = SummaryWriter(logdir='results/logs', comment='training log')

    # define multi-label face classifier model
    print('Loading Face Classifier model ...')
    model = FaceClassifier(n_classes, pretrained=True)
    model.train()
    model.to(device)

    # define image transforms
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # define faces dataset and dataloader
    print('Building dataset ...')
    train_dataset = FaceDataset(faces_root, pickle_file, preprocess)
    train_db, val_db = torch.utils.data.random_split(train_dataset, [len(train_dataset)-num_valid, num_valid])
    train_loader = torch.utils.data.DataLoader(dataset=train_db,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_db,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)

    # define binary cross-entropy loss
    criterion = nn.BCELoss()

    # define ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0)

    # define number of training and validation steps
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    # main training loop
    print(f'Training on {len(train_db)} samples / Validation on {len(val_db)} samples')
    max_val_acc = float('-inf')
    for epoch in range(num_epoch):
        i = 0
        correct = 0
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
            # correct labels calculation
            result = output > 0.5
            correct += (result == labels).sum().item()
            # training loss to log writer
            writer.add_scalar('train_loss', loss.item(), epoch*train_steps+i)
        # calculate average training loss
        total_train_loss = np.mean(np.array(train_losses))
        total_train_acc = correct / ((len(train_dataset)-num_valid) * 32)
        print("epoch:{:2d} training loss:{:.3f} training accuracy:{:.3f}%".format(epoch+1, total_train_loss, total_train_acc * 100.0))
        # model validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            correct = 0
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
                # correct labels calculation
                result = output > 0.5
                correct += (result == labels).sum().item()
                # validation loss to log writer
                writer.add_scalar('val_loss', loss.item(), epoch*val_steps+j)
        # calculate average validation loss
        total_val_loss = np.mean(np.array(val_losses))
        total_val_acc = correct / (num_valid * 32)
        print("epoch:{:2d} validation loss:{:.3f} validation accuracy:{:.3f}%".format(epoch+1, total_val_loss, total_val_acc * 100.0))
        # model back to train mode
        model.train()
        # LR scheduler step
        scheduler.step(total_val_loss)
        # save model state dictionary every epoch interval
        if total_val_acc > max_val_acc:
            print("Saving checkpoint at epoch {:2d}".format(epoch+1))
            torch.save(model.state_dict(), f'results/models/ckpt_best_epoch_{epoch+1}.pth')
            max_val_acc = total_val_acc
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
