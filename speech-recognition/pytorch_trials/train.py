import torch
import torch.nn as nn
import numpy as np
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from model import SpeechRecognition
from utils import IterMeter

# define the hyperparameters

BatchSize = 1
epochs = 10
lr = 0.0001
training_dir = ""
validating_dir = ""

device = "cuda" if torch.cuda.is_available() else "cpu"


NoClasses = 29
NoCNNs = 3
NoRNNs = 5
Nofeatrues = 64
RNNCells = 512

# TODO
def train(model, device, train_loader, criterion, optimizer, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)


# TODO
def Validate(model, device, validation_loader, criterion, epoch, iter_meter):
    print('\nevaluating...')
    model.eval()


def main():

    # load the data   -- TODO

    # create the model
    model = SpeechRecognition( CNN_number=NoCNNs, RNN_number=NoRNNs, RNNCells=RNNCells, NoClasses=NoClasses, features=Nofeatrues, dropOut=0.1).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # create the loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CTCLoss(blank=28).to(device)

    # start training and validating


