import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from model import SpeechRecognition
from utils import IterMeter , TextTransform, data_processing , GreedyDecoder, cer, wer

# define the hyperparameters

BatchSize = 1
epochs = 10
lr = 0.0001
numWorkers = 1

device = "cuda" if torch.cuda.is_available() else "cpu"


NoClasses = 29
NoCNNs = 3
NoRNNs = 5
Nofeatrues = 64
RNNCells = 512



# TODO
def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):

    model.train()
    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            optimizer.step()
            scheduler.step()
            iter_meter.step()

            torch.save(
            {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 
            "./home/monda/Documents/ForthYear/gp/pytorchSpeech")

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))



def Validate(model, device, validation_loader, criterion, text_transform, epoch, iter_meter):
    print('\n ----------------------------------- evaluating -------------------------------------------------')
    model.eval()

    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(validation_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(validation_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths, text_transform)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def main():

    torch.manual_seed(7)

    text_transform = TextTransform()

    # create path for the dataset
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # load dataset
    traing_dataset  = torchaudio.datasets.LIBRISPEECH("./data", url='train-clean-100', download=True)
    testing_dataset = torchaudio.datasets.LIBRISPEECH("./data", url='test-clean', download=True)

    # data loaders
    training_loader = data.DataLoader(dataset=traing_dataset, batch_size=BatchSize, shuffle=True, collate_fn=lambda x: data_processing(x,text_transform , 'train'), num_workers = numWorkers, pin_memory = True)

    testing_loader = data.DataLoader(dataset=testing_dataset, batch_size=BatchSize, shuffle=False, collate_fn=lambda x: data_processing(x, text_transform, 'valid'), num_workers = numWorkers, pin_memory = True )

    # create the model
    model = SpeechRecognition( CNN_number=NoCNNs, RNN_number=NoRNNs, RNNCells=RNNCells, NoClasses=NoClasses, features=Nofeatrues, dropOut=0.1).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # create the loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr= lr,  steps_per_epoch=int(len(training_loader)), epochs=epochs, anneal_strategy='linear')

    iter_meter = IterMeter()

    # start training and validating
    for epoch in range(1, epochs + 1):
        train(model, device, training_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        Validate(model, device, testing_loader, criterion,text_transform , epoch, iter_meter)
        


