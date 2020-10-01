import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import os

from dataset import FaceDataset, collate_fn
from infersent import InferSent
from stylegan2.stylegan2_lib import StyleGAN2Generator

## GLOBAL VARIABLES

# network parameters
word_emb_dim = 300
enc_lstm_dim = 256
pool_type = 'max'
dpout_model = 0.0

# training hyperparameters
initial_lr = 0.1
momentum = 0.9
weight_decay = 5e-4
num_epoch = 100
batch_size = 64
num_workers = 4

# reconstruction loss
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    def forward(self, recon_x, x):
        d = 1 if len(recon_x.shape) == 2 else (1,2,3)
        return torch.sum(self.mse_loss(recon_x, x), dim=d)

def train(dataset_path, w2v_path, model_version, network_pkl, truncation_psi, result_dir):
    # perform networks initialization and training
    # define infersent model
    print('Loading infersent model ...')
    infersent_params = {'word_emb_dim': word_emb_dim, 'enc_lstm_dim': enc_lstm_dim,
                    'pool_type': pool_type, 'dpout_model': dpout_model}
    infersent_model = InferSent(infersent_params)
    infersent_model.train()

    # define stylegan2 generator
    print('Loading stylegan2 generator ...')
    stylegan_gen = StyleGAN2Generator(network_pkl, truncation_psi, result_dir)

    # define log writer
    writer = SummaryWriter(logdir=os.path.join(result_dir, 'log'), comment='training log')

    # define optimizer
    optimizer = torch.optim.SGD(infersent_model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infersent_model.to(device)

    # define dataset
    train_dataset = FaceDataset(dataset_path, w2v_path, word_emb_dim, model_version)
    train_dataset.build_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True, collate_fn=collate_fn)
                                               
    # define losses
    latent_loss = nn.MSELoss()
    recons_loss = ReconstructionLoss()

    # training loop
    print(f'Training on {len(train_dataset)} samples')
    total_step = len(train_loader)
    for epoch in range(num_epoch):
        i = 0
        epoch_l_loss = 0.0
        epoch_r_loss = 0.0
        epoch_total_loss = 0.0
        viz_samples = []
        for embeds, seq_len, l_vecs, images in tqdm(train_loader):
            i += 1
            # data to device
            embeds = embeds.to(device)
            seq_len = seq_len.to(device)
            l_vecs = l_vecs.to(device)
            images = images.to(device)

            # forward pass
            out_embed = infersent_model((embeds, seq_len))
            out_embed = torch.unsqueeze(out_embed, 1)

            # latent loss
            l_loss = latent_loss(out_embed, l_vecs)

            # reconstruction loss
            rand_idx = np.random.randint(batch_size)
            r_loss = torch.tensor([0.0], requires_grad=True)
            for i in range(batch_size):
                recons_img = stylegan_gen.generate_images(out_embed[i])
                r_loss += recons_loss(recons_img.to(device), images[i])
                if i == rand_idx:
                    viz_samples.append(recons_img)
            
            # back-propagation on all losses
            total_loss = l_loss + r_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # add current losses to total epoch losses
            epoch_l_loss += l_loss.item()
            epoch_r_loss += r_loss.item()
            epoch_total_loss += total_loss.item()

            # write training logs to tensorboard writer
            if (i + 1) % 10 == 0:
                writer.add_scalar('latent loss', l_loss.item(), epoch*total_step+i)
                writer.add_scalar('reconstruction loss', r_loss.item(), epoch*total_step+i)
                writer.add_scalar('total loss', total_loss.item(), epoch*total_step+i)

        # print logs of total epoch losses
        print('Epoch [{}/{}], Total Epoch Loss: \n latent loss: {:.4f}, reconstruction loss: {:.4f}, total loss: {:.4f}'
                .format(epoch + 1, num_epoch, epoch_l_loss, epoch_r_loss, epoch_total_loss))

        # write visualization samples to tensorboard writer
        viz_data = np.stack(viz_samples, axis=0)
        writer.add_images('output samples', torch.from_numpy(viz_data), global_step=epoch+1)
            
        # LR scheduler step
        scheduler.step()
        # save model checkpoint per epoch
        torch.save(infersent_model, os.path.join(os.path.join(result_dir, 'models'), f'model_{epoch}.pt'))
    
    # save final model checkpoint
    print('Finalizing training process ...')
    torch.save(infersent_model, os.path.join(os.path.join(result_dir, 'models'), 'model_final.pt'))
    writer.close()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-dsp', '--dataset_path', type=str, help='root directory of chosen dataset')
    argparser.add_argument('-w2v', '--w2v_path', type=str, help='path to word2vec file')
    argparser.add_argument('-mv', '--model_version', type=int, help='model version : (1) GloVe (2) FastText', default=1)
    argparser.add_argument('-pkl', '--network_pkl', type=str, help='path to stylegan2 model pickle file')
    argparser.add_argument('-psi', '--truncation_psi', type=float, help='stylegan2 generator truncation psi', default=1.0)
    argparser.add_argument('-rd', '--result_dir', type=str, help='directory to save logs and stylegan2 generated images')

    args = argparser.parse_args()

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
        os.mkdir(os.path.join(args.result_dir, 'images'))
        os.mkdir(os.path.join(args.result_dir, 'log'))
        os.mkdir(os.path.join(args.result_dir, 'models'))

    train(args.dataset_path, args.w2v_path, args.model_version, args.network_pkl, args.truncation_psi, args.result_dir)
