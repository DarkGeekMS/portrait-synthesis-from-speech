"""Training script for sentence embedding NLP module"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import json
import cv2
import os

from dataset import FaceDataset, collate_fn
from sent_embed import SentEmbedEncoder
from stylegan2_generator import StyleGAN2Generator
from vgg import Vgg16
from loss import PixelwiseDistanceLoss, KLDLoss, LatentLoss

def train(network_config, train_config):
    # perform networks initialization and training
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define log writer
    writer = SummaryWriter(logdir=os.path.join(train_config['result_dir'], 'log'), comment='training log')

    # define dataset
    print('Building dataset ...')
    train_dataset = FaceDataset(train_config['dataset_path'], train_config['w2v_path'], network_config['emb_dim'], network_config['model_version'])
    train_dataset.build_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_config['batch_size'], num_workers=train_config['num_workers'],
                                               shuffle=True, collate_fn=collate_fn)

    # define sentence embedding model
    print('Loading sentence embedding model ...')
    sent_embed_model = SentEmbedEncoder(network_config)
    if os.path.isfile(train_config['model_path']):
        sent_embed_model.load_state_dict(torch.load(train_config['model_path']))
    sent_embed_model.train()
    sent_embed_model.to(device)

    # define stylegan2 generator
    print('Loading stylegan2 generator ...')
    stylegan_gen = StyleGAN2Generator(train_config['stylegan2_pkl'], train_config['truncation_psi'], train_config['result_dir'], network_config["extended_out"])

    # define VGG-16 model
    print('Loading VGG-16 pretrained model ...')
    vgg_model = Vgg16(requires_grad=False)
    vgg_model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(sent_embed_model.parameters(), lr=train_config['initial_lr'], weight_decay=train_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # define losses
    latent_loss = LatentLoss(losses_list=['kl'], reduction='mean')
    pixel_loss = PixelwiseDistanceLoss(reduction='mean')

    # training loop
    print(f'Training on {len(train_dataset)} samples')
    total_step = len(train_loader)
    for epoch in range(train_config['num_epoch']):
        i = 0
        epoch_l_loss = 0.0
        epoch_r_loss = 0.0
        epoch_p_loss = 0.0
        epoch_total_loss = 0.0
        viz_samples = []
        for embeds, l_vecs, images in tqdm(train_loader):
            i += 1
            # data to device
            embeds = embeds.to(device)
            l_vecs = l_vecs.to(device)
            images = images.to(device)

            # forward pass
            out_embed = sent_embed_model(embeds)

            # latent loss
            l_loss = latent_loss(out_embed, l_vecs)

            # reconstruction loss
            recons_imgs = stylegan_gen.generate_images(out_embed.cpu().detach().numpy())
            r_loss = pixel_loss(torch.div(torch.from_numpy(recons_imgs).to(device), 255.0).permute(0, 3, 1, 2), images)

            # perceptual loss
            recons_features = vgg_model(torch.div(torch.from_numpy(recons_imgs).to(device).permute(0, 3, 1, 2), 255.0))
            target_features = vgg_model(images)
            p_loss = pixel_loss(recons_features.relu1_2, target_features.relu1_2) + \
                    pixel_loss(recons_features.relu2_2, target_features.relu2_2) + \
                    pixel_loss(recons_features.relu3_3, target_features.relu3_3) + \
                    pixel_loss(recons_features.relu4_3, target_features.relu4_3)

            # add a visualization sample to samples list
            rand_idx = np.random.randint(train_config['batch_size'])
            if recons_imgs.shape[0] > rand_idx and i%train_config['save_interval'] == 0:
                viz_samples.append(recons_imgs[rand_idx].transpose(2, 1, 0))
            
            # back-propagation on all losses
            total_loss = l_loss + r_loss + p_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # add current losses to total epoch losses
            epoch_l_loss += l_loss.item()
            epoch_r_loss += r_loss.item()
            epoch_p_loss += p_loss.item()
            epoch_total_loss += total_loss.item()

            # write training logs to tensorboard writer
            writer.add_scalar('latent_loss', l_loss.item(), epoch*total_step+i)
            writer.add_scalar('reconstruction_loss', r_loss.item(), epoch*total_step+i)
            writer.add_scalar('perceptual_loss', p_loss.item(), epoch*total_step+i)
            writer.add_scalar('total_loss', total_loss.item(), epoch*total_step+i)

        # print logs of total epoch losses
        print('Epoch [{}/{}], Total Epoch Loss: \n latent loss: {:.4f}, reconstruction loss: {:.4f}, perceptual loss: {:.4f}, total loss: {:.4f}'
                .format(epoch + 1, train_config['num_epoch'], epoch_l_loss, epoch_r_loss, epoch_p_loss, epoch_total_loss))

        # write visualization samples to tensorboard writer
        viz_data = np.stack(viz_samples, axis=0)
        writer.add_images('output_samples', torch.from_numpy(viz_data), global_step=epoch+1)
            
        # LR scheduler step
        scheduler.step()
        # save model checkpoint per epoch
        torch.save(sent_embed_model.state_dict(), os.path.join(os.path.join(train_config['result_dir'], 'models'), f'model_{epoch}.pt'))
    
    # save final model checkpoint
    print('Finalizing training process ...')
    torch.save(sent_embed_model.state_dict(), os.path.join(os.path.join(train_config['result_dir'], 'models'), 'model_final.pt'))
    writer.close()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-ncfg', '--network_config', type=str, help='path to config file of network parameters', default='configs/network_params.json')
    argparser.add_argument('-tcfg', '--train_config', type=str, help='path to config file of training parameters', default='configs/train_config.json')

    args = argparser.parse_args()
    
    # load config JSONs
    with open(args.network_config) as network_file:
        network_config = json.load(network_file)
    with open(args.train_config) as train_file:
        train_config = json.load(train_file)

    # create results directory
    results_dir = train_config['result_dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    # create experiment folder
    exp_name = f'exp-{len(os.listdir(results_dir))}'
    exp_dir = os.path.join(results_dir, exp_name)
    os.mkdir(exp_dir)
    os.mkdir(os.path.join(exp_dir, 'log'))
    os.mkdir(os.path.join(exp_dir, 'models'))

    # set new results directory
    train_config['result_dir'] = exp_dir

    # call main training driver
    train(network_config, train_config)
