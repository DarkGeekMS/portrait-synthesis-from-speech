"""Testing script for the whole text-to-face generation network"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import os
import json
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from nltk.tokenize import word_tokenize

from sent_embed import SentEmbedEncoder
from stylegan2_generator import StyleGAN2GeneratorTF, StyleGANGeneratorPT

def tokenize(sent, model_version):
    # tokenize sentence
    if model_version == 2:
        sent = ' '.join(word_tokenize(sent))
        sent = sent.replace(" n't ", "n 't ")
        return sent.split()
    else:
        return word_tokenize(sent)

def get_word_dict(sentences, model_version, bos, eos, tokenize=True):
    # create vocab of words
    word_dict = {}
    sentences = [s.split() if not tokenize else tokenize(s, model_version) for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict[bos] = ''
    word_dict[eos] = ''
    return word_dict

def get_w2v(word_dict, w2v_path):
    # create word_vec with w2v vectors
    word_vec = {}
    with open(w2v_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.fromstring(vec, sep=' ')
    print('Found (%s/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
    return word_vec

def test(network_config, test_config):
    # perform text-to-face network inference
    # configure model variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if network_config['model_version'] == 1:
        bos = '<s>'
        eos = '</s>'
    else:
        bos = '<p>'
        eos = '</p>'

    # define sentence embedding model
    print('Loading sentence embedding model ...')
    sent_embed_model = SentEmbedEncoder(network_config)
    sent_embed_model.load_state_dict(torch.load(test_config['model_path']))
    sent_embed_model.eval()
    sent_embed_model.to(device)

    # define stylegan2 generator
    print('Loading stylegan2 generator ...')
    stylegan_gen = StyleGAN2Generator(test_config['stylegan2_pkl'], test_config['truncation_psi'], test_config['result_dir'])

    # read inference sentences
    sent_list = []
    with open(test_config['text_path']) as f:
        for line in f:
            line = line.strip()
            sent_list.append(line)
    
    # prepare vocabulary and word2vec embeddings
    word_dict = get_word_dict(sent_list, network_config['model_version'], bos, eos, tokenize)
    word_vec = get_w2v(word_dict, test_config['w2v_path'])
    print('Vocab size : %s' % (len(word_vec)))

    # loop over each sentence to perform inference
    for idx, sent in enumerate(sent_list):
        # process text description to extract embeddings
        sentence = [bos] + tokenize(sent, network_config['model_version']) + [eos]
        # filters words without w2v vectors
        s_f = [word for word in sentence if word in word_vec]
        if not s_f:
            warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                            Replacing by "</s>"..' % (sentence, idx))
            s_f = [eos]
        # extract embeddings from text description
        embed = np.zeros((len(s_f), 1, network_config['emb_dim']))
        for i in range(len(s_f)):
            embed[i, 0, :] = word_vec[s_f[i]]
        # perform sentence embedding
        embed = torch.from_numpy(embed).float().to(device)
        out_embed = sent_embed_model(embed)
        # generate face
        generated_face = stylegan_gen.generate_images(out_embed.cpu().detach().numpy())
        # save face image
        imsave(os.path.join(test_config['result_dir'], f'face-images/face{idx}.png'), generated_face[0])        

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-ncfg', '--network_config', type=str, help='path to config file of network parameters', default='configs/network_params.json')
    argparser.add_argument('-tcfg', '--test_config', type=str, help='path to config file of inference parameters', default='configs/test_config.json')

    args = argparser.parse_args()

    # load config JSONs
    with open(args.network_config) as network_file:
        network_config = json.load(network_file)
    with open(args.test_config) as test_file:
        test_config = json.load(test_file)

    # create results directory
    results_dir = test_config['result_dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    # create experiment folder
    exp_name = f'exp-{len(os.listdir(results_dir))}'
    exp_dir = os.path.join(results_dir, exp_name)
    os.mkdir(exp_dir)
    os.mkdir(os.path.join(exp_dir, 'face-images'))

    # set new results directory
    test_config['result_dir'] = exp_dir

    # call main testing driver
    test(network_config, test_config)
