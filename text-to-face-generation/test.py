"""Testing script for the whole text-to-face generation network"""

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import os
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from nltk.tokenize import word_tokenize

from infersent import InferSent
from stylegan2_generator import StyleGAN2Generator

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

def test(text_path, model_version, model_path, w2v_path, network_pkl, truncation_psi, result_dir):
    # perform text-to-face network inference
    # configure model variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_emb_dim = 300
    enc_lstm_dim = 256
    pool_type = 'max'
    dpout_model = 0.0
    if model_version == 1:
        bos = '<s>'
        eos = '</s>'
    else:
        bos = '<p>'
        eos = '</p>'

    # define infersent model
    print('Loading infersent model ...')
    max_pad = True if model_version == 1 else False
    infersent_params = {'word_emb_dim': word_emb_dim, 'enc_lstm_dim': enc_lstm_dim,
                    'pool_type': pool_type, 'dpout_model': dpout_model, 'max_pad': max_pad}
    infersent_model = InferSent(infersent_params)
    infersent_model.load_state_dict(torch.load(model_path))
    infersent_model.eval()
    infersent_model.to(device)

    # define stylegan2 generator
    print('Loading stylegan2 generator ...')
    stylegan_gen = StyleGAN2Generator(network_pkl, truncation_psi, result_dir)

    # read inference sentences
    sent_list = []
    with open(text_path) as f:
        for line in f:
            line = line.strip()
            sent_list.append(line)
    
    # prepare vocabulary and word2vec embeddings
    word_dict = get_word_dict(sent_list, model_version, bos, eos, tokenize)
    word_vec = get_w2v(word_dict, w2v_path)
    print('Vocab size : %s' % (len(word_vec)))

    # loop over each sentence to perform inference
    for idx, sent in enumerate(sent_list):
        # process text description to extract embeddings
        sentence = [bos] + tokenize(sent, model_version) + [eos]
        # filters words without w2v vectors
        s_f = [word for word in sentence if word in word_vec]
        if not s_f:
            warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                            Replacing by "</s>"..' % (sentence, idx))
            s_f = [eos]
        # extract embeddings from text description
        embed = np.zeros((len(s_f), 1, word_emb_dim))
        for i in range(len(s_f)):
            embed[i, 0, :] = word_vec[s_f[i]]
        # perform sentence embedding
        embed = torch.from_numpy(embed).float().to(device)
        seq_len = torch.from_numpy(np.array([embed.shape[0]])).long().to(device)
        out_embed = infersent_model((embed, seq_len))
        # generate face
        generated_face = stylegan_gen.generate_images(out_embed.cpu().detach().numpy())
        # save face image
        imsave(os.path.join(result_dir, f'face-images/face{idx}.png'), generated_face[0])        

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-tp', '--text_path', type=str, help='path to text file containing sentences to be tested', required=True)
    argparser.add_argument('-mv', '--model_version', type=int, help='model version : (1) GloVe (2) FastText', default=1)
    argparser.add_argument('-mp', '--model_path', type=str, help='path to initial weights of sentence embedding model', required=True)
    argparser.add_argument('-w2v', '--w2v_path', type=str, help='path to word2vec file', required=True)
    argparser.add_argument('-pkl', '--network_pkl', type=str, help='path to stylegan2 model pickle file', required=True)
    argparser.add_argument('-psi', '--truncation_psi', type=float, help='stylegan2 generator truncation psi', default=1.0)
    argparser.add_argument('-rd', '--result_dir', type=str, help='directory to save logs and stylegan2 generated images', default='results/')

    args = argparser.parse_args()

    # create results directory
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    
    # create experiment folder
    exp_name = f'exp-{len(os.listdir(args.result_dir))}'
    exp_dir = os.path.join(args.result_dir, exp_name)
    os.mkdir(exp_dir)
    os.mkdir(os.path.join(exp_dir, 'face-images'))

    # call main testing driver
    test(args.text_path, args.model_version, args.model_path, args.w2v_path, args.network_pkl, args.truncation_psi, exp_dir)
