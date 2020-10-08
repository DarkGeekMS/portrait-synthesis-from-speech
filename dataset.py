import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import os
import warnings
import random
import pickle
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from nltk.tokenize import word_tokenize

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, w2v_path, word_emb_dim=300, model_version=1):
        # initialize dataset object and read data lists
        super(FaceDataset, self).__init__()
        self.dataset_path = dataset_path
        self.w2v_path = w2v_path
        self.word_emb_dim = word_emb_dim
        self.model_version = model_version
        self.img_path = os.path.join(self.dataset_path, "face-images")
        self.text_path = os.path.join(self.dataset_path, "text-desc")
        self.latent_path = os.path.join(self.dataset_path, "latent-vectors")
        assert self.model_version in [1, 2]
        if self.model_version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.model_version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def tokenize(self, s):
        # tokenize sentence
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")
            return s.split()
        else:
            return word_tokenize(s)

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found (%s/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def build_dataset(self, tokenize=True):
        # build Text2Face dataset and its vocab
        # list dataset files
        self.text_list = [os.path.join(self.text_path, text_file) for text_file in os.listdir(self.text_path)]
        self.latent_list = [os.path.join(self.latent_path, latent_file) for latent_file in os.listdir(self.latent_path)]
        self.img_list = [os.path.join(self.img_path, img_file) for img_file in os.listdir(self.img_path)]
        # read text descriptions
        sents = []
        for fpath in self.text_list:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    sents.append(line)
        # build vocab dictionary
        word_dict = self.get_word_dict(sents, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    def __getitem__(self, index):
        # get a dataset item with index
        # read face image
        img = cv2.imread(self.img_list[index])
        # read latent vctor
        l_vec = np.load(self.latent_list[index])
        # read text description (NOTE : only the first description is considered)
        with open(self.text_list[index]) as f:
            txt_desc = f.readline().strip()
        # process text description to extract embeddings
        sentence = [self.bos] + self.tokenize(txt_desc) + [self.eos]
        # filters words without w2v vectors
        s_f = [word for word in sentence if word in self.word_vec]
        if not s_f:
            warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                            Replacing by "</s>"..' % (sentence, index))
            s_f = [self.eos]
        # extract embeddings from text description
        embed = np.zeros((len(s_f), self.word_emb_dim))
        for i in range(len(s_f)):
            embed[i, :] = self.word_vec[s_f[i]]
        # return (
        # text embeddings -> (sentence_length x self.word_emb_dim), 
        # latent vector -> (1 x 512), 
        # face image -> (1024 x 1024 x 3))
        return (embed, l_vec, img)

    def __len__(self):
        # return dataset length
        return len(os.listdir(self.img_path))

def collate_fn(batch):
    # process batch and convert to tensors
    # list all batch samples
    embed_list, l_vec_list, img_list = [], [], []
    for _embed, _l_vec, _img in batch:
        embed_list.append(_embed)
        l_vec_list.append(_l_vec)
        img_list.append(_img)
    # sort batch by sentence length
    len_list = [embed.shape[0] for embed in embed_list]
    sorted_idx = sorted(range(len(len_list)), key=lambda k: len_list[k])
    sorted_embed_list, sorted_len_list, sorted_l_vec_list, sorted_img_list = [], [], [], []
    for idx in sorted_idx:
        sorted_embed_list.append(embed_list[idx])
        sorted_len_list.append(len_list[idx])
        sorted_l_vec_list.append(l_vec_list[idx])
        sorted_img_list.append(img_list[idx])
    # handle padding of text embeddings to max length
    embed_tensor = np.zeros((max(len_list), len(sorted_embed_list), len(sorted_embed_list[0][0])))
    for i in range(len(sorted_embed_list)):
        for j in range(len(sorted_embed_list[i])):
            embed_tensor[j, i, :] = sorted_embed_list[i][j]
    # convert other batch components to tensors
    len_tensor = np.array(sorted_len_list)
    l_vec_tensor = np.stack(sorted_l_vec_list, axis=0)
    img_tensor = np.stack(sorted_img_list, axis=0)
    # return torch tensors of the processed numpy arrays
    return (torch.from_numpy(embed_tensor).float(),
            torch.from_numpy(len_tensor).long(),
            torch.from_numpy(l_vec_tensor).float(),
            torch.from_numpy(img_tensor).float())
