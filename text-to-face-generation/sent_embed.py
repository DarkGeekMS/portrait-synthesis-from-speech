import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
 
class SentEmbedEncoder(torch.nn.Module):
    """
    The class is an implementation of the paper `A Structured Self-Attentive Sentence Embedding` including regularization
    and without pruning. Slight modifications have been done for speedup
    """
    def __init__(self, config):
        # initialize sentence embedding model
        super(SentEmbedEncoder,self).__init__()
        self.lstm = torch.nn.LSTM(config["emb_dim"], config["lstm_hid_dim"], 1, batch_first=True, dropout=config["dpout_model"], bidirectional=True)
        self.linear_first = torch.nn.Linear(config["lstm_hid_dim"]*2, config["dense_hid_dim"])
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(config["dense_hid_dim"], config["att_hops"])
        self.linear_second.bias.data.fill_(0)
        self.linear_final = torch.nn.Linear(config["lstm_hid_dim"]*2, config["out_dim"])
        self.lstm_hid_dim = config["lstm_hid_dim"]
        self.r = config["att_hops"]
    
    def softmax(self,input, axis=1):
        # softmax applied to axis=n
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=axis)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
    
    def forward(self,x):
        # forward pass
        outputs, _ = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention@outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        return self.linear_final(avg_sentence_embeddings)
