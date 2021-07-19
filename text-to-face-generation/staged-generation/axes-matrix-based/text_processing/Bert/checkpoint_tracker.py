import os
import torch
from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
from model import BertRegressor

class CheckpointTracker():
    def __init__(self, architecture, device):
        self.architecture = architecture
        self.device = device
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), './checkpoints/' + self.architecture + '.pth')
    
    def load_checkpoint(self, resume):
        # load model weights
        if resume and os.path.exists('./checkpoints/' + self.architecture + '.pth'):
            model = BertRegressor(self.architecture).to(self.device)
            model.load_state_dict(torch.load('./checkpoints/' + self.architecture + '.pth'))
        else:
            model = BertRegressor(self.architecture).to(self.device)
        return model
        
    