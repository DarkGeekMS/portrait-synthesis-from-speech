from transformers import DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer, BertTokenizer
import torch
class CheckpointTracker():
    def __init__(self, device, architecture):
        self.device = device
        self.architecture = architecture
        # tokenizer
        if architecture == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if architecture == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if architecture == 'albert-base-v2':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        if architecture == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
        
        # dummy input to save traced cell (model graph + weights)
        sentence = 'dummy sentence'
        encodings = self.tokenizer([sentence], truncation=True, padding=True)
        self.dummy_input_ids = torch.tensor(encodings['input_ids']).to(self.device)
        self.dummy_attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), './checkpoints/' + self.architecture + '.pth')