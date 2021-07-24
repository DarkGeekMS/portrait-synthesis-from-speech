from Bert.inference import TextProcessor
from Bert.model import BertRegressor
from transformers import DistilBertTokenizer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

table = pd.read_csv('./test.csv')
descriptions = table['Description']

architecture = 'distilbert-base-uncased'
checkpoint_path = 'Bert/checkpoints/' + architecture + '.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BertRegressor(architecture, True).to(device)
model.load_state_dict(torch.load(checkpoint_path, device))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

y_hat = []
y_true = []
for index, row  in table.iterrows():
    sentence = row['Description']
    true_logits = list(row.values.tolist()[:-1])
    # print(sentence)
    # print(y_true)
    encodings = tokenizer([sentence], truncation=True, padding=True)
    input_ids = torch.tensor(encodings['input_ids']).to(device)
    attention_mask = torch.tensor(encodings['attention_mask']).to(device)
    logits = list(np.round(model(input_ids, attention_mask=attention_mask, train=False).cpu().data.numpy()[0]))
    y_hat = y_hat + logits
    y_true = y_true + true_logits

results = precision_recall_fscore_support(y_true, y_hat, average='micro')

