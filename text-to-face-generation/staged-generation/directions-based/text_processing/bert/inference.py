import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import pandas as pd
from importlib import import_module
import os

from .pybert.io.utils import collate_fn
from .pybert.io.bert_processor import BertProcessor
from .pybert.common.tools import logger
from .pybert.configs.basic_config import config
from .pybert.model.bert_for_multi_label import BertForMultiLable
from .pybert.io.task_data import TaskData
from .pybert.test.predictor import Predictor

class BERTMultiLabelClassifier():
    def __init__(self):
        self.checkpoint_dir = config['checkpoint_dir'] / 'bert'
        self.processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=True)
        self.label_list = self.processor.get_labels()
        self.model = BertForMultiLable.from_pretrained(self.checkpoint_dir, num_labels=len(self.label_list))
        self.predictor = Predictor(model=self.model,
                            logger=logger,
                            n_gpu='0')
        self.target = [0]*len(self.label_list)

    def predict(self, description):
        lines = list(zip([description], [self.target]))
        
        id2label = {i: label for i, label in enumerate(self.label_list)}

        test_data = self.processor.get_test(lines=lines)
        test_examples = self.processor.create_examples(lines=test_data,
                                                example_type='test',
                                                cached_examples_file='')
        test_features = self.processor.create_features(examples=test_examples,
                                                max_seq_len=256,
                                                cached_features_file='')
        test_dataset = self.processor.create_dataset(test_features)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,
                                    collate_fn=collate_fn)

        
        results = self.predictor.predict(data=test_dataloader)[0]
        half_length = len(results) // 2
        out = []
        for i in range(half_length):
            if results[i] > 0.5:
                # exists
                out.append(results[i+half_length])
                out.append(results[i+half_length])
            else:
                #doesn't exist
                out.append(0)

        return out


# description = "a man with light beard and long and smooth hair. He is fat. His eyes is narrow. His nose is tiny. he doesn't have mustache."
# bert = bertMultiLabelClassifier()
# print(bert.predict(description))
