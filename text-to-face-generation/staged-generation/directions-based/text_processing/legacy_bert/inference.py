import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import pandas as pd
import numpy as np
from importlib import import_module
import os
import json

from .pybert.io.utils import collate_fn
from .pybert.io.bert_processor import BertProcessor
from .pybert.common.tools import logger
from .pybert.configs.inference_config import config
from .pybert.model.bert_for_multi_label import BertForMultiLable
from .pybert.io.task_data import TaskData
from .pybert.test.predictor import Predictor

class BERTMultiLabelClassifier():
    def __init__(self):
        self.checkpoint_dir = config['checkpoint_dir'] / 'bert'
        self.processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=True, num_labels = 32)

        with open(self.checkpoint_dir / 'config.json') as config_file:
            configs = json.load(config_file)
            self.num_labels = len(configs['id2label'])

        self.model = BertForMultiLable.from_pretrained(self.checkpoint_dir, num_labels=self.num_labels)
        self.predictor = Predictor(model=self.model,
                            logger=logger,
                            n_gpu='0')
        self.target = [0]*self.num_labels

    def predict(self, description):
        lines = list(zip([description], [self.target]))
        
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
                if i != 31:
                    out.append(results[i+half_length])
                else: # young attribute is flipped to match age direction
                    out.append(1 - results[i+half_length])
            else:
                #doesn't exist
                out.append(-1)
        return np.array(out)
