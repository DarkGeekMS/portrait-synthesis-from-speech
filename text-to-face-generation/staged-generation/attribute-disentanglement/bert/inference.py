import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import pandas as pd
from importlib import import_module
import os

from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import logger
from pybert.configs.basic_config import config
from pybert.model.bert_for_multi_label import BertForMultiLable
from pybert.io.task_data import TaskData
from pybert.test.predictor import Predictor


def bertMultiLabelClassifier(description, num_labels, path_to_bert_classifier):

    config['checkpoint_dir'] = path_to_bert_classifier / config['checkpoint_dir'] / 'bert'

    target = [0]*num_labels
    data = TaskData()

    lines = list(zip([description], [target]))
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=True)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file='')
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=256,
                                              cached_features_file='')
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,
                                 collate_fn=collate_fn)

    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu='0')
    results = predictor.predict(data=test_dataloader)
    out = [int(i > 0.5) for i in results[0]]
    df = pd.DataFrame(list(zip(label_list, out)), 
                    columns =['attibutes', 'value'])
    return df


# description = "a man with light beard and long and smooth hair. He is fat. His eyes is narrow. His nose is tiny. he has mustache."
# print(bertMultiLabelClassifier(description, 32, './'))