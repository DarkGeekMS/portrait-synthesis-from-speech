#encoding:utf-8
import torch
import numpy as np
import pickle
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,model,logger,n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.attributes_max_values = np.load('text_processing/scale_bert/attributes_max.npy')

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data),desc='Testing')
        all_logits = None
        for step, batch in enumerate(data):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask).detach().cpu().numpy()
                logits_mod = np.copy(logits)
                logits_mod[logits < 1] = 1
                logits_mod[logits < 0.5] = 0
                logits_mod -= 1
                logits_mod = logits_mod / (self.attributes_max_values-1)
                # male-female -> no option for being not mentioned
                logits_mod[:,20] += 1

                logits = logits_mod
                logits_mod[logits_mod < 0] = -1  
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = np.concatenate([all_logits,logits],axis = 0)
            pbar(step=step)
        print()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits






