import numpy as np

def postprocess_text_logits(sent_pred, axes_range):
    # perform post-processing on text output logits
    # in order to align indices and scale logits
    # initialize processed logits array with -1
    proc_pred = np.array([-1.0]*32)
    # [18] face thickness attributes
    proc_pred[13] = sent_pred[18]
    # [19] gender attributes
    proc_pred[14] = sent_pred[19]
    # [20] age attributes
    proc_pred[15] = sent_pred[20]
    # re-scale all attributes based on considered axes range
    proc_pred_scaled = np.array(
        [(logit*axes_range*2.0)-axes_range if logit != -1.0 else -1.0 for logit in proc_pred]
    )
    # return scaled processed text logits
    return proc_pred_scaled
