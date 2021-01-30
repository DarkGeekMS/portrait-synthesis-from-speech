import numpy as np

def postprocess_text_logits(sent_pred, axes_range):
    # perform post-processing on text output logits
    # in order to align indices and scale logits
    # initialize processed logits array with -1
    proc_pred = np.array([-1.0]*32)
    # [0] bushy eyebrows attribute
    proc_pred[0] = sent_pred[1]
    # [1,2,3,4] hair color attributes
    if sent_pred[2] != -1:
        # black hair
        # NOTE : navigate in negative brown direction
        proc_pred[3] = 0
    else:
        # set hair color with output logits
        proc_pred[1] = sent_pred[3]
        proc_pred[2] = sent_pred[4]
        proc_pred[3] = sent_pred[5]
        proc_pred[4] = sent_pred[6]
    # [5] straight/curly hair attribute
    proc_pred[5] = sent_pred[7]
    # [6] receding hairline attribute
    proc_pred[6] = sent_pred[8]
    # [7] bald attribute
    proc_pred[7] = sent_pred[9]
    # [8] hair with bangs attribute
    proc_pred[8] = sent_pred[10]
    # [9] hair color attribute
    proc_pred[9] = sent_pred[11]
    # [10] beard attribute
    proc_pred[10] = sent_pred[14]
    # [11] skin color attribute
    proc_pred[11] = sent_pred[16]
    # [13] face thickness attribute
    proc_pred[13] = sent_pred[18]
    # [14] gender attribute
    proc_pred[14] = sent_pred[19]
    # [15] age attribute
    proc_pred[15] = sent_pred[20]
    # re-scale all attributes based on considered axes range
    proc_pred_scaled = np.array(
        [(logit*axes_range*2.0)-axes_range if logit != -1.0 else -1.0 for logit in proc_pred]
    )
    # return scaled processed text logits
    return proc_pred_scaled
