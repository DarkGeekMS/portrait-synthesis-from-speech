import numpy as np
import random
import math
import copy

def sigmoid(x):
    # compute sigmoid function of a given quantity
    return 1 / (1 + np.exp(-1*x))

def scale_logits(l_text, l_img, l_diff, scale_factor):
    # scale differentiated logits based on sigmoid function
    # threshold each logit between sigmoid(-2) and sigmoid(2)
    # NOTE: threshold is chosen between -2 and 2 because the differentiation effect vanishes outside this region
    # NOTE: this is mainly because sigmoid is almost flat outside this region
    l_text_low = l_text < sigmoid(-2)
    l_text_high = l_text > sigmoid(2)
    l_img_low = l_img < sigmoid(-2)
    l_img_high = l_img > sigmoid(2)
    # select positions where one logit is above sigmoid(2) and the other is below sigmoid(-2)
    l_scale = np.logical_or(np.logical_and(l_text_low, l_img_high), np.logical_and(l_text_high, l_img_low))
    l_scale = l_scale.astype(int)
    # apply scale factor where factor of logits outside the region is double that of inside the region
    l_scale[l_scale == 0] = scale_factor
    l_scale[l_scale == 1] = scale_factor * 2
    # scale differentiated logits 
    l_diff_scaled = np.multiply(l_diff, l_scale)
    # return scaled differentiated logits
    return l_diff_scaled

def differentiate_logits(l_text, l_img, scale_factor):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    l_diff = np.array([l1 - l2 if l1 != -1 else 0 for l1, l2 in zip(l_text, l_img)])
    # scale differentiated logits
    l_diff_scaled = scale_logits(l_text, l_img, l_diff, scale_factor)
    # return scaled differentiated logits
    return l_diff_scaled

def manipulate_latent(seed_latent_vec, seed_logits, text_logits, feature_directions, scale_factor=10):
    # manipulate random latent vector based on predicted features
    # differentiate and scale predicted logits
    logits_diff = differentiate_logits(text_logits, seed_logits, scale_factor)
    # loop over each feature and navigate the latent space
    final_latent_vec = copy.deepcopy(seed_latent_vec)
    for axis_idx in range(len(logits_diff)):
        # if feature exists in text description
        if logits_diff[axis_idx] != 0:
            # navigate using differentiated logits
            latent_shift = feature_directions[axis_idx] * logits_diff[axis_idx]
        # if feature is unspecified in text description
        else:
            # navigate using random step size
            latent_shift = feature_directions[axis_idx] * random.uniform(-1*scale_factor, scale_factor)
        # apply latent shift of single feature
        final_latent_vec += latent_shift
    return final_latent_vec
