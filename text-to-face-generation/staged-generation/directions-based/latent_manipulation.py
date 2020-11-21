import numpy as np
import random
import math
import copy

def differentiate(l_text, l_img):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    l_diff = [l1 - l2 if l1 != 0 else 0 for l1, l2 in zip(l_text, l_img)]
    return np.array(l_diff)

def manipulate_latent(seed_latent_vec, seed_logits, text_logits, feature_directions, scale_factor=5):
    # manipulate random latent vector based on predicted features
    # TODO : add reweighting to differentiated logits
    # differentiate predicted logits
    logits_diff = differentiate(text_logits, seed_logits)
    # loop over each feature and navigate the latent space
    final_latent_vec = copy.deepcopy(seed_latent_vec)
    for axis_idx in range(len(logits_diff)):
        # if feature exists in text description
        if logits_diff[axis_idx] != 0:
            # navigate using differentiated logits
            latent_shift = feature_directions[axis_idx] * logits_diff[axis_idx] * scale_factor
        # if feature is unspecified in text description
        else:
            # navigate using random step size
            latent_shift = feature_directions[axis_idx] * random.uniform(-1, 1) * scale_factor
        # apply latent shift of single feature
        final_latent_vec += latent_shift
    return final_latent_vec
