import numpy as np
import random
import math
import copy

def l1_normalize(v):
    # normalize vector v using l1 distance
    return np.divide(v, np.sum(np.absolute(v)))

def differentiate(l_text, l_img):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    # as mentioned in the original paper
    l_diff = [l1 - l2 if l1 != 0 else 0 for l1, l2 in zip(l_text, l_img)]
    return np.array(l_diff)

def perform_nonlinear_reweight(l_diff):
    # perform non-linear reweighting by rescaling and then applying tan()
    l_diff_rescaled = np.divide(l_diff, 3.0/math.pi)
    return np.tan(l_diff_rescaled)

def manipulate_latent(seed_latent_vec, seed_logits, text_logits, feature_directions):
    # manipulate random latent vector based on predicted features
    # differentiate predicted logits
    logits_diff = differentiate(text_logits, seed_logits)
    # perform non-linear reweighting on differentiated logits
    logits_diff = perform_nonlinear_reweight(logits_diff)
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
            latent_shift = feature_directions[axis_idx] * random.uniform(-math.sqrt(3), math.sqrt(3))
        # apply latent shift of single feature
        final_latent_vec += latent_shift
        # re-normalize the final latent vector using l1 distance
        final_latent_vec = l1_normalize(final_latent_vec)
    return final_latent_vec
