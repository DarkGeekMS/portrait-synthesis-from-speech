import numpy as np
import copy

def differentiate_logits(l_text, l_img):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    l_diff = np.array([l1 - l2 if l1 != -1 else 0 for l1, l2 in zip(l_text, l_img)])
    # return differentiated logits
    return l_diff

def manipulate_latent(seed_latent_vec, seed_logits, text_logits, feature_directions):
    # manipulate random latent vector based on predicted features
    # differentiate and scale predicted logits
    logits_diff = differentiate_logits(text_logits, seed_logits)
    # loop over each feature and navigate the latent space
    final_latent_vec = copy.deepcopy(seed_latent_vec)
    for axis_idx in range(len(logits_diff)):
        # navigate using differentiated logits
        latent_shift = feature_directions[axis_idx] * logits_diff[axis_idx]
        # apply latent shift of single feature
        final_latent_vec += latent_shift
    return final_latent_vec
