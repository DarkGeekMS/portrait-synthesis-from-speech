from .multilabel_lr import MultiLabelLogisticRegression
import numpy as np
import random
import math
import copy

def l1_normalize(v):
    # normalize vector v using l1 distance
    return np.divide(v, np.sum(np.absolute(v)))

def l2_normalize(v):
    # normalize vector v using l2 distance
    return np.divide(v, np.sqrt(np.dot(v, v)))

def project_vector(v1, v2):
    # project vector v2 onto vector v1
    gs_cofficient = np.dot(v2, v1) / np.dot(v1, v1)
    return v1 * gs_cofficient

def perform_gram_schmidt(feature_axes_matrix):
    # perform Gram-Schmidt process to convert a matrix into orthonormal
    # get matrix transpose to orthogonalize columns
    matrix = feature_axes_matrix.transpose()
    # loop over all columns
    orthogonal_vecs = list()
    for i in range(matrix.shape[0]):
        # pick current column
        temp_vec = matrix[i]
        # loop over previously orthogonalized columns
        for vec in orthogonal_vecs:
            # project previous column onto current one
            proj_vec = project_vector(vec, matrix[i])
            # subtract projection from current vector
            temp_vec = temp_vec - proj_vec
        orthogonal_vecs.append(temp_vec)
    # convert the vectors from orthogonal to orthonormal through normalization
    orthonormal_vecs = list()
    for i in range(len(orthogonal_vecs)):
        orthonormal_vecs.append(l2_normalize(orthogonal_vecs[i]))
    orthonormal_matrix = np.stack(orthonormal_vecs, axis=0).transpose()
    return orthonormal_matrix

def differentiate(l_text, l_img):
    # differentiate between text and image labels, while keeping unspecified labels as zero
    # as mentioned in the original paper
    l_diff = [l1 - l2 if l1 != 0 else 0 for l1, l2 in zip(l_text, l_img)]
    return np.array(l_diff)

def perform_nonlinear_reweight(l_diff):
    # perform non-linear reweighting by rescaling and then applying tan()
    l_diff_rescaled = np.divide(l_diff, 3.0/math.pi)
    return np.tan(l_diff_rescaled)

def get_feature_axes(random_noise, image_logits, threshold=0.5, strategy='ovr', max_iter=10000):
    # get orthonormal feature axes matrix
    # threshold image logits to get predicted labels
    image_labels = copy.deepcopy(image_logits)
    image_labels[image_labels >= threshold] = 1
    image_labels[image_labels < threshold] = 0
    # solve multi-label logistic regression to get the relationship
    ml_classifier = MultiLabelLogisticRegression(strategy=strategy, max_iter=max_iter)
    ml_classifier.fit(random_noise, image_labels)
    # get coefficients of logistic regressor
    classifier_coefficients = ml_classifier.get_coefficients()
    # perform Gram-Schmidt process to convert the relationship matrix into orthonormal
    feature_axes_matrix = perform_gram_schmidt(classifier_coefficients)
    return feature_axes_matrix

def manipulate_latent(random_noise, image_logits, text_logits, feature_axes_matrix):
    # manipulate random latent vector based on predicted features
    # differentiate predicted logits
    logits_diff = differentiate(text_logits, image_logits)
    # perform non-linear reweighting on differentiated logits
    logits_diff = perform_nonlinear_reweight(logits_diff)
    # loop over each feature and navigate the latent space
    final_latent_vec = copy.deepcopy(random_noise)
    for axis_idx in range(len(logits_diff)):
        # if feature exists in text description
        if logits_diff[axis_idx] != 0:
            # navigate using differentiated logits
            latent_shift = feature_axes_matrix[:,axis_idx] * logits_diff[axis_idx]
        # if feature is unspecified in text description
        else:
            # navigate using random step size
            latent_shift = feature_axes_matrix[:,axis_idx] * random.uniform(-math.sqrt(3), math.sqrt(3))
        # apply latent shift of single feature
        final_latent_vec += latent_shift
        # re-normalize the final latent vector using l1 distance
        final_latent_vec = l1_normalize(final_latent_vec)
    return final_latent_vec

def get_target_latent_vector(random_noise, image_logits, text_logits):
    """
    Get target latent vector using latent manipulation and attributes disentanglement
    Parameters
    ----------
    random_noise : ndarray (batch_size X latent_dim)
        Batch of initial random latent noise
    image_logits : ndarray (batch_size X num_features)
        Batch of generated image logits
    text_logits : ndarray (num_features)
        Logits vector of text description
    """
    # get feature axes matrix
    feature_axes_matrix = get_feature_axes(random_noise, image_logits)
    # pick a random element from random noise and image logits to manipulate using text logits
    rand_index = np.random.randint(0, random_noise.shape[0]-1)
    # perform latent manipulation
    final_latent_vec = manipulate_latent(random_noise[rand_index], image_logits[rand_index], text_logits, feature_axes_matrix)
    return final_latent_vec
