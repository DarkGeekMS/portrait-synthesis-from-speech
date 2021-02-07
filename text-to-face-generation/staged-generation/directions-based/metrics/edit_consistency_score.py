"""
Edit consistency score : projected components of generated face code along feature axes
"""

import numpy as np

def get_unit_vector(vector):
    # get unit vector of a given vector
    return np.divide(vector, np.sqrt(np.dot(vector, vector)))

def edit_consistency_score(latent_vector, feature_directions, target_features=None):
    # compute edit consistency score of latent vector with respect to features
    # check if no target features
    if not target_features:
        # compute score with respect to all features
        target_features = list(range(feature_directions.shape[0]))
    # intialize scores dictionary
    scores_dict = dict()
    # loop over each target feature index
    for feature_idx in target_features:
        # initialize layers components list
        components = list()
        # loop over each layer in feature direction
        for layer_idx in range(feature_directions[feature_idx].shape[0]):
            # calculate unit vector of feature direction layer
            unit_direction = get_unit_vector(feature_directions[feature_idx][layer_idx])
            # project latent vector layer of unit direction layer
            components.append(np.dot(latent_vector[layer_idx], unit_direction))
        # get components average
        scores_dict[feature_idx] = sum(components) / len(components)
    # return scores dictionary
    return scores_dict
