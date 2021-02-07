"""
Disentanglement score : angles between generated feature axes
"""

import numpy as np
import math

def get_unit_vector(vector):
    # get unit vector of a given vector
    return np.divide(vector, np.sqrt(np.dot(vector, vector)))

def get_included_angle(first_vector, second_vector):
    # get included angle between two one-dimensional vector
    return math.degrees(
        math.acos(
            np.dot(get_unit_vector(first_vector), get_unit_vector(second_vector))
        )
    )

def disentanglement_score(source_direction, target_direction):
    # compute disentanglement score between two feature directions
    # initialize angles list
    included_angles = list()
    # loop over each layer in directions
    for layer in range(source_direction.shape[0]):
        # append included angle between two layer vectors
        included_angles.append(
            get_included_angle(source_direction[layer], target_direction[layer])
        )
    # return average of included angles of all layers
    return sum(included_angles) / len(included_angles)
