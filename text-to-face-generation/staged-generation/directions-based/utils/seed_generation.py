import numpy as np

def generate_seed(feature_directions, stylegan2_generator, seed):
    # generate a random seed of extended latent vector and corresponding logits
    # initialize random state
    rand_state = np.random.RandomState(seed)
    # generate random unextended latent vector
    z = rand_state.randn(1, 512)
    # map unextended to extended latent space
    w = stylegan2_generator.map_latent_vector(z)
    # project latent vector onto all feature directions
    logits = []
    # loop over all feature directions
    for direction in feature_directions:
        # check whether direction doesn't exist (all zeros)
        if np.count_nonzero(direction) == 0:
            logits.append(0)
            continue
        components = []
        # project each layer independently
        for idx in range(direction.shape[0]):
            # get unit vector of direction for each layer (512)
            unit_direction = np.divide(direction[idx], np.sqrt(np.dot(direction[idx], direction[idx])))
            # project latent vector onto unit direction vector
            components.append(np.dot(w[0][idx], unit_direction))
        # average projected components of all layers
        avg_component = sum(components) / len(components)
        logits.append(avg_component)
    # return random latent vector and its logits
    return w[0], np.array(logits)
