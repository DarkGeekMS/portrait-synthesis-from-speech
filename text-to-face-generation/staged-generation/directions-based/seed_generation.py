import os
import pickle
import argparse
import numpy as np
from numba import cuda

from stylegan2_generator import StyleGAN2Generator

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
            logits.append(float('-inf'))
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
    # calculate sigmoid of all feature components to get logits
    logits = np.array(logits)
    logits =  1.0 / (1.0 + np.exp(-logits))
    return w[0], logits

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-df', '--directions_npy', type=str, help='npy file containing feature directions', required=True)
    argparser.add_argument('-nf', '--network_pkl', type=str, help='pkl file containing StyleGAN2 network', required=True)
    argparser.add_argument('-psi', '--truncation_psi', type=float, help='StyleGAN2 latent space truncation factor', default=0.5)
    argparser.add_argument('-seed', '--initial_seed', type=int, help='initial random seed', default=1000)
    argparser.add_argument('-od', '--output_dir', type=str, help='output directory for seed saving', default='results/')

    args = argparser.parse_args()

    # read feature directions
    feature_directions = np.load(args.directions_npy)

    # initialize StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(args.network_pkl, truncation_psi=args.truncation_psi, use_projector=False)

    # perform seed generation
    w, logits = generate_seed(feature_directions, stylegan2_generator, args.initial_seed)

    # save generated seed
    seed_dict = {'W': w, 'logits': logits}
    with open(os.path.join(args.output_dir, 'seed_dict.pkl')) as f:
        pickle.dump(seed_dict, f)
