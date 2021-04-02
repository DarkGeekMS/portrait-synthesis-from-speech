from stylegan2_generator import StyleGAN2Generator

import numpy as np
import skimage.io as io
import argparse

def refine_faces(
    network_pkl, truncation_psi, latent_vector, attributes_directions,
    morph_directions, attributes_changes, morph_changes
):
    # initialize StyleGAN2 generator
    stylegan2_generator = StyleGAN2Generator(
        network_pkl, truncation_psi=truncation_psi, use_projector=False
    )

    # attributes edits
    for idx in range(attributes_changes):
        if attributes_changes[idx]:
            latent_vector += attributes_changes[idx] * attributes_directions[idx]

    # morphological edits
    for idx in range(morph_changes):
        if morph_changes[idx]:
            latent_vector += morph_changes[idx] * morph_directions[idx]

    # generate new face
    face_image = stylegan2_generator.generate_images(latent_vector)[0]

    # save refined face
    io.imsave(f'refined_face.png', face_image)

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-pkl', '--network_pkl', type=str, help='path to stylegan2 network pickle', default='networks/ffhq.pkl'
    )
    argparser.add_argument(
        '-psi', '--trunc_psi', type=int, help='truncation psi', default=0.5
    )
    argparser.add_argument(
        '-lv', '--latent_vector', type=str, help='path to latent vector file', required=True
    )
    argparser.add_argument(
        '-ad', '--attributes_directions', type=str, help='path to attributes directions file',
        default='stylegan2_directions/attributes_directions.npy'
    )
    argparser.add_argument(
        '-md', '--morph_directions', type=str, help='path to morphological directions file',
        default='stylegan2_directions/morph_directions.npy'
    )
    argparser.add_argument(
        '-ac', '--attributes_changes', type=str, help='path to attributes changes file', required=True
    )
    argparser.add_argument(
        '-mc', '--morph_changes', type=str, help='path to morphological changes file', required=True
    )

    args = argparser.parse_args()

    # run face refinement function
    refine_faces(args.network_pkl, args.truncation_psi, args.latent_vector, args.attributes_directions,
                args.morph_directions, args.attributes_changes, args.morph_changes)
