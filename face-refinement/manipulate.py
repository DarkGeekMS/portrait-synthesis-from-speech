from stylegan2_networks import SynthesisNetwork

import numpy as np
import skimage.io as io
import argparse

def refine_faces(
    network_pt, truncation_psi, latent_vector, attributes_directions,
    morph_directions, attributes_changes, morph_changes
):
    # initialize CUDA device
    device = torch.device('cuda')

    # initialize StyleGAN2 generator
    stylegan2_generator = SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
    stylegan2_generator.load_state_dict(torch.load(network_pt))
    stylegan2_generator.to(device)

    # attributes edits
    for idx in range(attributes_changes):
        if attributes_changes[idx]:
            latent_vector += attributes_changes[idx] * attributes_directions[idx]

    # morphological edits
    for idx in range(morph_changes):
        if morph_changes[idx]:
            latent_vector += morph_changes[idx] * morph_directions[idx]

    # generate new face
    w_tensor = torch.tensor(np.expand_dims(latent_vector, axis=0), device=device)
    face_image = stylegan2_generator(w_tensor, noise_mode='const')
    face_image = face_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
    face_image[face_image < -1.0] = -1.0
    face_image[face_image > 1.0] = 1.0
    face_image = (face_image + 1.0) * 127.5
    face_image = face_image.astype(np.uint8)

    # save refined face
    io.imsave(f'refined_face.png', face_image)

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-pt', '--network_pt', type=str, help='path to stylegan2 network .pt file', default='networks/stgan2_model.pt'
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
    refine_faces(args.network_pt, args.truncation_psi, args.latent_vector, args.attributes_directions,
                args.morph_directions, args.attributes_changes, args.morph_changes)
