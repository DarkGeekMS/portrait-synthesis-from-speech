# Face Generation From Text (Axes Matrix-based Latent Manipulation)

This folder contains the main code for `staged` experiments of `face generation from text` using [axes matrix-based latent manipulation](https://arxiv.org/abs/2006.07606).

## Installation

-   Install requirements in main `README.md`.

-   Download [FFHQ StyleGAN2 model](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/) into `networks` folder.

-   Refer to `text_processing/scale_bert/README.md`  or `text_processing/legacy_bert/README.md` for _BERT_ model installation.

## Usage

-   Edit `configs/config.json` with required parameters. Note that `operation` parameter can be :
    -   `complete` : run complete pipeline.
    -   `fit` : run feature axes matrix fitting and save the output.
    -   `generate` : generate faces from text descriptions using previously-extracted feature axes matrix.

-   Run `generate_faces.py` :
    ```bash
    python generate_faces.py -cfg /path/to/config/file
    ```
