# Face Refinement

This folder contains the main code and experiments for `face refinement`. The `face refinement` module is an adaptation of the work done in `face generation` module using _StyleGAN2_ and the generated feature directions.

## Installation

-   Install requirements in main `README.md`.

-   Download [FFHQ StyleGAN2 model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/) into `networks` folder.

## Usage

-   To run face refinement :
    ```bash
    python manipulate.py -pkl /path/to/stylegan2/pkl \
                        -psi truncation_psi \
                        -lv /path/to/latent/vector/file \
                        -ad /path/to/attributes/directions/file \
                        -md /path/to/morph/directions/file \
                        -ac /path/to/attributes/changes/file \
                        -mc /path/to/morph/changes/file
    ```
