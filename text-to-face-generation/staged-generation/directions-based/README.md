# Face Generation From Text (Directions-based Latent Manipulation)

This folder contains the main code for `staged` experiments of `face generation from text` using `directions-based latent manipulation`.

## Installation

-   Install requirements in main `README.md`.

-   Download [StyleGAN2 model](https://drive.google.com/drive/folders/1yanUI9m4b4PWzR0eurKNq6JR1Bbfbh6L) into `networks` folder.

-   Download [initial seed latent vectors](https://drive.google.com/file/d/1f-TAGkMTcjLh4zR2Q-gdCwiqPpsIXCCM/view?usp=sharing) into `dataset` folder.

-   Refer to `text_processing/scale_bert/README.md` for _BERT_ model installation.

## Usage

-   To run face generation from text :
    -   Add the desired text descriptions in single `.txt` file (one description per line).
    -   Edit `configs/config.json` with required parameters. 
    -   Run `generate_faces.py` :
        ```bash
        python generate_faces.py -cfg /path/to/config/file
        ```
