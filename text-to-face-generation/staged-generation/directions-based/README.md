# Face Generation From Text (Directions-based Latent Manipulation)

This folder contains the main code for `staged` experiments of `face generation from text` using `directions-based latent manipulation`.

## Installation

-   Install requirements in main `README.md`.

-   Download [FFHQ StyleGAN2 synthesis model](https://drive.google.com/file/d/1owlrJmUc_najBuT14Rq0zsfu0N1bfnQW/view?usp=sharing) into `networks` folder.

-   Download [initial seed latent vectors](https://drive.google.com/file/d/1f-TAGkMTcjLh4zR2Q-gdCwiqPpsIXCCM/view?usp=sharing) into `dataset` folder.

-   Refer to `text_processing/README.md` for _BERT_ model installation.

## Usage

-   To run face generation from text :
    -   Add the desired text descriptions in single `.txt` file (one description per line).
    -   Edit `configs/generation_config.json` with required parameters. 
    -   Run `generate_faces.py` :
        ```bash
        python generate_faces.py -cfg /path/to/config/file
        ```
