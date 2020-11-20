# Face Generation From Text (Directions-based Latent Manipulation)

This folder contains the main code for `staged` experiments of `face generation from text` using `directions-based latent manipulation`.

## Installation

-   Install requirements in main `README.md`.

## Usage

-   To run initial seed generation :
    ```bash
    python seed_generation.py -df /path/to/directions/npy/file \
                            -nf /path/to/stylegan2/network/file \
                            -psi trunction_psi \
                            -seed initial_seed \
                            -od /path/to/output/directory
    ```

-   To run face generation from text :
    -   Add the desired text descriptions in single `.txt` file (one description per line).
    -   Edit `configs/config.json` with required parameters. 
    -   Run `generate_faces.py` :
        ```bash
        python generate_faces.py -cfg /path/to/config/file
        ```
