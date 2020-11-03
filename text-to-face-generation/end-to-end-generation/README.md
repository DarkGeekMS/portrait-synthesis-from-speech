# Face Generation From Text (End-to-End)

This folder contains the main code for `end-to-end` experiments of `face generation from text`.

## Installation

-   Install requirements in main `README.md`.

-   Run the following `python` snippet once, in order to have the `NLTK` tokenizer :
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

-   Refer to `dataset-prep` folder for utilities of dataset preparation.

-   Create a new folder in current directory and place your dataset in three sub-folders : `face-images`, `text-desc` and `latent-vectors`.

-   Edit network, train and test configuration in `configs/`.

-   For network training :
    ```bash
    python train.py -ncfg /path/to/network/config/json -tcfg /path/to/train/config/json
    ```

-   For network inference :
    ```bash
    python test.py -ncfg /path/to/network/config/json -tcfg /path/to/test/config/json
    ```
