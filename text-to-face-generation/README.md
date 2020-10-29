# Face Generation From Text

This folder contains the main code and experiments for `face generation from text`.

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

-   For network training :
    ```bash
    python train.py -dsp /path/to/dataset/root/dir -w2v /path/to/word2vec/file -mv model_version(1|2) -mp /path/to/initial/model/weights -pkl /path/to/stylegan2/model/file -psi truncation_psi -rd /path/to/results/dir
    ```

-   For network inference :
    ```bash
    python test.py -tp /path/to/sentences/text -w2v /path/to/word2vec/file -mv model_version(1|2) -mp /path/to/model/weights -pkl /path/to/stylegan2/model/file -psi truncation_psi -rd /path/to/results/dir
    ```
