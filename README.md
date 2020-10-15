# Portrait Synthesis From Speech

This repository contains the main code for `portrait synthesis from speech` project and its experiments. The project mainly consists of :
-   Speech Recognition.
-   Face generation from Text.
-   Face Refinement.
-   Multiple Head Poses Generation.

## Installation

-   Install `python3` and `python3-pip`.

-   Install `requirements.txt` using `PyPi` :
    ```bash
    pip3 install -r requirements.txt
    ```

-   Alternatively, `Dockerfile` is provided with all required dependencies :
    ```bash
    docker build . -t portrait_syn_from_speech
    ```

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
    python train.py -dsp /path/to/dataset/root/dir -w2v /path/to/word2vec/file -mv model_version(1|2) -pkl /path/to/stylegan2/model/file -psi truncation_psi -rd /path/to/results/dir
    ```
