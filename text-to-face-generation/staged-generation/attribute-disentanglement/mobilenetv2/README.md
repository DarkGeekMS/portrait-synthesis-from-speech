# MobileNet v2

This folder contains the code for `MobileNetv2` network loading and fine-tuning on face images.

## Installation

-   Install requirements in main `README.md`.

-   Download our [pre-trained weights](https://drive.google.com/file/d/1v43am-DXDItB23veBD2-pnv9IxHjooBy/view?usp=sharing) for multi-label classification on `CelebA-HQ` faces.

## Usage

-   For network fine-tuning :
    ```bash
    python train.py -fr /path/to/faces/daatset/root -pkl /path/to/labels/pickle/file
    ```
