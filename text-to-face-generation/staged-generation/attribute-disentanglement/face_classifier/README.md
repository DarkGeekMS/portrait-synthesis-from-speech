# Multi-label Face Classifier

This folder contains the code for `Multi-label Face Classifier` network loading and fine-tuning on face images.

## Installation

-   Install requirements in main `README.md`.

## Network Weights

-   [Custom MobileNetv2](https://github.com/suikei-wong/Facial-Attributes-Classification) : [GDrive](https://drive.google.com/file/d/1yuBfD-tidXt-pIXPY6co0OGcsc1h92Cv/view?usp=sharing).

## Usage

-   For network fine-tuning :
    ```bash
    python train.py -fr /path/to/faces/daatset/root -pkl /path/to/labels/pickle/file
    ```
