# Multi-label Face Classifier

This folder contains the code for `Multi-label Face Classifier` network loading and fine-tuning on face images.

## Installation

-   Install requirements in main `README.md`.

## Network Weights

| Model | Loss | Optimizer | Trained Weights | Validation Accuracy |
|-------|------|-----------|-----------------|---------------------|
| MobileNetv2 | BCE | ADAM | [GDrive](https://drive.google.com/file/d/1JrExEFcs7vuGhX5oGuF7lPMUx3ajg8g6/view?usp=sharing) | 91.379% |

## Usage

-   For network fine-tuning :
    ```bash
    python train.py -fr /path/to/faces/daatset/root -pkl /path/to/labels/pickle/file
    ```
