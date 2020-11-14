# Multi-label Face Classifier

This folder contains the code for `Multi-label Face Classifier` network loading and fine-tuning on face images.

## Installation

-   Install requirements in main `README.md`.

## Network Weights

| Model | Trained Weights | Validation Accuracy |
|-------|-----------------|---------------------|
| [Custom MobileNetv2](https://github.com/suikei-wong/Facial-Attributes-Classification) | [GDrive](https://drive.google.com/file/d/1yuBfD-tidXt-pIXPY6co0OGcsc1h92Cv/view?usp=sharing) | 90.918% |
| [PyTorch Pretrained ResNet50](https://pytorch.org/hub/pytorch_vision_resnet/) | [GDrive](https://drive.google.com/file/d/1v3BniQgB6xWGHs1BmxAb8Ihh-Fh_re69/view?usp=sharing) | 91.076% |

## Usage

-   For network fine-tuning :
    ```bash
    python train.py -fr /path/to/faces/daatset/root -pkl /path/to/labels/pickle/file
    ```
