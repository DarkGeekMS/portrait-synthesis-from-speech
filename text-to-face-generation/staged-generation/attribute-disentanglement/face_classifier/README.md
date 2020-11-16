# Multi-label Face Classifier

This folder contains the code for `Multi-label Face Classifier` network loading and fine-tuning on face images.

## Installation

-   Install requirements in main `README.md`.

## Network Weights

| Model | Loss | Optimizer | Trained Weights | Validation Accuracy |
|-------|------|-----------|-----------------|---------------------|
| MobileNetv2 | BCE | ADAM | [GDrive](https://drive.google.com/file/d/1JrExEFcs7vuGhX5oGuF7lPMUx3ajg8g6/view?usp=sharing) | 91.379% |
| MobileNetv2 | Focal Loss | ADAM | [GDrive](https://drive.google.com/file/d/1qRQLhzHC8iGgjh9N7UOGyARW_VA6ggMU/view?usp=sharing) | 91.314% |
| MobileNetv2 | BCE | SGD | [GDrive](https://drive.google.com/file/d/12GDKmiu2IhMW7X4LTjAPkwNp36cq_1Zn/view?usp=sharing) | 91.438% |
| MobileNetv2 | Focal Loss | SGD | [GDrive](https://drive.google.com/file/d/1FsOBt6kOzmvqyQSqPDHrLAKHDWM8oA4r/view?usp=sharing) | 91.248% |

## Usage

-   For network fine-tuning :
    ```bash
    python train.py -fr /path/to/faces/daatset/root -pkl /path/to/labels/pickle/file
    ```
