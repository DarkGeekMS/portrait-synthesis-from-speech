# Face Alignment

This folder contains a utility for face alignment using **face_recognition** module to be projected and used in better fitting axes of **StyleGAN2**.

## Installation
For `python3` users 
```bash
sudo apt install cmake
pip3 install face_recognition
```
For `python2` users, use `pip` instead of `pip3`

## Usage

Generate aligned faces:
```bash
python face_alignment.py -root path/to/root/directory/containing/folders/each/contains/images/to/be/aligned
```

