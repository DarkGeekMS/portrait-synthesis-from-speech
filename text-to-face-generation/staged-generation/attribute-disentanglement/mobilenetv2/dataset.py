import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

import os
import numpy as np
import pickle
from PIL import Image

import time

class FaceDataset(data.Dataset):
    """
    Faces dataset class.
    Parameters
    ----------
    faces_root : str
        Path to faces dataset root
    pickle_file : str
        Path to labels pickle file
    transforms : <Compose object> (default=None)
        Face image transforms
    """
    def __init__(self, faces_root, pickle_file, transforms=None):
        super(FaceDataset, self).__init__()
        self.faces_root = faces_root
        with open(pickle_file, 'rb') as file:
            self.faces_labels = pickle.load(file)
        self.transforms = transforms

    def __len__(self):
        return len(self.faces_labels)

    def __getitem__(self, idx):
        data_sample = self.faces_labels[idx]
        labels = torch.from_numpy(data_sample[1].astype(int)).float()
        image = Image.open(os.path.join(self.faces_root, data_sample[0]))
        if self.transforms:
            image = self.transforms(image)
        return (image, labels)
