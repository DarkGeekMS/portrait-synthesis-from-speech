import torch
from torch import nn
from torchvision.models import resnet50

class FaceClassifier(nn.Module):
    """
    Multi-label Face Classifier class.
    Parameters
    ----------
    n_classes : int (default=32)
        Number of classes
    pretrained : bool (default=True)
        Whether to use ImageNet pretrained weights or not
    """
    def __init__(self, n_classes=32, pretrained=True):
        super(FaceClassifier, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        # add custom classifier
        self.model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        out = self.sigmoid(x)
        return out
