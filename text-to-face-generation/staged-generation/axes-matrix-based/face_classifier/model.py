import torch
from torch import nn
from torchvision import models

class FaceClassifier(nn.Module):
    """
    Multi-label Face Classifier class.
    Parameters
    ----------
    n_classes : int (default=32)
        Number of classes
    backbone : str (default='resnet50')
        Feature extractor to be used
    pretrained : bool (default=True)
        Whether to use ImageNet pretrained weights or not
    """
    def __init__(self, n_classes=32, backbone='resnet50', pretrained=True):
        super(FaceClassifier, self).__init__()
        if backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        elif backbone == 'mobilenetv2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True)
        elif backbone == 'vgg16':
            self.model = models.vgg16_bn(pretrained=pretrained)
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=n_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        out = self.sigmoid(x)
        return out
