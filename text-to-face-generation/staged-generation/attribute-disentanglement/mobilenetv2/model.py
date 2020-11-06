import torch
from torch import nn
from torchvision.models import mobilenet_v2

class MobileNet(nn.Module):
    """
    MobileNetv2 class.
    Parameters
    ----------
    n_classes : int (default=32)
        Number of classes
    pretrained : bool (default=True)
        Whether to use ImageNet pretrained weights or not
    """
    def __init__(self, n_classes=32, pretrained=True):
        super(MobileNet, self).__init__()
        model = mobilenet_v2(pretrained=pretrained)
        self.feature_extractor = model.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=model.last_channel, out_features=n_classes)
        )
        self.sigmoid = nn.Sigmoid()
        self.is_train = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        out = self.classifier(x)
        if not self.is_train:
            out = self.sigmoid(out)
        return out

    def train(self):
        self.feature_extractor.train()
        self.pool.train()
        self.classifier.train()
        self.is_train = True

    def eval(self):
        self.feature_extractor.eval()
        self.pool.eval()
        self.classifier.eval()
        self.sigmoid.eval()
        self.is_train = False
