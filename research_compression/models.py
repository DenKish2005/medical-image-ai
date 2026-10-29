"""DenseNet-121 CXR classifier divided at feature-block boundaries."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SPLIT_NAMES = ("denseblock1", "denseblock2", "denseblock3")


class DenseNet121Split(nn.Module):
    """One checkpoint feeds every split; dropout/test preprocessing stay fixed."""
    def __init__(self, num_classes: int, split: str = "denseblock2", pretrained: bool = False):
        super().__init__()
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        try:
            from torchvision.models import densenet121, DenseNet121_Weights
        except Exception as exc:
            raise RuntimeError("torchvision required for the optional DenseNet experiment") from exc
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        net = densenet121(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        layers = list(net.features.named_children())
        pivot = next(i for i, (name, _) in enumerate(layers) if name == split)
        self.edge = nn.Sequential(*(layer for _, layer in layers[:pivot+1]))
        self.cloud = nn.Sequential(*(layer for _, layer in layers[pivot+1:]))
        self.classifier = net.classifier
        self.split = split

    def features(self, image):
        return self.edge(image)

    def predict_from_features(self, features):
        x = self.cloud(features)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1,1)).flatten(1)
        return self.classifier(x)

    def forward(self, image):
        return self.predict_from_features(self.features(image))


class DownstreamHead(nn.Module):
    def __init__(self, model: DenseNet121Split):
        super().__init__()
        # Reference (uncompressed) and student predictions can share the same
        # trained baseline downstream weights for a controlled comparison.
        self.cloud = model.cloud
        self.classifier = model.classifier

    def forward(self, feature):
        return self.classifier(F.adaptive_avg_pool2d(F.relu(self.cloud(feature), inplace=False), (1,1)).flatten(1))
