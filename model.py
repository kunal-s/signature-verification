import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseResNet(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, embedding_dim)  # embedding size 128

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # normalize embeddings
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
