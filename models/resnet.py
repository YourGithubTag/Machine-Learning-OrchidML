'''

DISCLAIMER: ALL CODE IS WRITTEN FROM SCRATCH AND MAY CONTAIN SIMILARITIES TO CODE FOUND ONLINE.
            APPROPRIATE ACKNOWLEDGEMENTS HAVE BEEN MADE WHERE NECESSARY.

The ResNet-18 implementation below is identical to the published paper.
All layers and parameters used have been referenced to their appropriate sections of the report.
Note: Identical layers have not been commented due to similarities with other comments.

Link: https://arxiv.org/pdf/1512.03385.pdf

'''

import torch
import torch.nn as nn

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        # Section 4 - Experiments, Table 1 (Convolutional Layers, Max Pooling)
        self.features = nn.Sequential(
            # Conv. Layer 1
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # Section 3.4 - Implementation (Batch Normalisation, RELU)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),

            # Conv. Layer 2
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),

            # Conv. Layer 3
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            
            # Conv. Layer 4
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),

            # Conv. Layer 5
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512)
        )

        # Section 3.3 - Network Architectures (Average Pooling)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Section 3.3 - Network Architectures (FC Layer)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=17, bias=True)
        )
    
    # Forward() links all the layers together.
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x