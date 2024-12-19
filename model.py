import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        
        # Initial Block (C1) - RF: 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.03)
        )
        
        # Second Block (C2) with Depthwise Separable Conv - RF: 7
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(24, 48, kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.03)
        )
        
        # Third Block (C3) with Dilated Conv - RF: 15
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.03),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.03)
        )
        
        # Fourth Block (C4) - RF: 45
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.03)
        )
        
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Final 1x1 conv to get to number of classes
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)      # RF: 3x3
        x = self.conv2(x)      # RF: 7x7
        x = self.conv3(x)      # RF: 15x15
        x = self.conv4(x)      # RF: 45x45
        x = self.gap(x)        
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)