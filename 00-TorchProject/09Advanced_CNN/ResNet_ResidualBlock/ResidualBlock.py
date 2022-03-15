import torch
import torch.nn as nn


class ResidualBlock1(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.activate = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.activate(y)
        y = self.conv2(y)
        res = self.activate(x + y)
        return res


class ResidualBlock2(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.activate = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.activate(y)
        y = self.conv2(y)
        res = self.activate((x + y) * 0.5)
        return res


class ResidualBlock3(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(channels,channels,kernel_size=1,stride=1)
        self.activate = nn.ReLU()

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.activate(y)
        y1 = self.conv2(y1)
        y2 = self.conv3(x)
        res = self.activate(y1 + y2)
        return res
