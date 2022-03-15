import torch
import torch.nn as nn


class InceptionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionModel, self).__init__()
        # ----------------#
        self.polling = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # ----------------#
        self.branch1x1_1 = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch1x1_2 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # ----------------#
        self.branch5x5 = nn.Conv2d(16, 24, kernel_size=5, padding=2, stride=1)
        # ----------------#
        self.branch3x3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1, stride=1)
        self.branch3x3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1)
        # ----------------#

    def forward(self, x):
        forward_net1 = self.branch1x1_1(self.polling(x))
        forward_net2 = self.branch1x1_2(x)
        forward_net3 = self.branch5x5(self.branch1x1_2(x))
        forward_net4 = self.branch3x3_2(self.branch3x3_1(self.branch1x1_2(x)))
        out = [forward_net1, forward_net2, forward_net3, forward_net4]
        return torch.cat(out, dim=1)
