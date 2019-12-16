import torch
from torch import nn

from fem.util import init_weights
from fem.vgg import VggBackbone
from fem.depth import DepthToSpace


class GoodPoint(nn.Model):
    def __init__(self, grid_size, n_channels=1, activation=nn.ReLU(),
                 batchnorm=True, dustbin=0):
        super().__init__()
        self.dustbin = dustbin
        self.activation = activation
        self.vgg = VggBackbone(1, batchnorm=batchnorm)
        # Detector head
        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 64 + dustbin, kernel_size=1, stride=1, padding=0)
        # Descriptor
        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        if batchnorm:
            self.batchnormPa = nn.BatchNorm2d(256)
            self.batchnormDa = nn.BatchNorm2d(256)
        else:
            self.batchnormPa = lambda x: x
            self.batchnormDa = lambda x: x
        self.depth2space = DepthToSpace(grid_size)
        init_weights(self)

    def detector_head(self, x):
        x = self.batchnormPa(self.activation(self.convPa(x)))
        x = self.activation(self.convPb(x))
        return x

    def descriptor_head(self, x):
        x = self.batchnormDa(self.activation(self.convDa(x)))
        x = self.activation(self.convDb(x))
        return x

    def semi_forward(self, x):
        assert x.max() > 0.01
        assert x.max() < 1.01
        x = self.vgg(x)
        semi_det = self.detector_head(x)
        desc = self.descriptor_head(x)
        prob = nn.functional.softmax(semi_det, dim=1)
        return prob, desc
