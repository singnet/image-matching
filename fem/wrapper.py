import numpy
import torch
from torch import nn
from fem import nonmaximum, util
from fem.depth import DepthToSpace
from fem.goodpoint import GoodPoint


class SuperPoint(GoodPoint):
    def __init__(self, nms):
        GoodPoint.__init__(self, 8, n_channels=1, activation=nn.ReLU(),
                 batchnorm=False, dustbin=1, nms=nms)

    def detector_head(self, x):
        x = self.activation(self.convPa(x))
        x = self.convPb(x)
        return x
