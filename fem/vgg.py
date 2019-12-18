from torch import nn


def block(in_channels, out_channels,
          kernel_size=3, stride=1,
          padding=1,
          batchnorm=True, activation=None):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding)]
    if activation is not None:
        layers.append(activation)
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class VggBackbone(nn.Module):
    def __init__(self, n_input_channels, activation=nn.ReLU(),
                 batchnorm=True):
        pool = nn.MaxPool2d((2, 2))
        self.net = nn.Sequential(block(n_input_channels, 64, activation=activation),
                                 block(64, 64, activation=activation, batchnorm=batchnorm),
                                 pool,
                                 block(64, 64, activation=activation, batchnorm=batchnorm),
                                 block(64, 64, activation=activation, batchnorm=batchnorm),
                                 pool,
                                 block(64, 128, activation=activation, batchnorm=batchnorm),
                                 block(128, 128, activation=activation, batchnorm=batchnorm),
                                 pool,
                                 block(128, 128, activation=activation, batchnorm=batchnorm),
                                 block(128, 128, activation=activation, batchnorm=batchnorm))

    def forward(self, x):
        return self.net(x)
