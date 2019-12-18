import torch
from torch import nn

from fem.util import init_weights
from fem.depth import DepthToSpace


class GoodPoint(nn.Module):
    def __init__(self, grid_size, n_channels=1, activation=nn.ReLU(),
                 batchnorm=True, dustbin=0):
        super().__init__()
        self.dustbin = dustbin
        self.activation = activation
        stride = 1
        self.conv1a = nn.Conv2d(n_channels, 64, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv2b = nn.Conv2d(64, 64, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv3b = nn.Conv2d(128, 128, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=grid_size,
                        stride=stride, padding=1)

        self.conv4b = nn.Conv2d(128, 128, kernel_size=grid_size,
                        stride=stride, padding=1)
        self.pool = nn.MaxPool2d((2, 2))

        # Detector head
        self.convPa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(256, 64 + dustbin, kernel_size=1, stride=1, padding=0)
        # Descriptor
        self.convDa = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        if batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.batchnorm2 = nn.BatchNorm2d(64)
            self.batchnorm3 = nn.BatchNorm2d(64)
            self.batchnorm4 = nn.BatchNorm2d(64)
            self.batchnorm5 = nn.BatchNorm2d(128)
            self.batchnorm6 = nn.BatchNorm2d(128)
            self.batchnorm7 = nn.BatchNorm2d(128)
            self.batchnorm8 = nn.BatchNorm2d(128)
            self.batchnormPa = nn.BatchNorm2d(256)
            self.batchnormDa = nn.BatchNorm2d(256)
        else:
            l = lambda x: x
            self.batchnorm1 = l
            self.batchnorm2 = l
            self.batchnorm3 = l
            self.batchnorm4 = l
            self.batchnorm5 = l
            self.batchnorm6 = l
            self.batchnorm7 = l
            self.batchnorm8 = l
            self.batchnormPa = l
            self.batchnormDa = l
            self.batchnormPa = l
            self.batchnormDa = l
        self.depth_to_space = DepthToSpace(grid_size)
        init_weights(self)

    def detector_head(self, x):
        x = self.batchnormPa(self.activation(self.convPa(x)))
        x = self.activation(self.convPb(x))
        return x

    def descriptor_head(self, x):
        x = self.batchnormDa(self.activation(self.convDa(x)))
        x = self.activation(self.convDb(x))
        return x

    def superblock(self, x, conv1, conv2, batch1, batch2):
        x = batch2(self.activation(conv2(batch1(self.activation(conv1(x))))))
        return x

    def vgg(self, x):
        x = self.superblock(x, self.conv1a, self.conv1b, self.batchnorm1, self.batchnorm2)
        x = self.pool(x)
        x = self.superblock(x, self.conv2a, self.conv2b, self.batchnorm3, self.batchnorm4)
        x = self.pool(x)
        x = self.superblock(x, self.conv3a, self.conv3b, self.batchnorm5, self.batchnorm6)
        x = self.pool(x)
        x = self.superblock(x, self.conv4a, self.conv4b, self.batchnorm7, self.batchnorm8)
        return x

    def semi_forward(self, x):
        assert x.max() > 0.01
        assert x.max() < 1.01
        x = self.vgg(x)
        semi_det = self.detector_head(x)
        desc = self.descriptor_head(x)
        prob = nn.functional.softmax(semi_det, dim=1)
        return prob, desc

    @staticmethod
    def expand_results(depth_to_space, prob):
        """
        Expand probability maps from (batch, 65, h//8, w//8) to (batch, 2, h, w) form

        :param depth_to_space:
        :param prob: utils.DepthToSpace instance
        :return: torch.Tensor
            (batch, 2, h, w) where result[:, 0] is non-point probabilities
        """
        if (prob.shape[1] % 2) == 0:
            return depth_to_space(prob)
        result = depth_to_space(prob[:, :-1, :, :])
        # expand no-point item to same dimentions as result
        len_axis_1 = prob.shape[1]
        s = slice(len_axis_1 - 1, len_axis_1)
        no_point = torch.cat([prob[:, s, :, :] for p in \
                              range(depth_to_space.block_size ** 2)], dim=1)
        exp_no_point = depth_to_space(no_point)
        tmp = []
        for i in range(result.shape[0]):
            tmp.append(torch.cat([exp_no_point[i], result[i]]))
        result = torch.stack(tmp)
        return result
