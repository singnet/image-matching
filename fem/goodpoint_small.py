from fem.goodpoint import *


class GoodPointSmall(GoodPoint):
    def __init__(self, grid_size, n_channels=1, activation=nn.ReLU(),
                 batchnorm=True, dustbin=0, nms=None, align_corners=True,
                 base1=64, base2=None, base3=None):
        super().__init__(grid_size=grid_size, n_channels=n_channels,
                 activation=activation, batchnorm=batchnorm,
                 dustbin=dustbin, nms=nms, align_corners=align_corners)
        self.align_corners = align_corners
        self.dustbin = dustbin
        self.activation = activation
        stride = 1
        kernel = (3, 3)

        # quarter
        if base2 is None:
            base2 = base1 * 2
        if base3 is None:
            base3 = base2 * 2


        self.conv1a = nn.Conv2d(n_channels, base1, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv1b = nn.Conv2d(base1, base1, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv2a = nn.Conv2d(base1, base1, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv2b = nn.Conv2d(base1, base1, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv3a = nn.Conv2d(base1, base2, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv3b = nn.Conv2d(base2, base2, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv4a = nn.Conv2d(base2, base2, kernel_size=kernel,
                        stride=stride, padding=1)

        self.conv4b = nn.Conv2d(base2, base2, kernel_size=kernel,
                        stride=stride, padding=1)
        self.pool = nn.MaxPool2d((2, 2))

        # Detector head
        self.convPa = torch.nn.Conv2d(base2, base3, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(base3, 64 + dustbin, kernel_size=1, stride=1, padding=0)
        # Descriptor
        self.convDa = torch.nn.Conv2d(base2, base3, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(base3, 256, kernel_size=1, stride=1, padding=0)
        if batchnorm:
            self.batchnorm0 = nn.BatchNorm2d(base1)
            self.batchnorm1 = nn.BatchNorm2d(base1)
            self.batchnorm2 = nn.BatchNorm2d(base1)
            self.batchnorm3 = nn.BatchNorm2d(base1)
            self.batchnorm4 = nn.BatchNorm2d(base2)
            self.batchnorm5 = nn.BatchNorm2d(base2)
            self.batchnorm6 = nn.BatchNorm2d(base2)
            self.batchnorm7 = nn.BatchNorm2d(base2)
            self.batchnormPa = nn.BatchNorm2d(base3)
            self.batchnormPb = nn.BatchNorm2d(64 + dustbin)
            self.batchnormDa = nn.BatchNorm2d(base3)
        else:
            l = lambda x: x
            self.batchnorm0 = l
            self.batchnorm1 = l
            self.batchnorm2 = l
            self.batchnorm3 = l
            self.batchnorm4 = l
            self.batchnorm5 = l
            self.batchnorm6 = l
            self.batchnorm7 = l
            self.batchnormDa = l
            self.batchnormPb = l
            self.batchnormPa = l



        self.depth_to_space = DepthToSpace(grid_size)
        self.nms = nms
        init_weights(self)
