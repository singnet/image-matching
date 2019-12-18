import torch.nn.functional as F
import torch
import numpy


def make_grid(height, width):
    return torch.from_numpy(numpy.indices((width, height)).transpose().reshape((height * width, 2))).float()


def bilinear_sampling(x, H, w_template, h_template, to_numpy=True, mode='bilinear'):
    """
    Apply homography H to an image x.
    x shape is expected to follow opencv convention e.g. height x width x channels

    :param x:
    :param H:
    :param w_template:
    :param h_template:
    :return:
    """
    # assert len(x.shape) == 3
    # assert x.shape[-1] == min(x.shape)
    batch_size = 1
    n_channels = x.shape[-1]
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    if len(x.shape) == 4:
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
    else:
        x = x.permute(2, 0, 1).unsqueeze(0)

    grid = create_grid_batch(batch_size, h_template, w_template, x.device)
    grid_warped = apply_h_to_grid(H, grid, h_template, w_template)

    grid_warped[:, :, :, 0] = ((grid_warped[:, :, :, 0] / w_template) - 0.5) * 2
    grid_warped[:, :, :, 1] = ((grid_warped[:, :, :, 1] / h_template) - 0.5) * 2
    x_warped = torch.zeros((n_channels, batch_size, h_template, w_template))
    x_input = x.permute(1,0,2,3)

    for i in range(n_channels):
        xi_input = x_input[i, :, :, :]
        xi_input.unsqueeze_(1)
        xi_warped = F.grid_sample(xi_input.type(torch.FloatTensor).to(grid_warped.device),
                                  grid_warped,
                                  mode=mode,
                                  align_corners=True)
        # xi_warped.permute(1, 0, 2, 3)
        xi_warped.squeeze_()
        x_warped[i, :, :, :] = xi_warped

    x_warped = x_warped.permute(1, 2, 3, 0)
    x_warped.squeeze_()
    if to_numpy:
        x_warped = x_warped.detach().numpy().astype(numpy.uint8)

    return x_warped


def create_grid_batch(batch_size, h_template, w_template, device='cpu'):
    grid = make_grid(width=w_template, height=h_template)
    ones = torch.ones((grid.shape[0], 1))
    grid = torch.cat((grid, ones), dim=1)
    grid = torch.transpose(grid.repeat(batch_size, 1, 1), 1, 2).to(device)
    return grid


def apply_h_to_grid(H, grid, h_template=None, w_template=None):
    """
    Applies homography to an array of coordinates

    :param H: homography 3x3 torch.Tensor or numpy.array
    :param grid: 1x3xN - torch.Tensor of points
    :param h_template: height
    :param w_template: width
    :return: warped grid
    """

    if isinstance(H, torch.Tensor):
        H12 = H[:2, :]
    else:
        H12 = torch.from_numpy(H[:2, :]).to(grid.device)
    H12 = H12.unsqueeze_(0).to(grid.device)
    if isinstance(H, torch.Tensor):
        H3 = H[2, :]
    else:
        H3 = torch.from_numpy(H[2, :]).to(grid.device)
    H3 = H3.unsqueeze_(0).to(grid.device)
    grid_w = torch.matmul(H12.to(grid.dtype), grid)
    sn = torch.matmul(H3.to(grid.dtype), grid)
    # TODO: Check for division by zero?
    xw = grid_w[:, 0, :].unsqueeze_(1)
    xw = torch.div(xw, sn)

    # TODO: Is it necessary to clamp? (probably not)
    # xw.clamp_(-1, 1)

    yw = grid_w[:, 1, :].unsqueeze_(1)
    yw = torch.div(yw, sn)
    # yw.clamp_(-1, 1)
    if h_template is not None:
        xw = xw.view(-1, h_template, w_template, 1)
        yw = yw.view(-1, h_template, w_template, 1)
        grid_warped = torch.cat((xw, yw), dim=3)
    else:
        grid_warped = torch.cat((xw, yw), dim=1)
    return grid_warped


class HomographySamplerTransformer:
    """
    Applies random homography to an image
    """
    def __init__(self, num, **kwargs):
        """
        Initalize HomographySamplerTransformer with number of samples and
        :param num:
        :param beta: int
            How far(in pexels) can be moved image corners by randomization
            from it's original location
        :param theta: float
            Bounds for random rotations in radians, image will be rotated with
            random angle (-theta, theta)
        :param scale: float
            Coordinates scale, values of scale < 1.0 will cause zoom
        """
        from fem import noise
        self._homography = noise.HomographySample(**kwargs)
        self.num_samples = num
        self.sampler_kwargs = kwargs

    def sample_fixed_homography(self, h, w):
        """
        Fix homography for all examples with some random value
        """
        from fem import noise
        H, H_inv = self._homography.sample_homography(h, w)
        self._homography = noise.HomographySample(H=H,
                                                        H_inv=H_inv,
                                                        **self.sampler_kwargs)
        # # DEBUG with fixed homography
        # pts = numpy.array([[0, 0],
        #                    [w, 0],
        #                    [0, h],
        #                    [w, h]])
        # shift = [w/2.0, h/2.0]
        # theta = 0.2
        # R = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
        #      [numpy.sin(theta), numpy.cos(theta)]])
        # pts = (pts - shift) @ R + shift
        # self._homography = photometric.HomographySample(self.beta,
        #                                               *self._homography.find_homography(h, w,
        #                                                                               pts))

        # self._homography = photometric.HomographySample(self.beta,
        #                                                 numpy.eye(3).astype(numpy.float64),
        #                                                 numpy.eye(3).astype(numpy.float64))

    def __call__(self, img):
        """
        Apply current homography to grayscale images img
        :param img: batch * height * width
        :return: tuple of two numpy.array
            first is [unmodified image, homography image, ...]
            second is [homography, inverse homography, homography...]
            each array has length of self.num_samples
        """
        argmin = numpy.argmin(img.shape)
        assert len(img.shape) == 3
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if argmin == 0:
            img = img.transpose(1, 2, 0)
        tmp = [img]
        h_tmp = []
        mask = []
        for i in range(self.num_samples):
            hsample = self._homography(img)
            tmp.append(hsample['template_img'])
            h_tmp.append(numpy.vstack([hsample['p'], hsample['p_inv']]))
            mask.append(hsample['mask'])
        return numpy.stack(tmp), numpy.stack(h_tmp), numpy.stack(mask)

    def mask(self, h, w):

        img = numpy.ones((h, w), dtype=numpy.float32)
        hsample = self._homography(img)
        return hsample



