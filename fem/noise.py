import skimage.exposure

from scipy import ndimage
import numpy
import cv2

from fem.hom import bilinear_sampling
import scipy.ndimage
from skimage import util


def random_brightness(images, per_color=True, channel=0, range=(0, 70)):
    aligned_range = range[1] - range[0]
    if per_color:
        new_shape = [1 for x in images.shape]
        new_shape[channel] = images.shape[channel]
        r = numpy.random.random(new_shape) * aligned_range + range[0]
    else:
        r = numpy.random.random() * aligned_range + range[0]
    res = images + r
    return numpy.clip(res, 0, 255)


class RandomBrightness:
    def __init__(self, per_color=True, channel=0, range=(-30, 70)):
        self.per_color = per_color
        self.channel = channel
        self.range = range

    def __call__(self, image):
        return random_brightness(images=image,
                                 per_color=self.per_color,
                                 channel=self.channel,
                                 range=self.range)

# noise adapted from
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

def additive_gaussian(image, mean=0, var=None):
    """
    Additive gaussian noise
    Parameters
    ----------
    image : ndarray
        numpy array of any shape
    mean: mean of gaussian
    var: variance of gaussian
        default value = image.var()
    """
    if var is None:
        var = image.var()
    sigma = var**0.5
    gauss = numpy.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return numpy.clip(noisy, 0, 255)


class AdditiveGaussian:
    def __init__(self, mean=0, var=None):
        self.mean = mean
        self.var = var

    def __call__(self, image):
        return additive_gaussian(image, mean=self.mean, var=self.var)


class SaltPepper:
    def __init__(self, s_vs_p=0.5, amount=0.04):
        self.s_vs_p = s_vs_p
        self.amount = amount

    def __call__(self, image):
        return snp(image, s_vs_p=self.s_vs_p, amount=self.amount)


def snp(image, s_vs_p=0.5, amount=0.04):
    """
    salt & papper noise, randomly sets pixels to 0 or 255
    Parameters
    ----------
    image: ndarray
    s_vs_p: float
        ratio of salt noise, amount of papper noise is (1 - s_vs_p)
    amount: float
        ration of noisy pixes vs total number of pixes
    """
    out = image
    # Salt mode
    num_salt = numpy.ceil(amount * image.size * s_vs_p)
    coords = [numpy.random.randint(0, i - 1 if i != 1 else 1, int(num_salt))
              for i in image.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = numpy.ceil(amount* image.size * (1. - s_vs_p))
    coords = [numpy.random.randint(0, i - 1 if i != 1 else 1, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0
    return out


class Speckle:
    def __init__(self, var=0.125):
        self.var = var

    def __call__(self, image):
        return speckle(image, var=self.var)


def speckle(image, var=0.125):
    """
    Speckle noise
    Parameters
    ----------
    image: ndarray
    var: float
        variance of gaussian distribution
    """
    gauss = numpy.random.randn(*image.shape) * var
    noisy = image  + gauss * image
    return numpy.clip(noisy, 0, 255)


def additive_shade(image, nb_ellipses=20, transparency_range=[-0.6, 0.2],
                   kernel_size_range=[250, 350]):

    def _py_additive_shade(img):
        dtype = img.dtype
        min_dim = min(img.shape[1:]) / 4
        mask = numpy.zeros(img.shape[1:], numpy.uint8)
        for i in range(nb_ellipses):
            ax = int(max(numpy.random.rand() * min_dim, min_dim / 5))
            ay = int(max(numpy.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = numpy.random.randint(max_rad, img.shape[2] - max_rad)  # center
            y = numpy.random.randint(max_rad, img.shape[1] - max_rad)
            angle = numpy.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = numpy.random.uniform(*transparency_range)
        kernel_size = numpy.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(numpy.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[numpy.newaxis, ...]/255.)
        return numpy.clip(shaded, 0, 255).astype(dtype)

    return _py_additive_shade(image)


class AdditiveShade:
    def __init__(self, nb_ellipses=20, transparency_range=[-0.6, 0.2],
                   kernel_size_range=[250, 350]):
        self.nb_ellipses = nb_ellipses
        self.transparency_range = transparency_range
        self.kernel_size_range = kernel_size_range

    def __call__(self, image):
        return additive_shade(image,
                              nb_ellipses=self.nb_ellipses,
                              transparency_range=self.transparency_range,
                              kernel_size_range=self.kernel_size_range)


class RandomContrast:
    def __init__(self, strength_range=[0.5, 1.5]):
        self.strength_range = strength_range

    def __call__(self, image):
        return random_contrast(image, strength_range=self.strength_range)


def random_contrast(image, strength_range=[0.5, 1.5]):
    old_max = image.max()
    old_min = image.min()
    old_range = old_max - old_min
    new_range = old_range * (numpy.random.random() * (strength_range[1]  - strength_range[0]) + strength_range[0])
    middle = old_max / 2.0 + old_min / 2.0
    new_min = middle - new_range / 2.0
    new_max = middle + new_range / 2.0
    return numpy.clip(skimage.exposure.rescale_intensity(image, out_range=(new_min, new_max)), 0, 255)


def motion_blur(img, max_kernel_size=10):
    # Either vertial, hozirontal or diagonal blur
    mode = numpy.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = numpy.random.randint(2, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = numpy.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = numpy.eye(ksize)
    elif mode == 'diag_up':
        kernel = numpy.flip(numpy.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = numpy.repeat(numpy.arange(ksize)[:, numpy.newaxis], ksize, axis=-1)
    gaussian = numpy.exp(-(numpy.square(grid-center)+numpy.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= numpy.sum(kernel)
    shape = img.shape
    if shape[0] == 1:
        img = cv2.filter2D(img[0], -1, kernel).reshape(shape)
    else:
        img = cv2.filter2D(img, -1, kernel)
    return img


class MotionBlur:
    def __init__(self, max_kernel_size=4):
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        return motion_blur(img, max_kernel_size=self.max_kernel_size)


class Blur:
    def __init__(self, sigma=4):
        self.sigma = sigma

    def __call__(self, img):
        return blur(img, sigma=self.sigma)


def blur(img, sigma=4):
    random_sigma = min(0.5, numpy.random.random()) * sigma
    return ndimage.gaussian_filter(img, sigma=random_sigma)


def resize(img, size):
    if len(img.shape) == 3 and numpy.argmin(img.shape) == 0:
        img = img.transpose(1, 2, 0)
    if len(img.shape) == 3 and img.shape[2] != 1:
        return cv2.resize(img, size)
    return cv2.resize(img, size)[numpy.newaxis, :]


class Resize:
    def __init__(self, size, keep_ratio=False):
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, img, return_size=False):
        data_res, size = self.resize_data(img)
        if return_size:
            return data_res, size
        return data_res

    def resize_data(self, img):
        if self.keep_ratio:
            assert numpy.argmin(img.shape) == 2 or len(img.shape) == 2
            if img.shape[0] / self.size[0] != img.shape[1] / self.size[1]:
                main_axis = numpy.argmin(img.shape[:-1])
                ratio = img.shape[main_axis] / self.size[main_axis]
                new_size = round(img.shape[1] / ratio), round(img.shape[0] / ratio)
                return resize(img, new_size), new_size
        return resize(img, self.size), self.size


def resize_keypoints_scipy(keypoints, size):
    h_factor = size[0] / keypoints.shape[0]
    w_factor = size[1] / keypoints.shape[1]
    result = scipy.ndimage.zoom(keypoints, (h_factor, w_factor), order=0)
    assert result.shape[0] == size[0]
    assert result.shape[1] == size[1]
    return result


def resize_keypoints(keypoints, size):
    assert len(keypoints.shape) == 2
    h = size[1] / keypoints.shape[0]
    w = size[0] / keypoints.shape[1]
    coords = numpy.nonzero(keypoints)
    new_coords = tuple(numpy.rint(x).astype(numpy.int) for x in (coords[0] * h, coords[1] * w))
    result = numpy.zeros((size[1], size[0]))
    result[new_coords] = 1
    return result


class Threshold:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        return x >= self.threshold


class ExtractKeypoints(Threshold):

    def __call__(self, x):
        coords = (x >= self.threshold).nonzero()
        return numpy.stack(coords).transpose()


class ResizeKeypoints:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return resize_keypoints(x, self.size)


class ResizeKeypointsScipy(ResizeKeypoints):
    def __call__(self, x):
        return resize_keypoints_scipy(x, size=self.size)


class ColorInversion:
    def __call__(self, x):
        inverted = util.invert(x)
        return inverted


def random_crop(image, output_size, return_pos=False):
    argmin = numpy.argmin(image.shape)
    if argmin == 0:
        image = image.transpose(1, 2, 0)
    h, w = image.shape[:2]
    new_h, new_w = output_size
    diff_h = h - new_h
    diff_w = w - new_w
    top = 0 if (diff_h <= 0) else numpy.random.randint(0, diff_h)
    left = 0 if (diff_w <= 0) else numpy.random.randint(0, diff_w)
    image = image[top: top + new_h, left: left + new_w]
    if argmin == 0:
        image = image.transpose(2, 0, 1)
    if return_pos:
        return image, (top, left)
    return image


class RandomCropTransform:
    def __init__(self, size, beta=0):
        self.size = size
        self.beta = beta

    def __call__(self, data, return_pos=False):
        if self.beta:
            size = self.size + int(numpy.random.random() * self.beta * 2) - self.beta
        else:
            size = self.size
        img, pos = random_crop(data, (size, size), return_pos=True)
        if return_pos:
            return img, pos
        return img


class HomographySample:
    HEIGHT = 1
    WIDTH = 0

    def __init__(self, beta=None, H=None, H_inv=None, theta=None,
                 fixed_scale=1.0, random_scale_range=None,
                 perspective=None):
        """
        Random homography sampler. Homography will be sampled using
        four points (0,0), (0, width), (height, 0), (heigth, width)

        :param beta: float
            Parameter of uniform distribution, from which random shifts along x,y directions
            will be generatate. Random shift is generated for each point independently.
        :param H: numpy.array
            fixed homography 3x3
        :param H_inv:numpy.array
            fixed inverse homography
        :param theta: float
            parameter for random rotation in radian.
            Rotation will be in range (-theta, theta)
        :param fixed_scale: float
            fixed zoom, value > 1 gives zoom out, < 1 zoom in
        :param random_scale_range: tuple
            tuple of lower and upper bounds for random scale sampling
        :param perspective: float
            shift of left-top, left-bottom or right-top, right-bottom points towards each other
        """
        self.beta = beta
        self.H = H
        self.H_inv = H_inv
        self.theta = theta
        self.scale = fixed_scale
        self.random_scale_range = random_scale_range
        self.perspective = perspective

    def __call__(self, image):
        assert numpy.argmin(image.shape) != 0
        h, w = image.shape[:2]

        '''
        Define the initial warp W(x; p0) to be a translation that maps the domain of the template
        image to a square window centering in I
        '''

        # TODO: the maximum angle of the quadrilateral is restricted to being less than 3 4π during the random perturbation process

        h = image.shape[0]
        w = image.shape[1]
        if any(x is None for x in [self.H, self.H_inv]):
            H, H_inv = self.sample_homography(h, w)
        else:
            H, H_inv = self.H, self.H_inv

        img_template = bilinear_sampling(image, H, w, h)
        mask = bilinear_sampling(numpy.ones(image.shape, dtype=numpy.float32) * 255.0, H, w, h)
        mask = (mask > 0.5).astype(numpy.float32)
        sample = {'input_img': image,
                  'template_img': img_template.reshape(image.shape),
                  'p': H,
                  'p_inv': H_inv,
                  'mask': mask}
        return sample

    def find_homography(self, h, w, pts_pert):
        """

        :param h:
        :param w:
        :param pts_pert:
            (0, 0),
            (h, 0),
            (0, w),
            (h, w)
        :return:
        """
        pts_init = self.pts_init(h, w)
        H, H_inv = self._caluculate_h(pts_init, pts_pert)
        return H, H_inv

    def _caluculate_h(self, pts_init, pts_pert):
        H, _ = cv2.findHomography(pts_init, pts_pert)
        H_inv, _ = cv2.findHomography(pts_pert, pts_init)
        H = H.astype(numpy.float32)
        return H, H_inv

    def sample_homography(self, h, w):

        pts_init = self.pts_init(h, w)
        pts_pert = self.pts_init(h, w).transpose()
        if self.beta is not None:
            pts_rand = numpy.random.randint(low=-self.beta,
                                            high=self.beta,
                                            size=(2, 4)).astype(pts_init.dtype)
            pts_pert = pts_rand + pts_pert
        if self.perspective is not None:

            persp = numpy.random.randint(low=0, high=self.perspective)

            # pts_pert is 2 x 4 with width, height order
            # (0, 0) and (0, height)
            left = numpy.random.random() > 0.5
            side = numpy.random.random() > 0.5
            if side:
                self.perspective_side(left, persp, pts_pert)
            else:
                self.perspective_top_bottom(left, persp, pts_pert)
        shift = [w / 2.0, h / 2.0]
        scale = self.scale
        if self.random_scale_range is not None:
            rs_low, rs_top = self.random_scale_range
            range = (rs_top - rs_low)
            scale = scale * numpy.random.random() * range + rs_low
        pts_pert = (((pts_pert.transpose() - shift) * scale) + shift).transpose()
        if self.theta is not None:
            theta = numpy.random.random() * 2 * self.theta - self.theta
            R = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                 [numpy.sin(theta), numpy.cos(theta)]])
            pts_pert = ((pts_pert.transpose() - shift) @ R + shift).transpose()

        pts_pert = numpy.transpose(pts_pert)
        H, H_inv = self._caluculate_h(pts_init, pts_pert)
        return H, H_inv

    def perspective_side(self, left, persp, pts_pert):
        if left:
            pts_pert[self.HEIGHT][0] -= persp
            pts_pert[self.HEIGHT][2] += persp
        else:
            pts_pert[self.HEIGHT][1] -= persp
            pts_pert[self.HEIGHT][3] += persp

    def perspective_top_bottom(self, left, persp, pts_pert):
        if left:
            pts_pert[self.WIDTH][0] -= persp
            pts_pert[self.WIDTH][1] += persp
        else:
            pts_pert[self.WIDTH][2] -= persp
            pts_pert[self.WIDTH][3] += persp

    def pts_init(self, h, w):
        pts_init = numpy.zeros((2, 4))
        # e0 = (0, 0)
        pts_init[:, 1] = w, 0  # e1 = (width, 0)
        pts_init[:, 2] = 0, h  # e2 = (0, height)
        pts_init[:, 3] = w, h  # e3 = (width, height)
        pts_init = numpy.transpose(pts_init)
        return pts_init


def unfold_label(points, height, width):
    """
    generates labels of shape
    (2, height, width)

    Parameters
    ----------
    points: numpy.array
    height: int
    width: int

    Returns
    -------
    tensor of shape (2, height, width)
    """
    if len(points):
        assert points[:, 0].max() < height
        assert points[:, 1].max() < width
    tmp = numpy.ones((2, height, width))
    # (no-point, point), (no-point, point)..
    idx_no_point = 0
    idx_point = 1
    tmp[idx_point, :, :] *= 0
    # update result for current array
    h = tuple(min(height - 1, p) for p in numpy.rint(points[:, 0]).astype(numpy.int))
    w = tuple(min(width - 1, p) for p in numpy.rint(points[:, 1]).astype(numpy.int))

    tmp[idx_no_point, h, w] = 0
    tmp[idx_point, h, w] = 1
    return tmp[idx_point]


class UnfoldLabels:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, points):
        return unfold_label(points, height=self.height,
                            width=self.width)


mapping = {'random_brightness': RandomBrightness,
           'additive_gaussian': AdditiveGaussian,
           'salt_and_pepper': SaltPepper,
           'additive_shade': AdditiveShade,
           'speckle': Speckle,
           'random_contrast': RandomContrast,
           'motion_blur': MotionBlur,
           'blur': Blur,
           'resize': Resize
           }

label_mapping = {'resize': ResizeKeypoints}
