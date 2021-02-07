import os
import random
import enum
from operator import itemgetter

import torch
from torch.utils import data
import imageio
import numpy


class Mode(enum.Enum):
    """
    Mode of dataset to load
    """
    training = 1
    test = 2
    validation = 3


class ColorMode(enum.Enum):
    """
    Convert to greyscale or load as RGB
    """
    RGB = 1
    GREY = 2


def get_file_paths(files_path):
    result = []
    for filename in os.listdir(files_path):
        p = os.path.join(files_path, filename)
        if os.path.isfile(p):
            result.append((filename.split('.')[0], p))
    result.sort(key=itemgetter(0))
    return (x[1] for x in result)


class SynteticShapes(data.Dataset):
    """
    Class to read synthetic shapes dataset from a directory.
    Expects following layout:
    top_directory/
       subdirectory(like draw_cube)/
       ...
                /images
                    /test
                    /training
                    /validation
                /points
                    /test
                    /training
                    /validation
    """
    def __init__(self, path, mode=Mode.training, transform=None,
                 color=ColorMode.RGB, subset=None):
        """
        Create dataset
        :param path: str
        :param mode: Mode
        :param transform: callable
            data transformer, transform should accept two keyword arguments data, target;
            that is single image, and single keypoints array
        :param color: ColorMode
        :param subset: str
            if provided read only from subdirectories ending with subset
        """
        self.path = path
        self.mode = mode
        self.points = self._load_dataset(subset)
        no_op = lambda data, target: (data, target)
        if transform is None:
            transform = no_op
        self.transform = transform
        self.color = color
        self._size = len(self.points)

    def _load_image(self, path):
        pilmode = 'RGB'
        if self.color.value == ColorMode.GREY.value:
            pilmode = 'L'
        img = imageio.imread(path, pilmode=pilmode)
        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        return numpy.asarray(img)

    def __getitem__(self, idx):
        x, y = self.points[idx]
        img = self._load_image(x)
        target = numpy.load(y)
        if self.transform is not None:
            res = self.transform(data=img, target=target)
        else:
            res = data, target
        return res

    def __len__(self):
        return self._size

    def shuffle(self):
        random.shuffle(self.points)

    def load_items(self, path):
        mode = self.mode.name
        images = 'images'
        points = 'points'
        images_path = os.path.join(path, images, mode)
        numpy_path = os.path.join(path, points, mode)
        img_paths = list(get_file_paths(images_path))

        numpy_paths = [os.path.join(numpy_path, os.path.basename(x).replace(os.path.splitext(x)[1], '.npy')) for x in img_paths]
        result = list(zip(list(img_paths), list(numpy_paths)))
        return result

    def _load_dataset(self, subset=None):
        result = []
        p = self.path
        def filter_func(x):
            return os.path.isdir(x) and (subset is None or x.endswith(subset))
        # top-level directories
        top_dirs = list(filter(filter_func, (os.path.join(p, x) for x in  os.listdir(p))))
        # load images
        for top_dir in top_dirs:
            result.extend(self.load_items(top_dir))
        return result


class ImageDirectoryDataset(SynteticShapes):
    """
    Class to load images from a single directory
    e.g. do not go into subdirectories
    """

    def __init__(self, path, mode=Mode.training, transform=None,
                 color=ColorMode.RGB, subset=None):
        super().__init__(path, mode=mode, transform=transform,
                 color=color, subset=subset)
        if transform is None:
            self.transform = lambda data: data

    def __getitem__(self, idx):
        x = self.points[idx]
        img = self._load_image(x)
        transformed = self.transform(data=img)
        return transformed

    def _load_dataset(self, subset=None):
        result = []
        p = self.path
        for img_name in os.listdir(p):
            result.append(os.path.join(p, img_name))
        return result


class ImageSelfLearnDataset(ImageDirectoryDataset):
    """
    Class for creating a stack of homography-adapted images
    """
    def __getitem__(self, idx):
        x = self.points[idx]
        img = self._load_image(x)
        transformed = self.transform(img)
        images = torch.stack([torch.from_numpy(x.astype(numpy.float32)) for x in transformed[0]])
        H = torch.stack([torch.from_numpy(x) for x in transformed[1]])
        return images, H
