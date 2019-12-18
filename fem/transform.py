import numpy
import torch
import random
import abc


class ToTensor:
    def __call__(self, arg):
        return torch.from_numpy(arg)


def apply_transformers(x, transformers=[]):
    for item in transformers:
        if callable(item):
            x = item(x)
        else:
            x = item[0](x, **item[1])
    return x


def random_transformer(x, transformers=[]):
    orig_shape = x.shape
    assert (len(orig_shape) == 3)
    assert (orig_shape[2] <= orig_shape[0])
    assert (orig_shape[2] <= orig_shape[1])

    x = x.transpose(2, 0, 1)
    for transform in transformers:
        assert 'resize' not in str(transform.__class__).lower()
        if random.randint(0, 1):
            x_std = x.std()
            if x_std < 0.01:
                continue
            new_x = transform(x)
            if new_x.std() < 0.1 * x_std:
                print("{0} decreased std to {1} from {2}".format(transform, new_x.std(), x_std))
                continue
            x = new_x
    x = x.astype(numpy.float32)
    x = x.transpose(1, 2, 0)
    return x


class RandomTransformer:
    def __init__(self, transformers):
        self.transformers = transformers

    def __call__(self, x):
        return random_transformer(x, self.transformers)


class TransformCompose:

    @abc.abstractmethod
    def __call__(self, data=None, target=None):
        pass


class DataLabelCompose(TransformCompose):
    def __init__(self, data_transform, label_transform):
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __call__(self, data=None, target=None):
        return self.data_transform(data), self.label_transform(target)


class DataCompose(TransformCompose):
    def __init__(self, data_transform):
        self.data_transform = data_transform

    def __call__(self, data=None, target=None):
        return self.data_transform(data)


class FCompose(TransformCompose):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data=None, target=None):
        for tr in self.transforms:
            data, target = tr(data=data, target=target)
        return data, target


def label_transphormer(x, label_transformers=[], to_torch=None):
    for function, config in label_transformers:
        x = function(x, **config)
    if to_torch:
        x = torch.from_numpy(x).float()
    return x
