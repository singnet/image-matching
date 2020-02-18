

import os
import os.path
import imageio
from torch import nn

import torch
import numpy

from fem.hom import HomographySamplerTransformer
from fem.hom import bilinear_sampling
from fem.dataset import ColorMode
from fem import dataset
from torchvision.transforms import Compose
from fem.noise import resize



def resize_to_8(img):
    h, w = img.squeeze().shape
    new_h = h // 8 * 8
    new_w = w // 8 * 8
    if (new_h != h) or (new_w != w):
        print("resizing {0} to {1}".format((h,w), (new_h, new_w)))
        result = resize(img, (new_w, new_h))
        new_shape = result.squeeze().shape
        assert new_shape[0] - h <= 8
        assert new_shape[1] - w <= 8
        assert new_shape[0] % 8 == 0
        assert new_shape[1] % 8 == 0
    else:
        result = img
    return result


def convert_dataset(model, path, base_path, new_base, threshold=0.45):
    """
    Perform intersting point detection on dataset using homography adaptation

    :param model: object
        keypoint detector
    :param path: str
        path to dataset
    :param base_path: str
        top-level part of source dataset.
        This is the part which will be ignored during directory structure recreation.
    :param: new_base: str
        images and keypoints will be saved in directory relative to new_base.
    :param: threshold
        probability threshold for keypoints
    :return: None
    """
    # with this scale rotated image will likely fit in bounds,
    # otherwise more tricky scheme for mean probability computation will
    # be needed to account for some pixels not being processed, because due
    # to rotation they are outside of bounds

    homography = HomographySamplerTransformer(80, beta=45, theta=0.1,
                                              random_scale_range=(1.0, 1.3),
                                              perspective=55
                                              )

    data_transformer = Compose([resize_to_8, homography])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Augmentator(model).to(device).eval()
    data = dataset.ImageSelfLearnDataset(path,
                                         color=ColorMode.GREY,
                                         transform=data_transformer)
    data.shuffle()
    mode = 'training'

    def callback(idx):
        images, H = data[idx]
        if max(images.shape) < 100:
            raise RuntimeError("small image {0}".format(images.shape))
        H_ = H[:, 0:3].to(device).squeeze()
        H_inv = H[:, 3:6].to(device).squeeze()
        return gen.projection(images.to(device), H_, H_inv).squeeze(), images[0]

    for i in range(len(data)):

        target = lambda: callback(i)
        save_files(base_path, data.points[i], mode, new_base, target)
        if i % 100 == 0:
            print("iteration %i" % i)


def save_files(base_path, img_path, mode, new_base, callback):
    """
    Save image and points in format acceptable by dataset.SynteticShapes

    :param base_path: str
        top-level part of source dataset
    :param img_path: str
        path of current image in source dataset
    :param image: torch.Tensor or numpy.array
        array of channels x height x width
    :param mode:
    :param new_base:
    :param callback:
    :return:
    """

    new_path = img_path.replace(base_path, new_base)
    dirname = os.path.dirname(new_path)
    format = new_path.split('.')[-1]
    points_path = os.path.join(dirname, 'points', mode)
    images_path = os.path.join(dirname, 'images', mode)
    os.makedirs(points_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    basename = os.path.basename(new_path)

    # test
    #import drawing
    #drawing.show_points(image[0].numpy().astype(numpy.uint8), pts.astype(numpy.uint8), 'w')
    # test
    npy_path = os.path.join(points_path, basename.replace('.' + format, '.npy'))
    if os.path.exists(npy_path):
        return
    try:
        points, image = callback()
    except RuntimeError as e:
        print(e)
        print('skipping path: ' + npy_path)
        return
    assert min(image.shape) == 1 or len(image.shape) == 2
    imageio.imwrite(os.path.join(images_path, basename), image.squeeze().numpy().astype(numpy.uint8))
    numpy.save(npy_path,
               points)

def reproject(H_inv, prob, mode):
    """
    compute average of projections

    :param H_inv: batch * 3 * 3
    :param prob: batch * 2 * h * w
    :param mode: "nearest" or "bilinear"
    :return: batch * h * w
    """
    rep = [bilinear_sampling(r.permute(1, 2, 0), h.float(),
                             prob.shape[-1],
                             prob.shape[-2],
                             to_numpy=False, mode=mode)
           for r, h in zip(prob, H_inv)]
    rep1 = torch.stack(rep)
    # for i in range(len(H)):
    #    im = img_reproj[i].numpy().astype(numpy.uint8)
    #    draw_points(im, [x[1:] for x in coords_rep if x[0] == i])
    #    cv2.imshow('reproj' + str(i), im)
    return rep1


class Augmentator(nn.Module):
    def __init__(self, model, batch_size=16):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def projection(self, images, H, H_inv):
        """
        Extract keypoint location on homography augmented images

        :param images: torch.Tensor
            shape is (batch, h, w)
        :param hom: torch.Tensor
            shape is (batch, 6, 3), where (:, 0:3) is homography (:, 3:6) is inverse homography
        :return:
        """
        source_image = images[0]
        train_images = images[1:]
        # H_inv = [torch.from_numpy(numpy.linalg.inv(h)) for h in H]
        # for each image in images find points
        # project points back using H^-1 and average
        # apply NMS
        # project back masks and average to compute gradient mask.
        # use resulting image as target in loss function
        # apply mask to gradient
        prob = self.prob(train_images)
        # reproject using bilinear sampling
        rep1 = reproject(H_inv, prob, 'nearest')
        mean = rep1.mean(dim=0)
        # test
        # threshold = 0.25
        # points = self.get_points(mean, threshold)
        # import drawing
        # img = source_image.cpu().numpy().astype(numpy.uint8)
        # drawing.show_points(train_images[14] / 255., [], 'img14')
        # drawing.show_points(train_images[15] / 255., [], 'img15')
        # drawing.show_points(img, points, 'w')
        # drawing.show_points(img, self.get_points(self.nms(rep1[14].unsqueeze(0).unsqueeze(0)),
        #                                          threshold)[:,2:], 'w14')
        # import pdb;pdb.set_trace()
        # # test end
        return mean

    def prob(self, train_images):
        steps = max(train_images.shape[0] // self.batch_size, 1)
        tmp = []
        for i in range(steps):
            sp = slice(i * self.batch_size, i * self.batch_size + self.batch_size)
            tmp.append(self.model.heatmap(train_images[sp].permute(0, 3, 1, 2) / 255.0).detach())
        return torch.cat(tmp)


if __name__ == '__main__':
    from fem.goodpoint import GoodPoint
    path = '/mnt/fileserver/shared/datasets/at-on-at-data/COCO/train2014/'
    dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light/img'
    gtav_path = '/mnt/fileserver/shared/datasets/from_games/GTAV/02_images/images/'
    new_base = '/tmp/dataset/'
    base_path = '/mnt/fileserver/shared/datasets/'
    super_file = 'snapshots/super1600.pt'
    device = 'cpu'
    sp = GoodPoint(activation=torch.nn.ReLU(), grid_size=8,
               batchnorm=True, dustbin=0).to(device)
    state_dict = torch.load(super_file, map_location=device)
    sp.load_state_dict(state_dict['superpoint'])
    convert_dataset(sp,
                    path,
                    base_path=base_path,
                    new_base=new_base)


