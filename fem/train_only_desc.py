import os
import sys

from collections import defaultdict
import ignite
from packaging import version

import torch.optim as optim
from fem import util
from fem.util import mean
from fem.reinf_utils import threshold_nms, threshold_nms_dense
from torch.utils.data import DataLoader
from training import Stats

from fem.dataset import MinecraftDataset, Mode, ColorMode, ImageDirectoryDataset
from fem.transform import *
from fem import util

from ignite.engine import Engine
from fem.goodpoint import GoodPoint

import numpy
from torchvision.transforms import Compose

from fem.loss_reinf import compute_loss_hom_det
from fem.noise_transformer import NoisyTransformWithResize
from fem.loss_desc import compute_loss_desc


gtav_super = '/mnt/fileserver/shared/datasets/from_games/GTAV_super/01_images'
coco_super = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/'
coco_images = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/train2014/images/training/'
synth_path = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/mysynth/'

dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light'
dir_night = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_night_light'
poses_file = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/village_00.json'


from fem.training import train_loop, test, exponential_lr


def randomize(coords, low, high):
    coords[0] += torch.randint_like(coords[0], low, high)
    coords[1] += torch.randint_like(coords[1], low, high)
    return coords


def make_noisy_transformers():
    from fem.noise import AdditiveGaussian, RandomBrightness, AdditiveShade, MotionBlur, SaltPepper, RandomContrast
    totensor = ToTensor()
    # ColorInversion doesn't seem to be usefull on most datasets
    transformer = [
                   AdditiveGaussian(var=30),
#                   RandomBrightness(range=(-50, 50)),
                   AdditiveShade(kernel_size_range=[45, 85],
                                 transparency_range=(-0.25, .45)),
#                   SaltPepper(),
                   MotionBlur(max_kernel_size=5),
#                   RandomContrast([0.6, 1.05])
                   ]
    return Compose([RandomTransformer(transformer), totensor])


def train_iteration(batch, model, optimizer, scheduler, **kwargs):
    imgs = torch.cat([batch['img1'], batch['img2']]).float()
    height, width = imgs.shape[1], imgs.shape[2]
    assert len(imgs.shape) == 4

    desc_model = kwargs.get('desc_model', None)
    assert numpy.argmin(imgs.shape) == 3
    imgs = imgs.permute(0, 3, 1, 2)
    heatmaps, desc = model(imgs / 255.0)

    desc1 = desc[:len(desc) // 2]
    desc2 = desc[len(desc) // 2:]
    point_mask1 = torch.zeros_like(heatmaps[:len(desc)// 2])
    point_mask2 = torch.zeros_like(heatmaps[:len(desc)// 2])
    point_mask1 = model.expand_results(model.depth_to_space, point_mask1)
    point_mask2 = model.expand_results(model.depth_to_space, point_mask2)
    coords16 = torch.meshgrid(torch.arange(16, 256, 32), torch.arange(16, 256, 32), indexing='ij')
    coords32 = torch.meshgrid(torch.arange(8, 256, 16), torch.arange(8, 256, 16), indexing='ij')
    # permute coords a bit
    coords32 = torch.stack([coords32[0], coords32[1]])
    coords16 = torch.stack([coords16[0], coords16[1]])

    coords1 = randomize(coords16.clone(), -2, 3)
    coords2 = randomize(coords16.clone(), -2, 3)

    point_mask1[:, :, coords1[0], coords1[1]] = 1
    point_mask2[:, :, coords2[0], coords2[1]] = 1

    # now we can project points to homographically augmented images
    point_mask1 = point_mask1.squeeze()
    point_mask2 = point_mask2.squeeze().clone()

    _3d_data = batch
    tmp_results = defaultdict(list)
    for i in range(len(desc1)):
        tmp = compute_loss_desc({k: v[i] for (k,v) in _3d_data.items()},
                point_mask1[i], point_mask2[i],
                desc1[i], desc2[i], kwargs['norm_descriptors'])
        for key, value in tmp.items():
            tmp_results[key].append(value)
    result = {key: mean(value) for (key, value) in tmp_results.items()}
    optimizer.zero_grad()
    result['loss_desc'].backward()
    optimizer.step()
    scheduler.step()
    result = {k: v.item() for (k, v) in result.items()}
    result['lr'] = numpy.mean(scheduler.get_last_lr())
    return result


def train_good():
    batch_size = 22
    test_batch_size = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))
    lr = 0.0015
    epochs = 50
    weight_decay = 0.0005
    norm_descriptors = True
    parallel = False
    stats = Stats()

    super_file = "./mine0.pt"

    gp = GoodPoint(n_channels=3,
               activation=torch.nn.LeakyReLU(),
               grid_size=8,
               batchnorm=False,
               dustbin=0,
               desc_out=8,
               norm_descriptors=norm_descriptors).to(device)
    optimizer = optim.AdamW(gp.parameters(), lr=lr)
    if os.path.exists(super_file):
        state_dict = torch.load(super_file, map_location=device)
        print("loading weights from {0}".format(super_file))
        gp.load_state_dict(state_dict['superpoint'])

    decay_rate = 0.9
    decay_steps = 1000

    if parallel:
        gp = torch.nn.DataParallel(gp)

    def l(step, my_step=[0]):
        my_step[0] += 1
        return exponential_lr(decay_rate, my_step[0], decay_steps, staircase=False)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

    m_loader = get_loaders(batch_size, test_batch_size, 1)

    for epoch in range(epochs + 1):
        for i, batch in enumerate(m_loader):
            batch = {k: v.to(device) for (k, v) in batch.items()}
            result = train_iteration(batch, gp, optimizer, scheduler,
                    norm_descriptors=norm_descriptors)
            stats.update(**result)
            if i % 15 == 0:
                print(stats.stats)
    torch.save({'optimizer': optimizer.state_dict(),
            'superpoint': gp.state_dict()}, 'mine{0}.pt'.format(scheduler.last_epoch))


def get_loaders(batch_size, test_batch_size, num):
    dataset_path = '/mnt/fileserver/shared/datasets/minecraft-segm'
    noisy = make_noisy_transformers()
    m_dataset = MinecraftDataset(dataset_path,
                                 transform=NoisyTransformWithResize(num=num, noisy=noisy, theta=0.01,
                                                                    perspective=5),
                                 color=ColorMode.RGB)
                                 #color=ColorMode.GREY)

    m_loader = DataLoader(m_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return m_loader


if __name__ == '__main__':
    train_good()
