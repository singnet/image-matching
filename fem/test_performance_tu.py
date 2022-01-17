"""
measure execution time
"""

import torch
import numpy as np
import numpy
import time
from nonmaximum import PoolingNms
import os
from fem.drawing import show_points
from torch.utils.data import DataLoader
from dataset import ImageDirectoryDataset, ColorMode
from goodpoint import GoodPoint


device = 'cuda'
device = 'cpu'
def get_model_orig():
    weight = "./snapshots/super3400.pt"
    nms = PoolingNms(8)
    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    sp.to(device)
    return sp


def get_model_distilled():
    weight = "./snapshots/distilled_32_32_64.pt"
    weight = "./snapshots/distilled_32_64_128.pt"
    nms = PoolingNms(8)
    from goodpoint_small import GoodPointSmall
    sp = GoodPointSmall(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms,
                   base1=32).eval()
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    sp.to(device)
    return sp



def main():
    room1_path = "$pathDatasetTUM_VI/dataset-room1_512_16/mav0/cam0/data"
    outdoor8_path = "$pathDatasetTUM_VI/dataset-outdoors8_512_16/mav0/cam0/data"
    path = os.path.expandvars(outdoor8_path)
    transform = lambda data: data.astype(numpy.float32)
    dataset = ImageDirectoryDataset(path, transform=transform, color=ColorMode.GREY)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    show = False

    model = get_model_orig()
    times = []
    for i, img in enumerate(loader):
        img = img.to(device).permute(0, 3, 1, 2) / 255.0
        start = time.time()
        result = model(img, 0.041)
        end = time.time()
        if show:
            pts_2, desc_2 = result
            img_2 = ((torch.stack([img[0, 0]] * 3)) * 255).permute(1,2,0).numpy().astype(np.uint8).copy()
            show_points(img_2, pts_2[:, 0:2].numpy().astype(np.int32), 'rot', scale=2)
        times.append(end - start)
        if i % 10 == 0:
            print(i)
        if i == 100:
            break
    print('mean time %f', numpy.mean(times))

if __name__ == '__main__':
    main()
