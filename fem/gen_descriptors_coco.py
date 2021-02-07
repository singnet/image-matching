import time
import gzip

import numpy as np
import torch
from torch.utils.data import DataLoader
from airsim_dataset import AirsimIntVarDataset
from fem.goodpoint import GoodPoint
import os
from fem.nonmaximum import MagicNMS
from fem.nonmaximum import PoolingNms, MagicNMS
from fem.dataset import SynteticShapes, Mode, ColorMode, ImageDirectoryDataset


coco_super = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/train2014/images/training/'


def main():
    coco_dataset = ImageDirectoryDataset(coco_super,
                                    color=ColorMode.GREY)

    batch_size = 1
    coco_loader = DataLoader(coco_dataset,
                         batch_size=batch_size,
                         shuffle=True)

    device = 'cuda'
    nms = PoolingNms(8)
    weight = "./snapshots/super3400.pt"
    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    sp.to(device)

    thresh = 0.021
    result_txt = gzip.open('desc.txt.gz', 'wb')
    for i, batch in enumerate(coco_loader):
        batch = batch.permute(0, 3, 1, 2)
        with torch.no_grad():
            pts_2, desc_2 = sp.points_desc(batch.to(device), threshold=thresh)
            desc = desc_2[0]
            for row in desc:
                line = ' '.join(str(x) for x in row.cpu().numpy())
                result_txt.write(line.encode('ascii'))
                result_txt.write('\n'.encode('ascii'))
            result_txt.write('\n'.encode('ascii'))
        if i % 100 == 0:
            print('iteration ', i)

main()
