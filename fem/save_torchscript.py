import time
import gzip

import numpy as np
import torch
from torch.utils.data import DataLoader
from airsim_dataset import AirsimIntVarDataset
from fem.goodpoint import GoodPoint
import os
from fem.nonmaximum import MagicNMS
from fem.util import remove_module
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

    device = 'cpu'
    device = 'cuda'
    nms = PoolingNms(8)
    weight = "./snapshots/super3400.pt"
    weight = './tu_model.pt'
    weight = './super37630.pt'
    pt_path = 'tu_model.gpu.pt'
    if device == 'cpu':
        pt_path = 'tu_model.cpu.pt'


    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    sp.load_state_dict(remove_module(torch.load(weight, map_location=device)['superpoint']))
    sp.to(device)

    thresh = 0.021
    for i, batch in enumerate(coco_loader):
        batch = batch.permute(0, 3, 1, 2)
        with torch.no_grad():
            #import pdb;pdb.set_trace()
            pts_2, desc_2 = sp.points_desc(batch.to(device), threshold=thresh)
            module = torch.jit.trace(sp, (batch.to(device) / 255.0, torch.as_tensor(thresh)))
            torch.jit.save(module, pt_path)
            print('saved to ', pt_path)
            break

main()
