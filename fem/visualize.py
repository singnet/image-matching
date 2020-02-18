import cv2
import imageio
import torch
import numpy
from fem.noise import Resize
from fem.wrapper import SuperPoint
from fem.goodpoint import GoodPoint
from fem.nonmaximum import MagicNMS
import matplotlib.pyplot as plt
import drawing


def show_heat(model, img):
    timg1 = numpy.expand_dims(numpy.expand_dims(img.astype('float32'), axis=0), axis=0)
    heat = model.heatmap(torch.from_numpy(timg1).to(next(model.parameters())))
    heat = heat.squeeze().cpu().detach().numpy() / heat.max().item()
    plt.imshow(heat)
    plt.show()


def show_points(model, img, conf_thresh, name):
    timg1 = numpy.expand_dims(numpy.expand_dims(img.astype('float32'), axis=0), axis=0)
    pts_1, desc_1_ = model.points_desc(torch.from_numpy(timg1).to(next(model.parameters())), threshold=conf_thresh)
    print(len(pts_1))
    drawing.show_points(img, pts_1[:,:2].round().astype(numpy.int32), name, 2.0, save_path=name + '.png')



cube = '/mnt/fileserver/shared/datasets/at-on-at-data/COCO/val2014/COCO_val2014_000000000073.jpg'
cube = '/home/noskill/COCO_val2014_000000032952.jpg'

def show():
    device = 'cuda'
    sp_path = '/home/noskill/projects/neuro-fem/fem/superpoint_magicleap/superpoint_v1.pth'
    gp_path = './snapshots/super12300.pt'
    resize = Resize((282, 320))
    img = imageio.imread(cube, pilmode='L')
    cv2.imshow('img1', img)
    img = resize(img).squeeze()

    sp = SuperPoint(MagicNMS()).to(device)
    sp.load_state_dict(torch.load(sp_path))
    show_points(sp, img, 0.059, 'super')

    gp = GoodPoint(grid_size=8,
                   nms=MagicNMS(), batchnorm=True).to(device).eval()
    gp.load_state_dict(torch.load(gp_path)['superpoint'])
    show_points(gp, img, 0.035, 'good')


show()