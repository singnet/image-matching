import os
import torch
import numpy
import imageio

from fem.goodpoint import GoodPoint


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))
    weights = './snapshots/mine8.pt'
    gp = GoodPoint(n_channels=3,
           activation=torch.nn.LeakyReLU(),
           grid_size=8,
           batchnorm=False,
           dustbin=0,
           desc_out=8).to(device)

    if os.path.exists(weights):
        state_dict = torch.load(weights, map_location=device)
        print("loading weights from {0}".format(weights))
        gp.load_state_dict(state_dict['superpoint'])

    img_path = '/mnt/fileserver/shared/datasets/minecraft-segm/img387.png'
    img = imageio.imread(img_path)
    points = numpy.asarray([[10, 20],
                          [40, 40]])
    descriptors = gp.get_descriptors(img, points)


if __name__ == '__main__':
    main()

