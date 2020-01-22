import sys

from collections import defaultdict
import ignite
import torch

import torch.optim as optim
from fem.reinf_utils import threshold_nms, sample
from torch.utils.data import DataLoader

from fem.dataset import SynteticShapes, Mode, ColorMode
from fem.transform import *
from fem import util

from fem.hom import HomographySamplerTransformer
from fem.glue import Glue

from ignite.engine import Engine
from fem.goodpoint import GoodPoint
from fem.loss_match import compute_loss_matcher
from fem.stats import Stats

import numpy




gtav_super = '/mnt/fileserver/shared/datasets/from_games/GTAV_super/01_images'
coco_super = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/'
coco_images = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/train2014/images/training/'
synth_path = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/mysynth/'

dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light'
dir_night = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_night_light'
poses_file = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/village_00.json'


from fem.training import train_loop, TransformWithResize, TransformWithResizeThreshold, \
    make_noisy_transformers, train, test, exponential_lr


def train(batch, model, optimizer, scheduler, device, function, stats, count=[0], **kwargs):
    model.eval()
    glue = kwargs.get('glue')
    glue.train()
    optimizer.zero_grad()

    for k, v in batch.items():
        batch[k] = v.to(device)

    out = function(batch, model, **kwargs)
    stats.update(**{k:float(v) for k,v in out.items()})
    if count[0] % 10 == 0:
        print("stats: {0}".format({k:float(v) for k,v in stats.stats.items()}))

    loss = out['loss']
    loss.backward()
    optimizer.step()
    scheduler.step()
    result = {k: v.cpu().detach().numpy() for (k,v) in out.items()}
    result['lr'] = scheduler.get_lr()
    sys.stdout.flush()

    count[0] += 1
    if count[0] and count[0] % 100 == 0:
        torch.save({'optimizer': optimizer.state_dict(),
                    'glue': glue.state_dict()}, 'glue{0}.pt'.format(count[0]))
    return result


def swap3d(_3d_data):
    K1 = _3d_data['K1']
    K2 = _3d_data['K2']
    depth1 = _3d_data['depth1']
    depth2 = _3d_data['depth2']
    pose1 = _3d_data['pose1']
    pose2 = _3d_data['pose2']
    return dict(K2=K1,
                K1=K2,
                depth2=depth1,
                depth1=depth2,
                pose2=pose1,
                pose1=pose2)


def not_none(*args):
    return [x for x in args if x is not None]


def mean(lst):
    if len(lst) == 0:
        return 0
    return torch.mean(torch.stack(lst))


def train_glue(batch, model, **kwargs):
    glue = kwargs.get('glue')
    imgs = torch.cat([batch['img1'], batch['img2']]).float()
    height, width = imgs.shape[1], imgs.shape[2]
    assert len(imgs.shape) == 3

    desc_model = kwargs.get('desc_model', None)
    heatmaps, desc_back = model.semi_forward(imgs.unsqueeze(1) / 255.0)
    if not desc_model:
        desc = desc_back
    else:
        del desc_back

    keypoints_prob = model.expand_results(model.depth_to_space, heatmaps)
    point_mask32 = threshold_nms(keypoints_prob, pool=32, take=None)
    point_mask16 = threshold_nms(keypoints_prob, pool=32, take=None)

    desc1 = desc[:len(heatmaps) // 2]
    desc2 = desc[len(heatmaps) // 2:]
    point_mask1 = point_mask32[:len(heatmaps) // 2]
    point_mask2 = point_mask16[len(heatmaps) // 2:]

    heatmap1 = keypoints_prob[:len(heatmaps) // 2][:,0]
    heatmap2 = keypoints_prob[len(heatmaps) // 2:][:,0]

    heat_fold1 = heatmaps[:len(heatmaps) // 2]
    heat_fold2 = heatmaps[len(heatmaps) // 2:]
    _3d_data = batch
    tmp_results = defaultdict(list)
    for i in range(len(heatmap1)):
        tmp = compute_loss_matcher(heatmap1[i], heatmap2[i].clone(),
                {k: v[i] for (k,v) in _3d_data.items()},
                point_mask1[i], point_mask2[i], desc1[i], desc2[i], glue)
        for key, value in tmp.items():
            tmp_results[key].append(value)
    result = {key: mean(value) for (key, value) in tmp_results.items()}
    result['max_max'] = heatmaps[:,0:].max()
    result['max_mean'] = heatmaps[:,0:].mean(dim=(0,2,3)).max()
    return result




def train_super():
    batch_size = 20
    test_batch_size = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))
    lr = 0.005
    epochs = 5
    weight_decay = 0.001
    aggregate = False
    if aggregate:
        batch_size = 1


    super_file = "./snapshots/super12300.pt"


    glue = Glue().to(device)
    state_dict = torch.load(super_file, map_location=device)
    sp = GoodPoint(activation=torch.nn.ReLU(), grid_size=8,
               batchnorm=True, dustbin=0).to(device)
    # detector_layer_names = ('convPa', 'convPb', 'batchnormPa', 'batchnormPb')
    print("loading weights from {0}".format(super_file))
    # to_pop = [k for k in state_dict['superpoint'] if k.startswith(detector_layer_names)]
    # print("reset: {0}".format(to_pop))
    # for p in to_pop:
    #    state_dict['superpoint'].pop(p)
    sp.load_state_dict(state_dict['superpoint'], strict=True)
    print("loading optimizer")
    optimizer = optim.RMSprop(glue.parameters(), lr=lr, weight_decay=weight_decay)
    state_dict['optimizer']['param_groups'][0]['lr'] = lr
    state_dict['optimizer']['param_groups'][0]['initial_lr'] = lr



    decay_rate = 0.9
    decay_steps = 1000

    def l(step, my_step=[0]):
        my_step[0] += 1
        # if my_step[0] < 100:
        #     return my_step[0] / 100
        return exponential_lr(decay_rate, my_step[0], decay_steps, staircase=False)

    l = lambda step: 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

    train_loader, test_loader, coco_loader, synth = get_loaders(batch_size, test_batch_size, 30 if aggregate else 1)
    test_engine = Engine(lambda eng, batch: test(eng, batch, sp, device, loss_function=test_callback))
    util.add_metrics(test_engine, average=True)

    stats = Stats()
    for epoch in range(epochs):
        for batch in coco_loader:
            train(batch, sp, optimizer, scheduler, device, train_glue, stats, glue=glue)

    torch.save({'optimizer': optimizer.state_dict(),
                'glue': sp.state_dict()}, 'glue{0}.pt'.format(scheduler.last_epoch))


class NmsImgTransform:

    def __call__(self, sample):
        sample['points1'] = self.process(sample['points1'])
        # import drawing
        # pts = (torch.from_numpy(sample['points1']) > 0.2).nonzero()
        # drawing.show_points(sample['img1'],pts, 'img1', 2)
        sample['points2'] = self.process(sample['points2'])
        return sample

    def process(self, heatmap):
        h, w = heatmap.shape
        # target_nms = self.nms(torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)).numpy().squeeze()
        # assert target_nms.shape == (h, w)
        # target_thres = self.target_threshold(target_nms).astype(numpy.float32)
        # in case data doesn't have non-point layer
        target = numpy.stack([1 - heatmap, heatmap])
        return target


def collate(list_of_dicts):
    keys = list_of_dicts[0].keys()
    result = {key: torch.stack([item[key] for item in list_of_dicts]) for key in keys}
    return result


def to_torch(kwargs):
    return {k: (torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v) for (k, v) in kwargs.items()}


class NoisyTransformWithResize(TransformCompose):
    def __init__(self, num=1):
        from fem import noise
        from fem.training import make_noisy_transformers
        self.noisy = make_noisy_transformers()

        self.imgcrop = noise.RandomCropTransform(size=256, beta=0)
        self.resize = noise.Resize((256, 256))
        self.to_tensor = ToTensor()
        self.homography = HomographySamplerTransformer(num=1,
                                                  beta=14,
                                                  theta=0.08,
                                                  random_scale_range=(0.8, 1.3),
                                                  perspective=65)
        self.num = num

    def __call__(self, data=None, target=None):
        # crop
        image, pos = self.imgcrop(data, return_pos=True)
        resized = self.resize(image)
        img = self.to_tensor(resized)
        tmp = []
        if self.num == 1:
            return self.sample(img)
        for i in range(self.num):
            tmp.append(to_torch(self.sample(img)))
        result = collate(tmp)
        result['img1'] = result['img1'][0]
        return result

    def sample(self, x):
        # x is batch 1, h, w
        self.homography.sample_fixed_homography(h=x.shape[-2], w=x.shape[-1])
        # apply noise to source image before sample
        to_sample = self.noisy(x.permute(1, 2, 0).cpu().numpy()).permute(2, 0, 1)
        template, hom, mask = self.homography(to_sample.permute(1, 2, 0))
        # use different noise for training
        source = self.noisy(x.permute(1, 2, 0).cpu().numpy())
        #import pdb;pdb.set_trace()
        #import cv2
        #cv2.imshow('template[1]', template[1] / 256. * mask[0][..., numpy.newaxis])
        #cv2.imshow('source', (source / 256.).numpy())
        #cv2.waitKey(100)
        return dict(img1=source.squeeze(), img2=torch.from_numpy(template[1].squeeze()),
                    H=torch.from_numpy(hom[0][0:3]), H_inv=hom[0][3:6], mask=mask)



def get_loaders(batch_size, test_batch_size, num):
    from fem.airsim_dataset import AirsimWithTarget
    frame_offset=5
    train_dataset = AirsimWithTarget(dir_day, dir_night, poses_file, frame_offset=frame_offset,
                                     transform=NmsImgTransform())

    coco_dataset = SynteticShapes(coco_super, Mode.training,
                              transform=NoisyTransformWithResize(num),
                              color=ColorMode.GREY)
    synth_dataset = SynteticShapes(synth_path, Mode.training,
                              transform=NoisyTransformWithResize(num),
                                    color=ColorMode.GREY)
    coco_dataset.shuffle()
    train_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=False)


    test_loader = DataLoader(coco_dataset, batch_size=test_batch_size)
    # synth_dataset._size = 10000
    # assert(len(synth_dataset) == 10000)
    # coco_dataset._size = 10000
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size,
                              shuffle=True)

    coco_loader = DataLoader(coco_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader, coco_loader, synth_loader


if __name__ == '__main__':
    train_super()
