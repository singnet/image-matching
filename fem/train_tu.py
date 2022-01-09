import sys
import os

from collections import defaultdict
import ignite
from packaging import version

import torch.optim as optim
from fem.reinf_utils import threshold_nms, threshold_nms_dense
from torch.utils.data import DataLoader

from fem.dataset import SynteticShapes, Mode, ColorMode, ImageDirectoryDataset, Multidirectory
from fem.transform import *
from fem import util

from ignite.engine import Engine
from fem.goodpoint import GoodPoint

import numpy

from fem.loss_reinf import compute_loss_hom_det
from fem.noise_transformer import NoisyTransformWithResize

gtav_super = '/mnt/fileserver/shared/datasets/from_games/GTAV_super/01_images'
coco_super = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/'
coco_images = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/COCO_super/train2014/images/training/'
synth_path = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/mysynth/'

dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light'
dir_night = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_night_light'
poses_file = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/village_00.json'


from fem.training import train_loop, test, exponential_lr


def setup_engine(Engine, train, model, device, optimizer, scheduler,
                 print_every=100,
                 **kwargs):
    engine = Engine(lambda eng, batch: train(batch, model=model, device=device,
                                            optimizer=optimizer, scheduler=scheduler,
                                             **kwargs))
    util.add_moving_average(engine, 'loss', decay=0.95)
    util.add_moving_average(engine, 'loss_desc', decay=0.95)
    util.add_moving_average(engine, 'expected_reward', decay=0.95)
    util.add_moving_average(engine, 'points', decay=0.95)
    util.add_moving_average(engine, 'loss_points', decay=0.95)
    util.add_moving_average(engine, 'loss_non_points', decay=0.95)
    util.add_moving_average(engine, 'quality', decay=0.95)
    util.add_moving_average(engine, 'diff', decay=0.95)
    util.add_moving_average(engine, 'quality_desc', decay=0.95)
    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, lambda eng: util.on_iteration_completed(eng,
                                                                                                          print_every=print_every))
    return engine



def train_points_reinf(batch, model, optimizer, scheduler, device, function, count=[0], **kwargs):
    model.train()
    optimizer.zero_grad()

    for k, v in batch.items():
        batch[k] = v.to(device)

    out = function(batch, model, **kwargs)
    if count[0] % 10 == 0:
        print("out: {0}".format({k:float(v) for k,v in out.items()}))
        # (out['loss_points']).backward(retain_graph=True)
        # print("out['loss_points']: " + str(out['loss_points']))
        # print("points: " + str(out['points']))
        # print(model.convPa.weight.grad.max())
        # print(model.convPa.bias.grad.max())
    loss = out['loss']
    loss.backward()
    # loss.backward()
    optimizer.step()
    scheduler.step()
    result = {k: v.cpu().detach().numpy() for (k,v) in out.items()}
    if version.parse(torch.__version__) < version.parse('1.4.0'):
        result['lr'] = scheduler.get_lr()
    else:
        result['lr'] = scheduler.get_last_lr()
    sys.stdout.flush()

    count[0] += 1
    if count[0] and count[0] % 100 == 0:
        torch.save({'optimizer': optimizer.state_dict(),
                    'superpoint': model.state_dict()}, 'super{0}.pt'.format(count[0]))
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


def draw_m(img1, img2, matches, points, points2):
    from super_debug import draw_matches
    img_pair = numpy.hstack([img1, img2]).astype(numpy.uint8)
    img_pair = numpy.stack([img_pair, img_pair, img_pair], axis=2).squeeze()
    draw_matches(0, img_pair, matches, util.swap_rows(points.cpu().numpy().T),
                 util.swap_rows(points2.cpu().numpy().T))


def not_none(*args):
    return [x for x in args if x is not None]


def mean(lst):
    if len(lst) == 0:
        return 0
    return torch.mean(torch.stack(lst))


def train_maxpool_by_pairs(batch, model, **kwargs):
    imgs = torch.cat([batch['img1'], batch['img2']]).float()
    height, width = imgs.shape[1], imgs.shape[2]
    assert len(imgs.shape) == 3

    desc_model = kwargs.get('desc_model', None)
    if desc_model:
        desc_model = desc_model.eval()
        _, desc = desc_model.semi_forward(imgs.unsqueeze(1) / 255.0)
        del _
        desc = desc.detach()

    heatmaps, desc_back = model(imgs.unsqueeze(1) / 255.0)
    if not desc_model:
        desc = desc_back
    else:
        del desc_back

    keypoints_prob = model.module.expand_results(model.module.depth_to_space, heatmaps)
    point_mask32 = threshold_nms(keypoints_prob, pool=32, take=None)
    point_mask16 = threshold_nms(keypoints_prob, pool=16, take=None)
    # drawing.show_points(batch['img1'][0].cpu().numpy() / 255.0, point_mask32[0].nonzero(), 'img1')
    # drawing.show_points(batch['img2'][0].cpu().numpy() / 255.0, point_mask16[20].nonzero(), 'img2')

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
        tmp = compute_loss_hom_det(heatmap1[i], heatmap2[i].clone(),
                {k: v[i] for (k,v) in _3d_data.items()},
                point_mask1[i], point_mask2[i], desc1[i], desc2[i],
                heat1_fold=heat_fold1[i],
                heat2_fold=heat_fold2[i])
        for key, value in tmp.items():
            tmp_results[key].append(value)
    result = {key: mean(value) for (key, value) in tmp_results.items()}
    result['max_max'] = heatmaps[:,0:].max()
    result['max_mean'] = heatmaps[:,0:].mean(dim=(0,2,3)).max()
    return result


def train_good():
    batch_size = 22
    test_batch_size = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device {0}'.format(device))
    lr = 0.005
    epochs = 3
    weight_decay = 0.0005
    parallel = True
    aggregate = False
    if aggregate:
        batch_size = 1

    super_file = "snapshots/super3400.pt"

    state_dict = torch.load(super_file, map_location=device)
    sp = GoodPoint(activation=torch.nn.LeakyReLU(), grid_size=8,
               batchnorm=True, dustbin=0).to(device)



    print("loading weights from {0}".format(super_file))


    print("loading optimizer")
    optimizer = optim.AdamW(sp.parameters(), lr=lr)
    optimizer.load_state_dict(state_dict['optimizer'])

    decay_rate = 0.9
    decay_steps = 1000

    sp.load_state_dict(state_dict['superpoint'], strict=True)
    if parallel:
        sp = torch.nn.DataParallel(sp)

    def l(step, my_step=[3700 * 4]):
        my_step[0] += 1
        # if my_step[0] < 100:
        #     return my_step[0] / 100
        return exponential_lr(decay_rate, my_step[0], decay_steps, staircase=False)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

    test_loader, tu_loader = get_loaders(batch_size, test_batch_size, 30 if aggregate else 1)
    test_engine = Engine(lambda eng, batch: test(eng, batch, sp, device, loss_function=test_callback))
    util.add_metrics(test_engine, average=True)

    _3d_engine = setup_engine(Engine, train_points_reinf,
                              sp, device, optimizer, scheduler, function=train_maxpool_by_pairs,
                            print_every=10,
                            desc_model=None)


    train_loop(_3d_engine,
               None,
               tu_loader, test_loader,
               epochs, optimizer, sp)

    torch.save({'optimizer': optimizer.state_dict(),
                'superpoint': sp.state_dict()}, 'super{0}.pt'.format(scheduler.last_epoch))


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


def get_loaders(batch_size, test_batch_size, num):
    fire = '/mnt/fileserver/shared/datasets/fundus/FIRE/Images/'
    from fundus import FundusTransformWithResize

    fundus_dataset = ImageDirectoryDataset(fire,
                                           transform=FundusTransformWithResize(num),
                                           color=ColorMode.GREY)
    fundus_dataset.points = fundus_dataset.points * 50
    fundus_dataset._size = len(fundus_dataset.points)
    coco_dataset = SynteticShapes(coco_super, Mode.training,
                                  transform=NoisyTransformWithResize(num),
                                  color=ColorMode.GREY)
    coco_dataset.shuffle()
    base_tu = os.path.expandvars("$pathDatasetTUM_VI")
    tu_dirs = ('dataset-outdoors4_512_16', 'dataset-room1_512_16', 'dataset-room3_512_16', 'dataset-room6_512_16',
'dataset-outdoors8_512_16', 'dataset-room2_512_16', 'dataset-room4_512_16')

    img_path = 'dso/cam1/images/'
    img_path_test = 'dso/cam0/images/'
    tu_dataset = Multidirectory([os.path.join(base_tu, x, img_path_test) for x in tu_dirs],
        transform=NoisyTransformWithResize(num), color=ColorMode.GREY)

    tu_dataset_test = Multidirectory([os.path.join(base_tu, x, img_path) for x in tu_dirs],
        transform=NoisyTransformWithResize(num), color=ColorMode.GREY)


    print('total number of tu images {0}'.format(len(tu_dataset)))
    tu_loader = DataLoader(tu_dataset, batch_size=test_batch_size, shuffle=True)

    test_loader = DataLoader(tu_dataset_test, batch_size=test_batch_size)

    return test_loader, tu_loader


if __name__ == '__main__':
    train_good()
