import sys

from collections import defaultdict
import ignite
from packaging import version

import torch.optim as optim
from fem.reinf_utils import threshold_nms, threshold_nms_dense
from torch.utils.data import DataLoader

from fem.dataset import SynteticShapes, Mode, ColorMode, ImageDirectoryDataset
from fem.transform import *
from fem import util

from ignite.engine import Engine
from fem.goodpoint import GoodPoint
from fem.goodpoint_small import GoodPointSmall

import numpy

import kornia.filters
from fem.loss_reinf import compute_loss_hom_det
from fem.noise_transformer import NoisySimpleTransformWithResize, NoisyTransformWithResize

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
    util.add_moving_average(engine, 'desc_loss', decay=0.95)
    util.add_moving_average(engine, 'det_loss1', decay=0.95)
    util.add_moving_average(engine, 'points_s', decay=0.95)
    util.add_moving_average(engine, 'det_loss2', decay=0.95)
    util.add_moving_average(engine, 'det_loss', decay=0.95)
    util.add_moving_average(engine, 'points_t', decay=0.95)
    util.add_moving_average(engine, 'diff', decay=0.95)
    util.add_moving_average(engine, 'quality_desc', decay=0.95)
    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, lambda eng: util.on_iteration_completed(eng,
                                                                                                          print_every=print_every))
    return engine



def train_points_reinf(batch, model, optimizer, scheduler, device, function, count=[0], **kwargs):
    model_source, model_target = model
    model_target.train()

    for k, v in batch.items():
        batch[k] = v.to(device)
    for i in range(1):
        optimizer.zero_grad()
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
                    'step': count[0],
                    'superpoint': model_target.state_dict()}, 'distilled{0}.pt'.format(count[0]))
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


def train_distill(batch, models, **kwargs):
    model_source, model_target = models
    model_source.eval()
    model_target.train()
    choice = random.choice([1,2])
    imgs = batch['img' + str(choice)].squeeze()
    mask = batch['mask' + str(choice)]
    if len(imgs.shape) == 4:
        imgs = imgs.flatten(start_dim=0, end_dim=1)

    height, width = imgs.shape[1], imgs.shape[2]
    assert len(imgs.shape) == 3

    with torch.no_grad():
        heatmaps, desc = model_source(imgs.unsqueeze(1) / 255.0)
    heatmaps_t, desc_t = model_target(imgs.unsqueeze(1) / 255.0)
    keypoints_prob = model_source.expand_results(model_source.depth_to_space, heatmaps)
    keypoints_prob_t = model_source.expand_results(model_target.depth_to_space, heatmaps_t)
    keypoints_prob = keypoints_prob * mask
    keypoints_prob_t = keypoints_prob_t * mask
    point_mask32 = threshold_nms(keypoints_prob, pool=32, take=None)
    point_mask16 = threshold_nms(keypoints_prob, pool=16, take=None)
    # import drawing
    # drawing.show_points(batch['img1'][0].cpu().numpy() / 255.0, point_mask32[0].nonzero(), 'img1')
    # drawing.show_points(batch['img2'][0].cpu().numpy() / 255.0, point_mask16[0].nonzero(), 'img2')
    heatmaps_source = keypoints_prob[:, 0]
    heatmaps_target = keypoints_prob_t[:, 0]
    flat_t = heatmaps_target.flatten(-2)
    flat = heatmaps_source.flatten(-2)

    # sort along last dimention
    sorted_flat, idx = torch.sort(flat)
    sorted_flat_t = flat_t.gather(dim=-1, index=idx)
    # use only top 50 of values, all other don't matter that much
    source_values = sorted_flat[:, -50:]
    target_values = sorted_flat_t[:, -50:]
    eps = 0.00001
    # different kinds of cross-entropy
    # cross_entropy = - (source_values * torch.log(target_values + eps) + (1 - source_values) * torch.log(1 - target_values + eps))
    # label = (heatmaps_source > 0.02)
    #cross_entropy = - label * torch.log(heatmaps_target + eps) - (1 - label) * torch.log(1 - heatmaps_target + eps)
    det_loss1 = ((heatmaps_source * 120 - heatmaps_target * 120) ** 2).mean()
    spatial_s = kornia.filters.spatial_gradient(heatmaps_source.unsqueeze(1))
    spatial_t = kornia.filters.spatial_gradient(heatmaps_target.unsqueeze(1))
    det_loss2 = ((spatial_s * 150 - spatial_t * 150) ** 2).mean()
    det_loss = det_loss2 + det_loss1
    desc_loss = ((desc * 10 - desc_t * 10) ** 2).mean()
    points_t = (heatmaps_target > 0.021).sum() / len(heatmaps_target)
    points_s = (heatmaps_source > 0.021).sum() / len(heatmaps_source)

    result = dict()
    result['loss'] = det_loss + desc_loss
    result['det_loss'] = det_loss
    result['points_t'] = points_t
    result['points_s'] = points_s
    result['det_loss1'] = det_loss1
    result['det_loss2'] = det_loss2
    result['desc_loss'] = desc_loss.detach()
    result['max_source'] = source_values.max()
    result['max_target'] = target_values.max()
    result['max_mean_source'] = source_values.mean(dim=1).max()
    result['max_mean_target'] = target_values.mean(dim=1).max()
    return result


def train_good():
    num_sample = 1
    batch_size = 24 // num_sample
    assert 24 % num_sample == 0
    test_batch_size = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('using device {0}'.format(device))
    lr = 0.0015
    epochs = 5
    weight_decay = 0.0005
    parallel = False

    super_file = "./snapshots/super3400.pt"

    sp = GoodPoint(activation=torch.nn.LeakyReLU(), grid_size=8,
               batchnorm=True, dustbin=0).to(device)
    state_dict = torch.load(super_file, map_location=device)
    print("loading weights from {0}".format(super_file))
    sp.load_state_dict(state_dict['superpoint'], strict=True)
    distilled = GoodPointSmall(activation=torch.nn.LeakyReLU(), grid_size=8,
               batchnorm=True, dustbin=0,
               base1=32).to(device)
               #base1=32, base2=32, base3=64).to(device)
    distilled.load_state_dict(torch.load('snapshots/distilled3400.pt', map_location=device)['superpoint'])

    optimizer = optim.AdamW(distilled.parameters(), lr=lr)
    state_dict = torch.load('snapshots/distilled3400.pt', map_location=device)['optimizer']
    state_dict['param_groups'][0]['lr'] = lr
    state_dict['param_groups'][0]['initial_lr'] = lr
    optimizer.load_state_dict(state_dict)

    decay_rate = 0.98
    decay_steps = 1000

    if parallel:
        sp = torch.nn.DataParallel(sp)

    def l(step, my_step=[1]):
        my_step[0] += 1
        # if my_step[0] < 100:
        #     return my_step[0] / 100
        return exponential_lr(decay_rate, my_step[0], decay_steps, staircase=False)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l)

    test_loader, coco_loader, fundus_loader = get_loaders(batch_size, test_batch_size,  num_sample)
    test_engine = Engine(lambda eng, batch: test(eng, batch, (sp, distilled), device, loss_function=test_callback))
    util.add_metrics(test_engine, average=True)

    _3d_engine = setup_engine(Engine, train_points_reinf,
                              (sp, distilled), device, optimizer, scheduler, function=train_distill,
                            print_every=10,
                            desc_model=None)


    train_loop(_3d_engine,
               None,
               coco_loader, test_loader,
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

    test_loader = DataLoader(coco_dataset, batch_size=test_batch_size)
    # synth_dataset._size = 10000
    # assert(len(synth_dataset) == 10000)
    # coco_dataset._size = 10000

    coco_loader = DataLoader(coco_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    fundus_loader = DataLoader(fundus_dataset,
                               batch_size=batch_size,
                               shuffle=True)

    return test_loader, coco_loader, fundus_loader


if __name__ == '__main__':
    train_good()
