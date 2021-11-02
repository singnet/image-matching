import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from airsim_dataset import AirsimIntVarDataset
from fem.goodpoint import GoodPoint
import os
from fem.nonmaximum import MagicNMS
from fem.eval_fem_airsim import loop
from fem.nonmaximum import PoolingNms, MagicNMS


PATH_SAVE = "/tmp/village_00_320x240_day_night_SP_fem_my"

village = dict(dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light',
               dir_night ='/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_night_light',
               poses_file ='/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/village_00.json')

fantasy_village = dict(dir_day = '/mnt/fileserver/shared/datasets/AirSim/fantasy_village_362x362/00_day_light/',
               dir_night ='/mnt/fileserver/shared/datasets/AirSim/fantasy_village_362x362/00_day_light_fog/',
               poses_file ='/mnt/fileserver/shared/datasets/AirSim/fantasy_village_362x362/00_day_light_fog/fantasy_village_00.json')


PATH_SAVE_PTS = PATH_SAVE + '/data/'

magicleap_file = "superpoint_magicleap/superpoint_v1.pth"
magicleap_file = None

PATH_WEIGHTS = None


PATH_WEIGHTS = "snapshots/super16000.pt"
PATH_WEIGHTS = "./snapshots/super12300.pt"
PATH_WEIGHTS = "./super6900.pt"


#PATH_WEIGHTS = "./snapshots/TWD/tr_0.5_dr_0.4/from_scratch_super17.pt"
#PATH_WEIGHTS = "./snapshots/TWD/tr_0.5_dr_0.4/super8.pt"


batchnorm=True
IMG_SIZE = [240, 320]

frame_offset = 5



batch_size = 1


conf_thresh= 0.020885
conf_thresh= 0.0455591090510123100629


def run_all_snapshots():
    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.ReLU(),
                   batchnorm=batchnorm,
                   grid_size=8,
                   nms=nms).eval()
    best_f1 = 0.0
    best_path = None
    for f in os.listdir('.'):
        if f.endswith('.pt'):
            current = loop(sp, weights=f)
            if best_f1 < current:
                print('new best: {0}, f1: {1} '.format(f, current))
                best_f1 = current
                best_path = f
    print(best_path)


def test_magicleap(loader, angle=0.0):

    from fem.wrapper import SuperPoint
    sp_path = '/home/noskill/projects/neuro-fem/fem/superpoint_magicleap/superpoint_v1.pth'

    sp = SuperPoint(nms).to(device)
    sp.load_state_dict(torch.load(sp_path))
    loop(sp, loader, thresh=0.015, print_res=False, draw=True, rotation_angle=angle)
    print('test superpoint completed')

def test_magicleap1(loader, angle=0.0):
    sp_path = '/home/noskill/projects/neuro-fem/fem/superpoint_magicleap/superpoint_v1.pth'

    from superpoint_magicleap.demo_superpoint import PointTracker, SuperPointFrontend
    fe = SuperPointFrontend(weights_path=sp_path,
                        nms_dist=8,
                        conf_thresh=0.015,
                        nn_thresh=0.8,
                        cuda=True)
    loop(loader=loader, sp=None, fe=fe, thresh=0.015, print_res=True, draw=False, rotation_angle=angle)


def test_distilled(loader):
    from superpoint_05 import SuperPointNet
    sp = SuperPointNet().eval()
    path = '/home/noskill/projects/neuro-fem/fem/airsim_realsense_gpnt_model_last.pth'
    sp.load_state_dict(torch.load(path, map_location=device))
    sp.nms = MagicNMS()
    loop(sp, loader, thresh=0.015, print_res=False, draw=False)
    print('test distilled completed')


def run_distilled(loader, angle=0.0):
    weight = "./snapshots/distilled3400.pt"
    weight = "./distilled13800.pt"
    from goodpoint_small import GoodPointSmall
    sp = GoodPointSmall(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms,
                   base1=32, base2=32, base3=64).eval()


    #sp_desc = GoodPoint(dustbin=0,
    #               activation=torch.nn.LeakyReLU(),
    #               batchnorm=True,
    #               grid_size=8,
    #               nms=nms).eval().cuda()

    #sp_desc.load_state_dict(torch.load('snapshots/super6300.pt', map_location=device)['superpoint'])
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    # sp.to(torch.bfloat16)
    # just in case
    torch.set_flush_denormal(True)
    loop(sp=sp, loader=loader, draw=False, print_res=False, thresh=0.0217075525,
            device=device, desc_model=None, rotation_angle=angle, N=100)
    print('test destilled {0} completed'.format(weight))


def run_good(loader, angle=0.0):
    weight = './snapshots/orbnet.d1.pt'
    weight = "./snapshots/super3400.pt"

    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()


    #sp_desc = GoodPoint(dustbin=0,
    #               activation=torch.nn.LeakyReLU(),
    #               batchnorm=True,
    #               grid_size=8,
    #               nms=nms).eval().cuda()

    #sp_desc.load_state_dict(torch.load('snapshots/super6300.pt', map_location=device)['superpoint'])

    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    loop(sp=sp, loader=loader, draw=False, print_res=False, thresh=0.021075525,
            desc_model=None, rotation_angle=angle, N=100)
    print('test goodpoint {0} completed'.format(weight))


def measure_performance(loader):
    device = 'cpu'
    weight = './snapshots/super3400.pt'
    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    ipow = [i for i in range(11)]
    ipow = [x * -1 for x in reversed(ipow)][:-1] + ipow
    for t_pow in ipow:
        weights = torch.load(weight, map_location=device)['superpoint']
        for key, wt in list(weights.items()):
            weights[key] = wt * (2 ** t_pow)
        sp.load_state_dict(weights)
        sp = sp.to(device)
        perf = emtpy_loop(sp, loader, device, thresh=0.021075525)
        print('pow {0} fps {1}'.format(t_pow, perf))


def emtpy_loop(sp, loader, device, thresh):
    total = 0.0
    for i_batch, sample in enumerate(loader):
        img_1_batch = sample['img1'].numpy()
        img_2_batch = sample['img2'].numpy()
        img_1 = img_1_batch[0, :, :]
        img_2 = img_2_batch[0, :, :]
        timg1 = np.expand_dims(np.expand_dims(img_1.astype('float32'), axis=0), axis=0)
        timg2 = np.expand_dims(np.expand_dims(img_2.astype('float32'), axis=0), axis=0)
        timg1 = torch.from_numpy(timg1).to(device)
        timg2 = torch.from_numpy(timg2).to(device)

        start = time.time()
        with torch.no_grad():
            pts_2, desc_2 = sp.points_desc(timg1, threshold=thresh)
            pts_2, desc_2 = sp.points_desc(timg2, threshold=thresh)

        end = time.time()
        total += (end - start)

        if i_batch == 100:
            break


    perf = ((i_batch + 1) * 2) / total
    return perf


dataset_village = AirsimIntVarDataset(**village, frame_offset=frame_offset)
dataset_fantasy_village = AirsimIntVarDataset(**fantasy_village, frame_offset=frame_offset)


village_loader = DataLoader(dataset_village, batch_size=batch_size, shuffle=False, num_workers=1)
fantasy_loader = DataLoader(dataset_fantasy_village, batch_size=batch_size, shuffle=False, num_workers=1)


if __name__ == '__main__':
    if PATH_WEIGHTS == "snapshots/super.snap.4.pt":
        conf_thresh = 0.15

    device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("using device {0}".format(device))



    batchnorm = True
    # from superpoint_magicleap.demo_superpoint import SuperPointFrontend
    # sp_magic = SuperPointFrontend(weights_path="superpoint_magicleap/superpoint_v1.pth",
    #                            nms_dist=8,conf_thresh=conf_thresh, nn_thresh=0.3)

    nms = MagicNMS()
    nms = PoolingNms(8)


    #run_all_snapshots()
    #test_distilled(fantasy_loader)
    #test_magicleap1(village_loader, angle=0.0)
    #test_magicleap(fantasy_loader, angle=5.0)

    # run_good(fantasy_loader, angle=0.0)
    run_distilled(fantasy_loader, angle=5.0)
    run_distilled(village_loader, angle=5.0)
    # run_good(village_loader, angle=5.0)
    print('village_loader')
    # print('fantasy_loader')

    #measure_performance(village_loader)
