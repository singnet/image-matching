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


def test_magicleap(loader):

    from fem.wrapper import SuperPoint
    sp_path = '/home/noskill/projects/neuro-fem/fem/superpoint_magicleap/superpoint_v1.pth'

    sp = SuperPoint(nms).to(device)
    sp.load_state_dict(torch.load(sp_path))
    loop(sp, loader, thresh=0.015, print_res=False, draw=False)
    print('test superpoint completed')

def test_distilled(loader):
    from superpoint_05 import SuperPointNet
    sp = SuperPointNet().eval()
    path = '/home/noskill/projects/neuro-fem/fem/airsim_realsense_gpnt_model_last.pth'
    sp.load_state_dict(torch.load(path, map_location=device))
    sp.nms = MagicNMS()
    loop(sp, loader, thresh=0.015, print_res=False, draw=False)
    print('test distilled completed')


def run_good(loader):
    weight = "./snapshots/super1600.pt"

    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.ReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    loop(sp, loader, draw=True, print_res=True, thresh=0.1295525, desc_model=None)
    print('test goodpoint {0} completed'.format(weight))



dataset_village = AirsimIntVarDataset(**village, frame_offset=frame_offset)
dataset_fantasy_village = AirsimIntVarDataset(**fantasy_village, frame_offset=frame_offset)


village_loader = DataLoader(dataset_village, batch_size=batch_size, shuffle=False, num_workers=1)
fantasy_loader = DataLoader(dataset_fantasy_village, batch_size=batch_size, shuffle=False, num_workers=1)



if PATH_WEIGHTS == "snapshots/super.snap.4.pt":
    conf_thresh = 0.15



device = 'cuda' if torch.cuda.is_available() else 'cpu'


print("using device {0}".format(device))



batchnorm = True
# from superpoint_magicleap.demo_superpoint import SuperPointFrontend
# sp_magic = SuperPointFrontend(weights_path="superpoint_magicleap/superpoint_v1.pth",
#                            nms_dist=8,conf_thresh=conf_thresh, nn_thresh=0.3)

nms = MagicNMS()



#run_all_snapshots()
# test_distilled(fantasy_loader)
#test_magicleap(village_loader)
#run_good(fantasy_loader)
#print('fantasy_loader')
run_good(village_loader)
print('village_loader')
