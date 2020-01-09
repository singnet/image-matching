from wrappers import cfem_wrapper as fm
from wrappers import eval_airsim_wrapper as evw

import numpy as np
import cv2, os
import random
from os import listdir
from os.path import isfile, join
from itertools import count
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from airsim_dataset import AirsimIntVarDataset
from fem.goodpoint import GoodPoint
import torch
from fem.nonmaximum import PoolingNms, MagicNMS
from fem import util
from scipy.special import expit


PATH_SAVE = "/tmp/village_00_320x240_day_night_SP_fem_my"

dir_day = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_day_light'
dir_night = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/00_night_light'
poses_file = '/mnt/fileserver/shared/datasets/AirSim/village_00_320x240/village_00.json'

PATH_SAVE_PTS = PATH_SAVE + '/data/'

magicleap_file = "superpoint_magicleap/superpoint_v1.pth"
magicleap_file = None

PATH_WEIGHTS = None

PATH_WEIGHTS = "./super12000.pt"
PATH_WEIGHTS = "snapshots/super16000.pt"


#PATH_WEIGHTS = "./snapshots/TWD/tr_0.5_dr_0.4/from_scratch_super17.pt"
#PATH_WEIGHTS = "./snapshots/TWD/tr_0.5_dr_0.4/super8.pt"


batchnorm=True
IMG_SIZE = [240, 320]
#a


conf_thresh= 0.020885
conf_thresh= 0.0455591090510123100629
if PATH_WEIGHTS == "snapshots/super.snap.4.pt":
    conf_thresh = 0.15



batchnorm = True
# from superpoint_magicleap.demo_superpoint import SuperPointFrontend
# sp_magic = SuperPointFrontend(weights_path="superpoint_magicleap/superpoint_v1.pth",
#                            nms_dist=8,conf_thresh=conf_thresh, nn_thresh=0.3)

nms = MagicNMS()



RESIZE = False
SAVE_RESULT = False
if RESIZE:
    PATH_SAVE = PATH_SAVE + "_resized/"
    PATH_SAVE_PTS = PATH_SAVE + 'data/'
    ratio = float(IMG_SIZE[1]) / 320.
    ratioy = float(IMG_SIZE[0]) / 240.
    if ratio > ratioy:
        ratio = ratioy

    IMG_SIZE[0] = int(IMG_SIZE[0] / ratio)
    IMG_SIZE[1] = int(IMG_SIZE[1] / ratio)

else:
    ratio = 1
    PATH_SAVE = PATH_SAVE + '/'


if SAVE_RESULT:
    import shutil
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
        if not os.path.exists(PATH_SAVE_PTS):
            os.makedirs(PATH_SAVE_PTS)
    else:
        shutil.rmtree(PATH_SAVE)
        os.makedirs(PATH_SAVE)
        if not os.path.exists(PATH_SAVE_PTS):
            os.makedirs(PATH_SAVE_PTS)


# filename = (PATH_SAVE + "img_points_%03i.png" % i for i in count(start=1, step=1))
filename = (PATH_SAVE + "img_points_%03i" % i for i in count(start=1, step=1))
filenamePts = (PATH_SAVE_PTS + "img_points_%03i" % i for i in count(start=1, step=1))

size = int(3)


correspCount = int(0)


frame_offset = 5
batch_size = 1
dataset = AirsimIntVarDataset(dir_day, dir_night, poses_file, frame_offset=frame_offset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)


N = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print("using device {0}".format(device))


sp = GoodPoint(dustbin=0,
               activation=torch.nn.ReLU(),
                batchnorm=batchnorm,
                grid_size=8,
                nms=nms).eval()


# sp_magic = get_superpoint_model(magicleap_file="superpoint_magicleap/superpoint_v1.pth",
#                 batchnorm=False,
#                 nms=nms).eval()

#sp_magic = SuperPoint(torch.nn.ReLU(),
#                batchnorm=True,
#                nms=nms).eval()
#sp_magic = sp_magic.to(device)
#
#
#sp_magic.load_state_dict(torch.load("snapshots/super12000.pt", map_location=device)['superpoint'])
sp_magic = None

sp.isTraining = False
sp.isDropout = False




def draw_points(pts, img, iscolor=True):
    r = 0
    g = 255
    b = 0
    for i in range(pts.shape[0]):
        pt = (int(round(pts[i, 1])), int(round(pts[i, 0])))
        if iscolor:
            r = random.randint(0, 32767) % 256
            g = random.randint(0, 32767) % 256
            b = 0 if (r + g > 255) else ( 255 - (r + g))

        color = (b, g, r)

        cv2.circle(img, pt, 2, color=color, thickness=-1)
    return img

def draw_matches(matches, pts1, pts2, imgpair, iscolor=True):
    # matches[0, :] = m_idx1
    # matches[1, :] = m_idx2
    # matches[2, :] = scores
    r = 0
    g = 255
    b = 0
    for i in range(matches.shape[1]):
        pt1 = (int(round(pts1[int(matches[0,i]), 1])), int(round(pts1[int(matches[0,i]), 0])))
        pt2 = (int( round(pts2[int(matches[1, i]), 1])  + imgpair.shape[1]/2 ), int( round(pts2[int(matches[1, i]), 0]) ))

        if iscolor:
            r = random.randint(0, 32767) % 256
            g = random.randint(0, 32767) % 256
            b = 0 if (r + g > 255) else ( 255 - (r + g))

        color = (b, g, r)

        cv2.circle(imgpair, pt1,  2, color=color, thickness=-1)
        cv2.circle(imgpair, pt2, 2, color=color, thickness=-1)
        cv2.line(imgpair, pt1, pt2, color, thickness=1)

    return imgpair


def rewrite_data_to_file(file_path, data):
    file = open(file_path, 'a')
    file.seek(0)
    np.savetxt(file, data, delimiter='\t')
    file.truncate()
    file.close()




def wrap_features(pts, desc):
    vDesc = fm.VDESCRIPTOR()
    vKpt = fm.VKEYPOINT()
    for i in range(pts.shape[0]):
        p = fm.KeyPoint_t()
        p.flX = float(pts[i, 1])
        p.flY = float(pts[i, 0])

        vKpt.append(p)
        d = fm.Descriptor_t()
        for k in range(fm.NFEATURES):
            val = desc[i, k] * 256
            if val > 127:
                val = 127
            if val < -127:
                val = -127

            d[k] = int(val)

        vDesc.append(d)
    return vDesc, vKpt

c = 0


def precesion(pts_1, pts_2, vM, depth_1, depth_2, H_batch):
    matches222 = np.zeros((3, len(vM)))
    # print("NMATCHES:", matches.shape[1])
    for i in range(len(vM)):
        matches222[0, i] = vM[i].nIndex1
        matches222[1, i] = vM[i].nIndex2
        matches222[2, i] = vM[i].flQuality

    H4x4 = evw.HOMOGRAPHY3D()
    for m in range(4):
        for n in range(4):
            H4x4[m * 4 + n] = float(H_batch[m, n])
    vdepth1 = evw.std_vector_float()
    vdepth2 = evw.std_vector_float()
    for m in range(IMG_SIZE[0]):
        for n in range(IMG_SIZE[1]):
            vdepth1.append(float(depth_1[m, n]))
            vdepth2.append(float(depth_2[m, n]))
    vKm1 = evw.VKEYPOINT()
    vKm2 = evw.VKEYPOINT()
    for i in range(len(vM)):
        idx1 = int(matches222[0, i])
        idx2 = int(matches222[1, i])
        p = evw.KeyPoint_t()
        p.flX = float(pts_1[idx1, 1])
        p.flY = float(pts_1[idx1, 0])
        vKm1.append(p)

        p.flX = float(pts_2[idx2, 1])
        p.flY = float(pts_2[idx2, 0])
        vKm2.append(p)

    ms = evw.calculateRepeatabilityAirsim(IMG_SIZE[1], IMG_SIZE[0], H4x4, vdepth1, vdepth2,
                                          vKm1, vKm2, size)

    return ms, matches222


def match(vk1, vd1, vk2, vd2):
    vM = fm.VMATCH()
    matcher = fm.CFeatureMatcher()
    q = matcher.match_basic(vM, vk1, vd1, vk2, vd2)
    nMatches = len(vM)
    return vM, nMatches


def geom_match_to_vm(geom_dist, ind2):
    class Idx:
        def __init__(self, idx1, idx2, quality):
            self.nIndex1 = idx1
            self.nIndex2 = idx2
            self.flQuality = quality

    idx1 = np.arange(len(ind2))
    idx2 = ind2
    quality = expit(1 / (geom_dist + 0.001))
    matches = []
    for i in range(len(geom_dist)):
        matches.append(Idx(idx1[i], idx2[i], quality[i]))
    return matches, len(matches)


def loop(draw=True, weights=PATH_WEIGHTS, print_res=True):
    sp.load_state_dict(torch.load(weights, map_location=device)['superpoint'])

    fe = sp.to(device)
    nCases = 0
    meanTime = 0
    meanRecall = 0.
    meanPrecisionInv = 0.
    repeatability = 0.
    c = 0
    for i_batch, sample in enumerate(dataloader):
        img_1_batch = sample['img1'].numpy()
        img_2_batch = sample['img2'].numpy()
        depth_1_batch = sample['depth1'].numpy()
        depth_2_batch = sample['depth2'].numpy()
        H_batch = sample['H'].numpy()
        nCases += 1
        for j in range(batch_size):

            img_1 = img_1_batch[j, :, :]
            img_2 = img_2_batch[j, :, :]
            depth_1 = depth_1_batch[j, :, :]
            depth_2 = depth_2_batch[j, :, :]


            timg1 = np.expand_dims(np.expand_dims(img_1.astype('float32'), axis=0), axis=0)
            timg2 = np.expand_dims(np.expand_dims(img_2.astype('float32'), axis=0), axis=0)

            t1 = time.time()

            pts_1, desc_1_ = fe.points_desc(torch.from_numpy(timg1).to(device), threshold=conf_thresh)

            if sp_magic:
                heatmap, desc_crude = sp_magic.forward(torch.from_numpy(timg1).to(device) / 255.0)
                desc_1_ = util.descriptor_interpolate(desc_crude[0],
                                                    240,
                                                    320,
                                                    pts_1[:,:2])
            # validate against magicleap frontend
            # pts, desc, heat = front.run(timg1.squeeze() / 255.0)
            #assert (desc.transpose() - desc_1[0].detach().numpy()).sum() < 0.0001

            t2 = time.time()
            meanTime = meanTime + (t2 - t1)

            pts_2, desc_2_ = fe.points_desc(torch.from_numpy(timg2).to(device), threshold=conf_thresh)
            if sp_magic:
                heatmap, desc_crude2 = sp_magic.forward(torch.from_numpy(timg2).to(device) / 255.0)
                desc_2_ = util.descriptor_interpolate(desc_crude2[0],
                                                    240,
                                                    320,
                                                    pts_2[:,:2])
            # print(len(pts_1))
            # print(len(pts_2))
            if isinstance(desc_1_, list):
                desc_1 = desc_1_[0].detach().cpu().numpy()
                desc_2 = desc_2_[0].detach().cpu().numpy()
            else:
                desc_1 = desc_1_.detach().cpu().numpy()
                desc_2 = desc_2_.detach().cpu().numpy()


            K1 = sample['K1'][j]
            K2 = sample['K2'][j]
            pose1 = sample['pose1'][j]
            pose2 = sample['pose2'][j]

            repeatability += repeat(H_batch[j], K1, K2, depth_1, depth_2, img_1, img_2, pose1, pose2, pts_1, pts_2, draw) * 0.5
            repeatability += repeat(np.linalg.inv(H_batch[j]), K2, K1, depth_2, depth_1, img_2, img_1, pose2, pose1,
                                    pts_2, pts_1, draw) * 0.5

            # show_points(img_1.copy(), pts_1[:, 0:2][:40].astype(np.int32), 'orig', scale=2)
            # show_points(img_2.copy(), new_coords1, 'projected', scale=2)

            # THIS IS MUCH SLOWER THAN THE NUMPY OPTION
            # desc_1 = desc_1_[0].detach().cpu()
            # desc_2 = desc_2_[0].detach().cpu()
            if draw:
                from super_debug import draw_desc_interpolate
                draw_desc_interpolate(img_1, img_2, pts_1, pts_2, desc_1, desc_2, 0)
                draw_desc_interpolate(img_2, img_1, pts_2, pts_1, desc_2, desc_1, 1)
                img_output = make_image_quad(img_1, img_2, pts_1, pts_2)
                img_output2 = make_image_quad(img_2, img_1, pts_2, pts_1)

            vD1, vK1 = wrap_features(pts_1, desc_1)
            vD2, vK2 = wrap_features(pts_2, desc_2)

            vM, nMatches = match(vK1, vD1, vK2, vD2)
            vM1, nMatches1 = match(vK2, vD2, vK1, vD1)
            if nMatches < 1 and nMatches1 < 1:
                continue

            if nMatches >= 1:
                ms, matches = precesion(pts_1, pts_2, vM, depth_1, depth_2, H_batch[j])
                if draw:
                    draw_matches(matches, pts_1, pts_2, img_output[IMG_SIZE[0]:, :, :])
                precinv = 1 - float(ms.nMatches) / float(nMatches)
                if print_res:
                    print("%i/%i\trecall: %f\tnMatches: %i\tTime: %f" % (nCases, N, ms.recall, ms.nMatches, t2 - t1))
                meanPrecisionInv += precinv
                meanRecall += 0.5 * ms.recall
            if nMatches1 >= 1:
                ms2, matches2 = precesion(pts_2, pts_1, vM1, depth_2, depth_1, np.linalg.inv(H_batch[j]))
                if draw:
                    draw_matches(matches2, pts_2, pts_1, img_output2[IMG_SIZE[0]:, :, :])
                if print_res:
                    print("%i/%i\trecall: %f\tnMatches: %i\tTime: %f" % (nCases, N, ms2.recall, ms2.nMatches, t2 - t1))
                    print('\n')
                meanRecall += 0.5 * ms2.recall

            if draw:
                cv2.imshow('filtered', img_output)
                cv2.imshow('filtered_inv', img_output2)
                c = cv2.waitKey(10)

                if c == 1048603:
                    break

            if SAVE_RESULT:
                fn = next(filename) + "_recall_%05f" % ms.recall + '.png'
                cv2.imwrite(fn, img_output)

                # with open(next(filenamePts) + '_recall_%05f' % ms.recall + '.txt', 'w') as fp:
                #
                #     for i in range(matches.shape[1]):
                #         pt1 = (int(round(pts_1[0, int(matches[0, i])])), int(round(pts_1[1, int(matches[0, i])])))
                #         pt2 = (int(round(pts_2[0, int(matches[1, i])])), int(round(pts_2[1, int(matches[1, i])])))
                #         fp.write('%i\t%i\t%i\t%i\n'%(pt1[0],pt1[1],pt2[0],pt2[1]))
                #
                #     fp.close()

            if nCases >= N:
                break

        if nCases >= N:
            break

        if c == 1048603:
            break

    mrecall = meanRecall / float(nCases)
    mrepeat = repeatability / float(nCases)
    f1 = 2 * mrecall * mrepeat / (mrecall + mrepeat)
    if print_res:
        print("Mean recall: %f\tMean 1-precision: %f\tMean time: %f"%(mrecall,
                                                                  meanPrecisionInv/float(nCases),
                                                                  meanTime/float(nCases)))
        print("Repeatability: {0}".format(mrepeat))
        print("F1: {0}".format(f1))
    return f1


def repeat(H_batch, K1, K2, depth_1, depth_2, img_1, img_2, pose1, pose2, pts_1, pts_2, draw):
    new_coords1 = util.project3d(K1, K2, depth_1, pts_1[:, 0:2].astype(np.int32), pose1, pose2)
    geom_dist, ind2 = util.geom_match(new_coords1, pts_2[:, 0:2])
    geom_match, nGeom = geom_match_to_vm(geom_dist, ind2.squeeze())
    ms1, matches1 = precesion(pts_1, pts_2, geom_match, depth_1, depth_2, H_batch)
    if draw:
        img_output1 = make_image_quad(img_1, img_2, pts_1, pts_2)
        draw_matches(matches1, pts_1, pts_2, img_output1[IMG_SIZE[0]:, :, :])
    # ms1.recall = True matches / Matches so it is precision
    # but if we have used ground truth for matching then it is
    # points recovered / points so it is recall
    return ms1.recall


def make_image_quad(img_1, img_2, pts_1, pts_2):
    img_output = np.zeros(shape=(2 * IMG_SIZE[0], 2 * IMG_SIZE[1], 3), dtype=np.uint8)
    img_1 = np.repeat(np.expand_dims(img_1.astype('uint8'), axis=2), 3, axis=2)
    img_2 = np.repeat(np.expand_dims(img_2.astype('uint8'), axis=2), 3, axis=2)
    img_output[:IMG_SIZE[0], :IMG_SIZE[1], :] = img_1
    img_output[:IMG_SIZE[0], IMG_SIZE[1]:, :] = img_2
    img_output[IMG_SIZE[0]:, :IMG_SIZE[1], :] = img_1
    img_output[IMG_SIZE[0]:, IMG_SIZE[1]:, :] = img_2
    draw_points(pts_1, img_output[:IMG_SIZE[0], :IMG_SIZE[1], :], iscolor=False)
    draw_points(pts_2, img_output[:IMG_SIZE[0], IMG_SIZE[1]:, :], iscolor=False)
    draw_points(pts_1, img_1, iscolor=False)
    draw_points(pts_2, img_2, iscolor=False)

    img_output[:IMG_SIZE[0], :IMG_SIZE[1], :] = img_1
    img_output[:IMG_SIZE[0], IMG_SIZE[1]:, :] = img_2
    return img_output.copy()


def run_all_snapshots():
    best_f1 = 0.0
    best_path = None
    for f in os.listdir('.'):
        if f.endswith('.pt'):
            current = loop(draw=False, weights=f, print_res=False)
            if best_f1 < current:
                print('new best: {0}, f1: {1} '.format(f, current))
                best_f1 = current
                best_path = f

loop(draw=False)
