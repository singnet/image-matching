import numpy as np
import numpy
import util
import cv2, os
import sys
import random
from os import listdir
from os.path import isfile, join
from itertools import count
from superpoint_magicleap.demo_superpoint import SuperPointFrontend
from superpoint_magicleap.demo_superpoint import PointTracker
from goodpoint import GoodPoint
from fem.nonmaximum import MagicNMS
import torch
import time
# import vis_utils

GOOD_MATCH_THRESHOLD = 3

dataset_root = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/hpatches-benchmark/data/hpatches-sequences-release'
ld = sorted(listdir(dataset_root))

PATH_WEIGHTS = "./superpoint_magicleap/superpoint_v1.pth"
fe = None
#fe = SuperPointFrontend(weights_path=PATH_WEIGHTS,
#                          nms_dist= 4,
#                          conf_thresh= 0.025,
#                          nn_thresh= 0.5,
#                          cuda=True)
weight = "./snapshots/super1600.pt"
nms = MagicNMS(nms_dist=8)

device = 'cuda'
sp = GoodPoint(dustbin=0,
               activation=torch.nn.ReLU(),
               batchnorm=True,
               grid_size=8,
               nms=nms).eval().to(device)

sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
thresh=0.056295525


nn_thresh = 0.8
matcher = PointTracker(max_length=2, nn_thresh=nn_thresh)

img_output = 0

nCasesTotal = 0
nCasesLight = 0

TotalAccuracy = 0
LightAccuracy = []
ViewAccuracy = []
LightReplication = []
ViewReplication = []

break_flag = False
for d in range(len(ld)):
    dir = ld[d]
    img_dir = join(dataset_root, dir)
    files_list = sorted([join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f))])
    img_list = sorted([im for im in files_list if '.ppm' in im])
    hom_list = sorted([f for f in files_list if 'H_' in f])
    N = len(img_list)

    is_attribs = False
    attribs = [a for a in files_list if 'attribs' in a]
    for a in attribs:
        with open(a, 'r') as f:
            atr = f.readlines()

        if 'light' in atr[0]:
            is_attribs = True
            break

    img_1_src = cv2.imread(img_list[0])
    img_1 = cv2.cvtColor(img_1_src, cv2.COLOR_RGB2GRAY)
    timg1 = np.expand_dims(np.expand_dims(img_1.astype('float32'), axis=0), axis=0)
    if any((1500 < x) for x in timg1.shape):
        device = 'cpu'
    else:
        device = 'cuda'

    if fe is not None:
        pts_1, desc_1, heatmap_1 = fe.run(img_1.astype('float32') / 255.)
    else:
        pts_1, desc_1 = sp.to(device).points_desc(torch.from_numpy(timg1).to(device), threshold=thresh)
        pts_1 = pts_1.T
        pts_1 = numpy.concatenate([util.swap_rows(pts_1[:2]), pts_1[2, :][numpy.newaxis,:]])
        desc_1 = desc_1[0].T.cpu().detach().numpy()
    fReplicatedRatioMean = 0
    for n in range(1, N):
        img_2_src = cv2.imread(img_list[n])

        H = np.loadtxt(hom_list[n-1])

        IMG_SIZE_MAX = [max(img_1_src.shape[0], img_2_src.shape[0]), max(img_1_src.shape[1], img_2_src.shape[1])]

        dy1 = abs(img_1_src.shape[0] - IMG_SIZE_MAX[0])/2
        dy2 = abs(img_2_src.shape[0] - IMG_SIZE_MAX[0])/2


        img_2 = cv2.cvtColor(img_2_src, cv2.COLOR_RGB2GRAY)
        timg2 = np.expand_dims(np.expand_dims(img_2.astype('float32'), axis=0), axis=0)

        if fe is not None:
            pts_2, desc_2, heatmap_2 = fe.run(img_2.astype('float32') / 255.)
        else:
            pts_2, desc_2 = sp.to(device).points_desc(torch.from_numpy(timg2).to(device), threshold=thresh)
            pts_2 = pts_2.T
            desc_2 = desc_2[0].T.cpu().detach().numpy()
            pts_2 = numpy.concatenate([util.swap_rows(pts_2[:2]), pts_2[2, :][numpy.newaxis,:]])
        nCasesTotal += 1



        matches = matcher.nn_match_two_way(desc_1, desc_2, nn_thresh=nn_thresh)
        nMatches = matches.shape[1]

        pt1 = pts_1[:2, matches[0, :].astype('int32')]
        pt2 = pts_2[:2, matches[1, :].astype('int32')]

        pt1 = np.vstack( [pt1, np.ones([1, pt1.shape[1]])] )
        pt2 = np.vstack( [pt2, np.ones([1, pt2.shape[1]])] )

        pt2_p = np.ones(pt1.shape)
        h1 = np.expand_dims(H[0, :], 0)
        pt2_p[0,:]= np.matmul(h1, pt1)

        h2 = np.expand_dims(H[1, :], 0)
        pt2_p[1, :] = np.matmul(h2, pt1)

        h3 = np.expand_dims(H[2, :], 0)
        pt2_p[2, :] = np.matmul(h3, pt1)

        pt2_p[0,:] = pt2_p[0,:]/ pt2_p[2,:]
        pt2_p[1, :] = pt2_p[1, :] / pt2_p[2, :]

        nGoodMatches = 0
        dx = pt2_p[0, :] - pt2[0, :]
        dy = pt2_p[1, :] - pt2[1, :]
        err = np.sqrt(dx * dx + dy * dy)

        for i in range(nMatches):
            if err[i] <= GOOD_MATCH_THRESHOLD:
                nGoodMatches += 1
                matches[2, i] = 0
        accuracy = float(nGoodMatches) / float(nMatches)

        if is_attribs:
            LightAccuracy.append(accuracy)
        else:
            ViewAccuracy.append(accuracy)

        nCasesTotal += 1

        # Checking detector repeatability
        pts_1_r=[]
        pts_2_r = pts_2[:2, :]
        if pts_1.shape[1] > 300:
            pts_1_r = pts_1[:2, :300]
        else:
            pts_1_r = pts_1[:2, :300]

        pts1_r = np.vstack([pts_1_r, np.ones([1, pts_1_r.shape[1]])])

        pt2_p = np.ones(pts1_r.shape)
        h1 = np.expand_dims(H[0, :], 0)
        pt2_p[0, :] = np.matmul(h1, pts1_r)

        h2 = np.expand_dims(H[1, :], 0)
        pt2_p[1, :] = np.matmul(h2, pts1_r)

        h3 = np.expand_dims(H[2, :], 0)
        pt2_p[2, :] = np.matmul(h3, pts1_r)

        pt2_p[0, :] = pt2_p[0, :] / pt2_p[2, :]
        pt2_p[1, :] = pt2_p[1, :] / pt2_p[2, :]

        nReplicated = 0
        for i in range(pts_1_r.shape[1]):
            match_found = False
            mindist = 1e8
            for j in range(pts_2_r.shape[1]):
                dx = pts_2_r[0, j] - pt2_p[0, i]
                dy = pts_2_r[1, j] - pt2_p[1, i]
                err = np.sqrt(dx * dx + dy * dy)
                if err <= GOOD_MATCH_THRESHOLD:
                    nReplicated += 1
                    match_found = True
                    break
            if match_found:
                continue

        fReplicatedRatio = float(nReplicated) / float(pts_1_r.shape[1])
        if is_attribs:
            LightReplication.append(fReplicatedRatio)
        else:
            ViewReplication.append(fReplicatedRatio)

        print("%i\t\tdir: %s\t\t'Replication ratio: %f'\t\tnMatches: %i\tnGoodMatches: %i\t\tMatching accuracy: %f" %
              (nCasesTotal, dir, fReplicatedRatio, nMatches, nGoodMatches, accuracy))

       # img_output = np.zeros(shape=(2 * IMG_SIZE_MAX[0], img_1_src.shape[1] + img_2_src.shape[1], 3), dtype=np.uint8)
       # img_1c = np.repeat(np.expand_dims(img_1.astype('uint8'), axis=2), 3, axis=2)
       # img_2 = np.repeat(np.expand_dims(img_2.astype('uint8'), axis=2), 3, axis=2)

       # img_output[IMG_SIZE_MAX[0]:IMG_SIZE_MAX[0]+img_1_src.shape[0], :img_1_src.shape[1], :] = img_1c
       # img_output[IMG_SIZE_MAX[0]:IMG_SIZE_MAX[0]+img_2_src.shape[0], img_1_src.shape[1]:, :] = img_2
       # import drawing
       # drawing.draw_matches(matches, pts_1, pts_2, img_output[IMG_SIZE_MAX[0]:, :, :])
        # vis_utils.draw_points(pts_1, img_1c, iscolor=False)
        # vis_utils.draw_points(pts_2, img_2, iscolor=False)

        c = 5
        #img_output[:img_1_src.shape[0], :img_1_src.shape[1], :] = img_1c
       # img_output[:img_2_src.shape[0], img_1_src.shape[1]:, :] = img_2
       # cv2.imshow('', img_output[IMG_SIZE_MAX[0]:, :, :])
       # c = cv2.waitKey(10)
        if c == 1048603:
            break_flag = True
            break

        sys.stdout.flush()

    if break_flag:
        break


meanLightAccuracy = sum(LightAccuracy) / float(len(LightAccuracy))
meanViewAccuracy = sum(ViewAccuracy) / float(len(ViewAccuracy))

meanLightReplication = sum(LightReplication) / float(len(LightReplication))
meanViewReplication = sum(ViewReplication) / float(len(ViewReplication))


print("Mean Light Replication: %f\tMean View Replication: %f\tMean Light Accuracy: %f\tMean View Accuracy: %f\t" %
      (meanLightReplication, meanViewReplication, meanLightAccuracy, meanViewAccuracy) )


print('threshold: {0}'.format(GOOD_MATCH_THRESHOLD))
