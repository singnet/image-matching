import numpy as np
import numpy
import util
import cv2, os
import sys

from os import listdir
from os.path import isfile, join


from superpoint_magicleap.demo_superpoint import PointTracker, SuperPointFrontend
from goodpoint import GoodPoint
from fem.wrapper import SuperPoint
from fem.nonmaximum import MagicNMS
import torch
from fem.bench import get_points_desc, preprocess, draw_matches, replication_ratio, coverage, harmonic


GOOD_MATCH_THRESHOLD = 5
fe = None
dataset_root = '/mnt/fileserver/shared/datasets/SLAM_DATA/hpatches-dataset/hpatches-benchmark/data/hpatches-sequences-release'
ld = sorted(listdir(dataset_root))

device = 'cuda'
PATH_WEIGHTS = "./superpoint_magicleap/superpoint_v1.pth"

weight = "./snapshots/super3400.pt"
nms = MagicNMS(nms_dist=8)


thresh = 0.0207122295525
thresh = 0.015
nn_thresh = 0.85



fe = SuperPointFrontend(weights_path=PATH_WEIGHTS,
                        nms_dist=8,
                        conf_thresh=thresh,
                        nn_thresh=nn_thresh,
                        cuda=True)

activation = torch.nn.LeakyReLU()

def strip_module(st):
    if 'module' in st:
        return st.split('module.')[1]
    return st



#sp = GoodPoint(dustbin=0,
#               activation=activation,
#               batchnorm=True,
#               grid_size=8,
#               nms=nms).eval().to(device)
#data = torch.load(weight, map_location=device)['superpoint']
#sp.load_state_dict({strip_module(x): v for (x,v) in data.items()})


#sp = SuperPoint(MagicNMS()).to(device).eval()
#sp.load_state_dict(torch.load(PATH_WEIGHTS))

if fe is not None:
    sp = None

matcher = PointTracker(max_length=2, nn_thresh=nn_thresh)

img_output = 0

nCasesTotal = 0
nCasesLight = 0

TotalAccuracy = 0
LightAccuracy = []
LightCoverage = []
ViewCoverage = []
ViewAccuracy = []
LightReplication = []
ViewReplication = []

break_flag = False


imwrite = False
draw = False
imwrite = True
result_dir = './results/'



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
    img_1 = preprocess(img_1_src)
    #if any((1900 < x) for x in img_1.shape):
    #    device = 'cpu'
    #else:
    #    device = 'cuda'
    pts_1, desc_1 = get_points_desc(fe, sp, device, img_1, thresh)
    fReplicatedRatioMean = 0
    for n in range(1, N):
        img_2_src = cv2.imread(img_list[n])

        H = np.loadtxt(hom_list[n-1])

        IMG_SIZE_MAX = [max(img_1_src.shape[0], img_2_src.shape[0]), max(img_1_src.shape[1], img_2_src.shape[1])]

        dy1 = abs(img_1_src.shape[0] - IMG_SIZE_MAX[0])/2
        dy2 = abs(img_2_src.shape[0] - IMG_SIZE_MAX[0])/2

        img_2 = preprocess(img_2_src)
        pts_2, desc_2 = get_points_desc(fe, sp, device, img_2, thresh)
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

        coverage1, coverage_mask, intersect = coverage(img_1_src[:, :, -1], None, matches, pts_1[:2, matches[0, :].astype('int32')])
        if draw:
            cv2.imshow('coverage', coverage_mask)
        accuracy = float(nGoodMatches) / float(nMatches) if nMatches else 0.0

        if is_attribs:
            LightCoverage.append(coverage1)
            LightAccuracy.append(accuracy)
        else:
            ViewCoverage.append(coverage1)
            ViewAccuracy.append(accuracy)

        nCasesTotal += 1

        # Checking detector repeatability
        pts_1_r=[]
        pts_2_r = pts_2[:2, :]
        pts_1_r = pts_1[:2, :]
        pts_1_r = pts_1[:2, :]

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

        # nReplicated = 0
        # for i in range(pts_1_r.shape[1]):
        #     match_found = False
        #     mindist = 1e8
        #     for j in range(pts_2_r.shape[1]):
        #         dx = pts_2_r[0, j] - pt2_p[0, i]
        #         dy = pts_2_r[1, j] - pt2_p[1, i]
        #         err = np.sqrt(dx * dx + dy * dy)
        #         if err <= GOOD_MATCH_THRESHOLD:
        #             nReplicated += 1
        #             match_found = True
        #             break
        #     if match_found:
        #         continue
        fReplicatedRatio = replication_ratio(pt2_p, pts_2_r, GOOD_MATCH_THRESHOLD)
        # fReplicatedRatio = float(nReplicated) / float(pts_1_r.shape[1])
        if is_attribs:
            LightReplication.append(fReplicatedRatio)
        else:
            ViewReplication.append(fReplicatedRatio)

        print("%i\t\tdir: %s\t\t'Replication ratio: %f'\t\tnMatches: %i\tnGoodMatches: %i\t\tMatching accuracy: %f \t\tCoverage: %f" %
              (nCasesTotal, dir, fReplicatedRatio, nMatches, nGoodMatches, accuracy, coverage1))
        if draw or imwrite:
            img_output = draw_matches(matches, pts_1, pts_2, img_1_src, img_2_src)
        # vis_utils.draw_points(pts_1, img_1c, iscolor=False)
        # vis_utils.draw_points(pts_2, img_2, iscolor=False)

        c = 5
        #cv2.imshow('', img_output[IMG_SIZE_MAX[0]:, :, :])
        if imwrite:
            cv2.imwrite(os.path.join(result_dir, ''.join(img_list[n].split('/')[-2:]) + '.png'), img_output)
        if draw:
            cv2.imshow('matches', img_output)
            c = cv2.waitKey(100)
            if c == ord('q'):
                break_flag = True
                break

        sys.stdout.flush()

    if break_flag:
        break


meanLightAccuracy = sum(LightAccuracy) / float(len(LightAccuracy))
meanViewAccuracy = sum(ViewAccuracy) / float(len(ViewAccuracy))

meanLightReplication = sum(LightReplication) / float(len(LightReplication))
meanViewReplication = sum(ViewReplication) / float(len(ViewReplication))

meanLightCoverage = sum(LightCoverage) / float(len(LightCoverage))
meanViewCoverage = sum(ViewCoverage) / float(len(ViewCoverage))

print("Mean Light Replication: %f\tMean View Replication: %f\tMean Light Accuracy: %f\tMean View Accuracy: "
      "%f\tMean Light Coverage %f\t Mean View Coverage: %f" %
      (meanLightReplication, meanViewReplication, meanLightAccuracy, meanViewAccuracy, meanLightCoverage, meanViewCoverage) )


print('harmonic mean: {0}'.format(harmonic(meanLightReplication, meanViewReplication,
                                           meanLightAccuracy, meanViewAccuracy,
                                           meanLightCoverage, meanViewCoverage)))
print('threshold: {0}'.format(GOOD_MATCH_THRESHOLD))
