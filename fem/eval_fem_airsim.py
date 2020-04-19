from wrappers import cfem_wrapper as fm
from wrappers import eval_airsim_wrapper as evw

import numpy
import numpy as np
import cv2, os
import random

from itertools import count
import time

import torch

from fem.bench import get_points_desc
from fem.bench import coverage, harmonic
from fem import util
from fem.drawing import show_points
from fem.drawing import make_image_quad
from scipy.special import expit


PATH_SAVE = "/tmp/village_00_320x240_day_night_SP_fem_my"
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


def project(H_inv, pts_2):
    transp = H_inv @ numpy.stack([pts_2[:,1], pts_2[:,0], numpy.ones(len(pts_2))], axis=1)[:, (1,0,2)].T
    row_col = numpy.stack((transp[0], transp[1])).T
    return numpy.asarray(row_col / transp[-1][np.newaxis].T)


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


def precesion(pts_1, pts_2, vM, depth_1, depth_2, H_batch, img_size):
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
    for m in range(img_size[0]):
        for n in range(img_size[1]):
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

    ms = evw.calculateRepeatabilityAirsim(img_size[1], img_size[0], H4x4, vdepth1, vdepth2,
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


# https://www.semicolonworld.com/question/55456/rotate-image-and-crop-out-black-borders

def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotateImage(image, angle):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]
    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LANCZOS4)
    rot_mat_notranslate = rot_mat_notranslate.T
    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    pts_init = np.asarray([[-w2, h2], [w2, h2], [-w2, -h2], [w2, -h2]])
    pts_pert = np.stack([tl, tr, bl, br])
    pts_init = pts_init + (w2, h2)
    pts_pert = pts_pert + (w2, h2)
    x_min = min(0, pts_pert[:,0].min())
    y_min = min(0, pts_pert[:,1].min())
    pts_pert = pts_pert - (x_min, y_min)
    pts_init = pts_init[:,(1,0)]
    pts_pert = pts_pert[:,(1,0)]
    H, H_inv = util.calculate_h(pts_init, pts_pert)

    return result, H, H_inv


def loop(sp, loader, draw=True, print_res=True, thresh=0.5, desc_model=None, N=None, device='cuda', rotation_angle=0.0, fe=None):
    if sp is not None:
        sp = sp.to(device)
    nCases = 0
    meanTime = 0
    meanRecall = 0.
    meanPrecisionInv = 0.
    repeatability = 0.
    icoverage = 0.0
    rotate = False
    if rotation_angle > 0.0:
        rotate = True
    if N is None:
        N = len(loader)
    c = 0
    for i_batch, sample in enumerate(loader):
        img_1_batch = sample['img1'].numpy()
        img_2_batch = sample['img2'].numpy()
        depth_1_batch = sample['depth1'].numpy()
        depth_2_batch = sample['depth2'].numpy()
        H_batch = sample['H'].numpy()
        nCases += 1
        for j in range(1):

            img_1 = img_1_batch[j, :, :]
            img_2 = img_2_batch[j, :, :]
            if rotate:
                img_2_rot, H, H_inv = rotateImage(img_2, rotation_angle)
                img_2 = img_2_rot
            img_size = img_1.shape
            depth_1 = depth_1_batch[j, :, :]
            depth_2 = depth_2_batch[j, :, :]


            timg1 = np.expand_dims(np.expand_dims(img_1.astype('float32'), axis=0), axis=0)
            timg2 = np.expand_dims(np.expand_dims(img_2.astype('float32'), axis=0), axis=0)

            t1 = time.time()

            pts_1, desc_1_ = get_points_desc(device=device,
                    img=img_1[np.newaxis, np.newaxis,...],
                    thresh=thresh,
                    fe=fe, sp=sp)
            pts_1 = pts_1.T[:,(1,0, 2)]
            desc_1_ = desc_1_.T

            # pts_1, desc_1_ = fe.points_desc(torch.from_numpy(timg1).to(device), threshold=thresh)

            if desc_model:
                heatmap, desc_crude = desc_model.forward(torch.from_numpy(timg1).to(device) / 255.0)
                desc_1_ = util.descriptor_interpolate(desc_crude[0],
                                                    240,
                                                    320,
                                                    pts_1[:,:2])
            # validate against magicleap frontend
            # pts, desc, heat = front.run(timg1.squeeze() / 255.0)
            #assert (desc.transpose() - desc_1[0].detach().numpy()).sum() < 0.0001

            t2 = time.time()
            meanTime = meanTime + (t2 - t1)

            pts_2, desc_2_ = get_points_desc(device=device,
                    img=img_2[np.newaxis, np.newaxis,...],
                    thresh=thresh,
                    fe=fe, sp=sp)
            pts_2 = pts_2.T[:,(1,0,2)]
            desc_2_ = desc_2_.T


            # pts_2, desc_2_ = fe.points_desc(torch.from_numpy(timg2).to(device), threshold=thresh)

            if rotate:
    #            show_points(img_2.copy(), pts_2[:, 0:2].astype(np.int32), 'rot', scale=2)

                reproj = project(H_inv, pts_2)
                hmax, wmax = img_1.shape
                in_bounds = (reproj[:,0] > 0) * (reproj[:,0] < hmax) * (reproj[:,1] > 0) * (reproj[:,1] < wmax)
                reproj = numpy.concatenate([reproj, pts_2[:, 2][:,np.newaxis]], axis=1)
                pts_2 = reproj[in_bounds.nonzero()[0]]
                desc_2_ = desc_2_[in_bounds.nonzero()[0]]
                img_2 = img_2_batch[j, :, :]
    #           show_points(img_2.copy(), pts_2[:, 0:2].astype(np.int32), 'reproj', scale=2)
            if desc_model:
                heatmap, desc_crude2 = desc_model.forward(torch.from_numpy(timg2).to(device) / 255.0)
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
                desc_1 = desc_1_
                desc_2 = desc_2_

            K1 = sample['K1'][j]
            K2 = sample['K2'][j]
            pose1 = sample['pose1'][j]
            pose2 = sample['pose2'][j]

            rep, cov = repeat(H_batch[j], K1, K2, depth_1, depth_2, img_1, img_2, pose1, pose2, pts_1, pts_2, draw)
            repeatability += rep * 0.5
            icoverage += cov * 0.5
            rep, cov = repeat(np.linalg.inv(H_batch[j]), K2, K1, depth_2, depth_1, img_2, img_1, pose2, pose1,
                                    pts_2, pts_1, draw)
            repeatability += rep * 0.5
            icoverage += cov * 0.5


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
                ms, matches = precesion(pts_1, pts_2, vM, depth_1, depth_2, H_batch[j], img_size)
                if draw:
                    draw_matches(matches, pts_1, pts_2, img_output[img_size[0]:, :, :])
                precinv = 1 - float(ms.nMatches) / float(nMatches)
                if print_res:
                    print("%i/%i\trecall: %f\tnMatches: %i\tTime: %f" % (nCases, N, ms.recall, ms.nMatches, t2 - t1))
                meanPrecisionInv += precinv
                meanRecall += 0.5 * ms.recall
            if nMatches1 >= 1:
                ms2, matches2 = precesion(pts_2, pts_1, vM1, depth_2, depth_1, np.linalg.inv(H_batch[j]), img_size)
                if draw:
                    draw_matches(matches2, pts_2, pts_1, img_output2[img_size[0]:, :, :])
                if print_res:
                    print("%i/%i\trecall: %f\tnMatches: %i\tCoverage: %f\tTime: %f" % (nCases, N, ms2.recall, ms2.nMatches, cov, t2 - t1))
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
    mcoverage = icoverage / float(nCases)
    f1 = 2 * mrecall * mrepeat / (mrecall + mrepeat)
    print("Mean recall: {0}".format(mrecall))
    print("Repeatability: {0}".format(mrepeat))
    print("Coverage: {0}".format(mcoverage))
    print('harmonic mean: {0}'.format(harmonic(mrecall, mrepeat, mcoverage)))
    print("F1: {0}".format(f1))
    return f1


def repeat(H_batch, K1, K2, depth_1, depth_2, img_1, img_2, pose1, pose2, pts_1, pts_2, draw):
    new_coords1 = util.project3d(K1, K2, depth_1, pts_1[:, 0:2].astype(np.int32), pose1, pose2)
    geom_dist, ind2 = util.geom_match(new_coords1, pts_2[:, 0:2], 1)
    geom_match, nGeom = geom_match_to_vm(geom_dist, ind2.squeeze())
    ms1, matches1 = precesion(pts_1, pts_2, geom_match, depth_1, depth_2, H_batch, img_size=img_1.shape)
    if draw:
        img_output1 = make_image_quad(img_1, img_2, pts_1, pts_2)
        draw_matches(matches1, pts_1, pts_2, img_output1[img_1.shape[0]:, :, :])
    matches = matches1
    # compute coverage
    pt2_p = new_coords1
    pt2_p = pt2_p.T
    pts_1 = pts_1.T
    pts_2 = pts_2.T
    pt1 = pts_1[:2, matches[0, :].astype('int32')]
    pt2 = pts_2[:2, matches[1, :].astype('int32')]

    dx = pt2_p[0, :] - pt2[0, :]
    dy = pt2_p[1, :] - pt2[1, :]
    err = numpy.sqrt(dx * dx + dy * dy)
    matches = matches1
    nMatches = len(matches[0])
    GOOD_MATCH_THRESHOLD = 3.0
    for i in range(nMatches):
        if err[i] <= GOOD_MATCH_THRESHOLD:
            matches[2, i] = 0
        else:
            matches[2, i] = err[i]
    # coverage wants (x, y) as opoused to (row, col)
    coverage1, coverage_mask, intersect = coverage(img_1, None, matches, pt1[(1,0),:], 20)
    # cv2.imshow('coverage', coverage_mask)
    # ms1.recall = True matches / Matches so it is precision
    # but if we have used ground truth for matching then it is
    # points recovered / points so it is recall
    return ms1.recall, coverage1
