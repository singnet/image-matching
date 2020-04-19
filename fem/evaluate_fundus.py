import os
import sys
import numpy
from fem.util import calculate_h, swap_rows, project_points2
from fem.wrapper import SuperPoint
from fem import drawing
import imageio
import torch
from superpoint_magicleap.demo_superpoint import PointTracker
from goodpoint import GoodPoint
from fem.nonmaximum import MagicNMS
from fem.bench import get_points_desc, preprocess, draw_matches, replication_ratio, coverage, harmonic
import cv2



def in_mask(pts_2, mask):
    pts2 = pts_2[:2].round().astype(numpy.int16).T
    return (mask[pts2[:, 1], pts2[:, 0]] > 128).nonzero()[0]


GOOD_MATCH_THRESHOLD = 5


def main(sp, thresh, nn_thresh=0.8, draw=True):

    matcher = PointTracker(max_length=2, nn_thresh=nn_thresh)
    fe = None
    dist_thresh = 0.8

    root_fire = '/mnt/fileserver/shared/datasets/fundus/FIRE/'
    gt_dir = os.path.join(root_fire, 'Ground Truth')
    images_dir = os.path.join(root_fire, 'Images')
    mask_dir = os.path.join(root_fire, 'Masks')

    gt_files = [os.path.join(gt_dir, x) for x in os.listdir(gt_dir)]
    nCasesTotal = 0
    mask = imageio.imread(os.path.join(mask_dir, 'mask.png'))
    mask = imageio.imread(os.path.join(mask_dir, 'feature_mask.png'))

    scale = 0.15
    mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    if draw:
        cv2.imshow('mask', mask)
    cov_all = []
    acc_all = []
    rep_all = []
    for gt_file in gt_files:
        name = gt_file.split('control_points_')[1].split('_')[0]
        img1_path = os.path.join(images_dir, name + '_1.jpg')
        img2_path = os.path.join(images_dir, name + '_2.jpg')
        img1 = imageio.imread(img1_path)
        img2 = imageio.imread(img2_path)
        img1 = cv2.resize(img1, dsize=None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, dsize=None, fx=scale, fy=scale)

        img_h, img_w = img1.shape[:2]
        arr = numpy.loadtxt(gt_file)
        arr[:, (0, 2)] = arr[:, (0, 2)] * scale
        arr[:, (1, 3)] = arr[:, (1, 3)] * scale
        points1 = arr[:, :2]
        points2 = arr[:, 2:]
        H, H_inv = calculate_h(points1, points2)
        # (x, y) to (row, col)
        points1 = swap_rows(arr[:, :2].T).T
        points2 = swap_rows(arr[:, 2:].T).T
        #in_bounds, points1_projected = project_points2(torch.from_numpy(H), None, torch.from_numpy(points1), img_h, img_w)
        #drawing.show_points(img1 / 256, points1.astype(numpy.int16), 'points1_img1', 0.5)
        #drawing.show_points(img2 / 256, points2.astype(numpy.int16), 'points2_img2', 0.5)
        #drawing.show_points(img2 / 256, points1_projected.cpu().long(), 'points1_img2', 0.5)

        desc_1, pts_1, desc_2, pts_2 = extract(device, fe, img1, img2, mask, sp, thresh)
        ### debug
        #img11 = drawing.draw_points(pts_1, img1.copy(), False)
        #img12 = drawing.draw_points(pts_2, img2.copy(), False)
        #in_bounds, points1_projected = project_points2(torch.from_numpy(H),
        #                                               None,
        #                                               torch.from_numpy(pts_1[:2].T),
        #                                               img_h,
        #                                               img_w)
        #img22 = drawing.draw_points(points1_projected.numpy().T, img2.copy(), False)
        #cv2.imshow('points11_img1', img11)
        #cv2.imshow('points22_img2', img12)
        #cv2.imshow('points11_img2', img22)
        #cv2.waitKey(0)

        nCasesTotal += 1
        matches = matcher.nn_match_two_way(desc_1, desc_2, nn_thresh=dist_thresh)
        nMatches = matches.shape[1]

        pt1 = pts_1[:2, matches[0, :].astype('int32')]
        pt2 = pts_2[:2, matches[1, :].astype('int32')]

        in_bounds, pt2_p = project_points2(torch.from_numpy(H), None, torch.from_numpy(pt1.T),
                                           img_h, img_w)
        nGoodMatches = 0
        pt2_p = pt2_p.T
        dx = pt2_p[0, :] - pt2[0, :]
        dy = pt2_p[1, :] - pt2[1, :]
        err = numpy.sqrt(dx * dx + dy * dy)
        rep = 0
        for i in range(nMatches):
            if err[i] <= GOOD_MATCH_THRESHOLD:
                nGoodMatches += 1
                matches[2, i] = 0
            else:
                matches[2, i] = err[i]

        replication = replication_ratio(pt2_p.cpu().numpy(), pts_2, GOOD_MATCH_THRESHOLD)
        accuracy = float(nGoodMatches) / float(nMatches) if nMatches else 0.0
        rep_all.append(replication)
        acc_all.append(accuracy)
        print('accuracy:', accuracy)
        print('replication:', replication)
        if draw:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            from super_debug import draw_desc_interpolate
            draw_desc_interpolate(img1, img2, swap_rows(pts_1[:2]).T, swap_rows(pts_2[:2]).T, desc_1.T, desc_2.T, 0)

            img_output = draw_matches(matches, pts_1, pts_2, img1, img2)
            # img_output[:, img_output.shape[1] // 2:] = drawing.draw_points(pt2_p.numpy(),
            #                                                                img_output[:, img_output.shape[1] // 2:], iscolor=False)
            cv2.imshow('matches', img_output)
        coverage1, coverage_mask, intersect = coverage(img1[:,:,-1], mask, matches, pt1)
        if draw:
            cv2.imshow('coverage', coverage_mask)
        cov_all.append(coverage1)
        if draw:
            cv2.imshow('itersect', intersect)
        print('converage:', coverage1)
        if draw:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        sys.stdout.flush()
    acc = numpy.mean(acc_all)
    cov = numpy.mean(cov_all)
    rep = numpy.mean(rep_all)
    print('average accuracy:', acc)
    print('average coverage:', cov)
    print('average replication:', rep)
    print('harmonic mean:', harmonic(acc, cov, rep))


def extract(device, fe, img1, img2, mask, sp, thresh):
    img1 = preprocess(img1, cv2.COLOR_BGR2GRAY)
    pts_1, desc_1 = get_points_desc(fe, sp, device, img1, thresh)
    in_mask1 = in_mask(pts_1, mask)
    # apply mask
    pts_1 = pts_1[:, in_mask1]
    desc_1 = desc_1[:, in_mask1]
    img2 = preprocess(img2, cv2.COLOR_BGR2GRAY)
    pts_2, desc_2 = get_points_desc(fe, sp, device, img2, thresh)
    in_mask2 = in_mask(pts_2, mask)
    # apply mask
    pts_2 = pts_2[:, in_mask2]
    desc_2 = desc_2[:, in_mask2]
    return desc_1, pts_1, desc_2, pts_2


if __name__ == '__main__':
    device = 'cuda'
    weight = "./snapshots/super3400.pt"
    weight = "./snapshots/super3400.pt"
    sp_path = '/home/noskill/projects/neuro-fem/fem/superpoint_magicleap/superpoint_v1.pth'
    nms = MagicNMS(nms_dist=8)
    sp = SuperPoint(MagicNMS()).to(device).eval()
    sp.load_state_dict(torch.load(sp_path))


    gp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval().to(device)

    gp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    super_thresh = 0.015
    thresh = 0.0207190856295525`
    main(gp, thresh=thresh, draw=False, nn_thresh=0.8)
