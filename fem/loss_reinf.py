from collections import defaultdict
from fem.util import numpy_nonzero
import sklearn.neighbors

import torch

from fem import util

import numpy

from fem import hom
from fem.reinf_utils import random_dist


def sigmoid_abs(x):
    """
    RBF based on sigmoid
    sigmoid_abs(0): 1.0
    sigmoid_abs(1): 0.75
    sigmoid_abs(2): 0.51
    sigmoid_abs(4): 0.076
    sigmoid_abs(8): -0.52
    sigmoid_abs(12): -0.81
    sigmoid_abs(16): -0.93
    """
    return (numpy.abs(1 / (1 + numpy.exp(-x / 4)) - 0.5) - 0.5) * -4 - 1


def loss_hom(average=[0.0], neg_reward=-0.5, **kwargs):
    """
    Loss
    :param H: torch.Tensor
        homography or inverse homography 3x3
    :param points1:
    :param logprob1:
    :param desc1:
    :param desc2:
    :param point1_mask:
    :param average:
    :param img1:
    :param img2:
    :return:
    """
    # map point to new image plane with H
    H = kwargs['H']
    point1_mask = kwargs['point1_mask']
    points11 = kwargs['points1']
    in_bounds, points1projected = util.project_points(H, point1_mask, points11)
    points1, points1projected = util.get_points_in_bounds(in_bounds, points11, points1projected)

    # compute descriptors
    # use only points whose projections are in bounds
    desc1 = kwargs['desc1']
    desc1_int = util.descriptor_interpolate(desc1, 256,
                                            256, points1)


    result = dict()
    if not bool(in_bounds.sum()):
        reward, loss_desc, quality = torch.ones((1,)).to(desc1) * -0.1, None, torch.zeros((1,)).to(desc1)
    else:
        desc2 = kwargs['desc2']
        points2 = kwargs.get('points2', None)
        desc2_points2 = None
        if points2 is None:
            points2 = points1projected
        desc2_int = util.descriptor_interpolate(desc2, 256,
                                                 256, points2)

        img1 = kwargs.get('img1')
        img2 = kwargs.get('img2')

        reward, loss_desc, quality, quality_desc, means = match_desc_reward(points1projected,
                                                           points2,
                                                           desc1_int, desc2_int,
                                                           img1=img1,
                                                           use_geom=kwargs.get('use_geom', False),
                                                           use_means=True,
                                                           img2=img2,
                                                           points=points1,
                                                           dist_thres=4,
                                                           neg_reward=neg_reward)
        result['quality_desc'] = quality_desc.squeeze()
    deb = True
    deb = False
    if deb:
        import cv2
        cv2.imshow('img1', (img1.cpu() / 256).numpy())
        cv2.waitKey(500)
        import drawing
        drawing.show_points(img1.cpu() / 256, points1.cpu(), 'points1_img1', 2)
        drawing.show_points(img2.cpu() / 256, points1projected.cpu(), 'points1_img2', 2)

    result['quality'] = quality.squeeze()
    rew = reward.mean()

    loss = None
    non_point_mask = (point1_mask == 0)
    logprob1 = kwargs.get('logprob1')
    logprob2 = kwargs.get('logprob2')
    if bool(point1_mask.sum()):
        if kwargs.get('use_geom', False) and numpy.prod(means.shape):
            means = means.long()

            H_inv = numpy.linalg.inv(H.cpu())
            loss_points = -logprob2[means.long()[:,0], means.long()[:,1]].mean()
            in_bounds, coords1 = util.project_points(H_inv, None, means)
            if numpy.prod(coords1.shape):
                loss_points = loss_points / 2.0 - logprob2[coords1[:, 0], coords1[:, 1]].mean()
            else:
                loss_points = (-logprob1[numpy_nonzero(point1_mask)] * (reward)).mean()
        else:
            loss_points = (-logprob1[numpy_nonzero(point1_mask)] * (reward)).mean()
            # points2_mask = kwargs['point2_mask']
            # loss_points = -((logprob1[numpy_nonzero(point1_mask)] + logprob2[numpy_nonzero(points2_mask)][k2]) * (reward)).mean()
        if loss is None:
            loss = 0.0
        loss = loss_points + loss
        points_expected_reward = (torch.exp(logprob1)[numpy_nonzero(point1_mask)] * reward).mean()
        result['loss_points'] = loss_points
        result['expected_reward'] = points_expected_reward
    average[0] = average[0] * 0.99 + 0.01 * rew.detach()

    if loss_desc is not None:
        loss = loss * 0.5 + loss_desc
    if isinstance(loss, torch.Tensor) and torch.isnan(loss).any():
        import pdb;pdb.set_trace()
    if loss is not None:
        result['loss'] = loss
    result.update(dict(rew=rew,
                       loss_desc=loss_desc,
                       points=torch.ones(1) * len(points1),
                       average=average[0]))
    return result



def unfold_coords(action):
    result = torch.zeros(list(action.shape) + [2])
    for row in range(action.shape[0]):
        for col in range(action.shape[1]):
            # action = 0 corresponds to absense of keypoints in give square
            a = action[row, col] - 1
            if a < 0:
                # central position of 8 by 8 cell
                result[row, col] = torch.from_numpy(numpy.array([row * 8 + 4, col * 8 + 4]))
                continue
            coords = row * 8 + a // 8, col * 8 + a % 8
            result[row, col] = torch.stack(coords)
    return result


def masked(actions1, mask=1.0, new_size=100):
    coords = unfold_coords(actions1)
    mask1 = (actions1 > 0) * mask
    size = mask1.sum()
    if size > new_size:
        points = mask1.nonzero()
        arange = numpy.arange(len(points))
        numpy.random.shuffle(arange)
        excessive_points = points[arange[new_size:]]
        mask1[excessive_points[:, 0], excessive_points[:, 1]] = 0
    points1 = coords.reshape(numpy.prod(coords.shape[0:2]), 2).long()
    points1masked = points1[mask1.flatten().nonzero()].squeeze(1).to(actions1)
    return mask1.to(actions1), points1masked


def compute_loss_hom_det(heatmap1, heatmap2, batch, point_mask1,
                     point_mask2, desc1, desc2, heat1_fold, heat2_fold,
                     average=[0]):

    H = batch['H']
    H_inv = batch['H_inv']
    mask2 = batch['mask2'].squeeze()
    mask1 = batch['mask1'].squeeze()

    # use Hinv for points1 -> points2 mapping
    logprob2 = torch.log(heatmap2 + 0.0000001)
    logprob1 = torch.log(heatmap1.clone() + 0.0000001)

    args = dict(H=H_inv,
                points1=point_mask1.nonzero(),
                points2=point_mask2.nonzero(),
                point1_mask=point_mask1,
                desc1=desc1,
                desc2=desc2,
                logprob1=logprob1,
                logprob2=logprob2,
                average=average,
                img1=batch['img1'],
                img2=batch['img2'],
                use_geom=True,
                neg_reward=-2.1)

    result = process(loss_hom(**args), )
    # loss_desc = compute_loss_desc_random_points(**args)
    det_diff = compute_det_diff(H, H_inv, heatmap1, heatmap2,
                                mask1.squeeze(), mask2.squeeze())
    result['det_diff'] = det_diff.mean()
    result['loss'] = result['loss'] + result['det_diff']
    return result


def compute_det_diff(H, H_inv, heatmap1, heatmap2, mask1, mask2, weight=2000):
    assert len(heatmap1.shape) == 2
    reproj = hom.bilinear_sampling(heatmap2.unsqueeze(2).clone(), H_inv,
                                   h_template=heatmap2.shape[0],
                                   w_template=heatmap2.shape[1],
                                   to_numpy=False, mode='bilinear').to(heatmap1)

    heat1proj = hom.bilinear_sampling(heatmap1.unsqueeze(2).clone(), torch.eye(3).to(heatmap1),
                                      h_template=heatmap1.shape[0],
                                      w_template=heatmap1.shape[1],
                                      to_numpy=False, mode='bilinear').to(heatmap1)

    mask2proj = hom.bilinear_sampling(mask2.unsqueeze(2).clone(), H_inv,
                                      h_template=heatmap1.shape[0],
                                      w_template=heatmap1.shape[1],
                                      to_numpy=False, mode='nearest').to(heatmap1) * mask1

    mask1proj = hom.bilinear_sampling(mask1.unsqueeze(2).clone(), H,
                                      h_template=heatmap1.shape[0],
                                      w_template=heatmap1.shape[1],
                                      to_numpy=False, mode='nearest').to(heatmap1) * mask2

    forward_proj = hom.bilinear_sampling(heatmap1.unsqueeze(2).clone(), H,
                                         h_template=heatmap1.shape[0],
                                         w_template=heatmap1.shape[1],
                                         to_numpy=False, mode='bilinear').to(heatmap1)

    heat2proj = hom.bilinear_sampling(heatmap2.unsqueeze(2).clone(), torch.eye(3).to(heatmap1),
                                      h_template=heatmap2.shape[0],
                                      w_template=heatmap2.shape[1],
                                      to_numpy=False, mode='bilinear').to(heatmap1)
    det_diff = ((reproj[numpy_nonzero(mask2proj)] - heat1proj[numpy_nonzero(mask2proj)]) ** 2 * weight).mean() + \
               ((forward_proj[numpy_nonzero(mask1proj)] - heat2proj[numpy_nonzero(mask1proj)]) ** 2 * weight).mean()
    return det_diff


def process(*dicts):
    res = defaultdict(float)
    for key in dicts[0].keys():
        tmp = [d[key] for d in dicts if (key in d) and (d[key] is not None)]
        if len(tmp):
            res[key] = torch.mean(torch.stack(tmp))
    return res


def match_desc_reward(points1projected, points2, desc1_int,
                      desc2,
                      dist_thres=40,
                      img1=None, img2=None, points=None, use_means=False,
                      use_geom=True, geom_dist_average=[0.75], neg_reward=-0.5):
    if numpy.prod(points1projected.shape) == 0:
        return torch.ones((1,)).to(desc1_int) * -0.1, None

    geom_dist, ind2 = util.geom_match(points1projected, points2)
    ind2 = ind2[:,0]

    geom_dist = geom_dist[:, 0]
    # mapping points1projected -> points2 with descriptors
    # fit(desc2)
    # query(desc1)
    num = 1
    if use_means:
        num = min(len(desc2), len(desc1_int))
        assert num != 0
    dist, k2_desc = util.match_descriptors(desc2.cpu().detach(), desc1_int.cpu().detach(), num)
    assert len(geom_dist) == len(dist)
    assert len(geom_dist) == len(points1projected)
    k2_desc = k2_desc.squeeze()
    reward, sim_expected, similarity_desc, similarity_rand = desc_reward(desc1_int,
                                                                         desc2,
                                                                         dist,
                                                                         ind2,
                                                                         k2_desc,
                                                                         use_means)
    if len(k2_desc.shape) == 2:
        k2_desc = k2_desc[:, 0]
    loss_dist_correct = 1.0 - sim_expected
    bigger = (similarity_rand > 0.2).to(similarity_rand)
    similarity_rand = similarity_rand[bigger.nonzero()]
    loss_desc = loss_dist_correct.mean()
    if numpy.prod(similarity_rand.shape):
        loss_desc = loss_desc + similarity_rand.mean()
    # descriptor's match different from geometrical match, and
    # ground truth geometric distance is less-equal than threshold
    neq = (k2_desc != ind2)
    correct_idx = numpy.invert(neq * (geom_dist <= dist_thres))
    quality = torch.ones(1).to(desc1_int)
    quality_desc = torch.ones(1).to(desc1_int)
    if bool(neq.sum()):
        # any non-matches are ok to optimize
        wrong_id = (neq * (geom_dist >= 7)).nonzero()[0]
        # sim_wrong1 = similiarity_by_idx(desc1_int, desc2_int,
        #                    wrong_id.squeeze(),
        #                    k2_desc[wrong_id].squeeze())
        if len(wrong_id):
            reward[wrong_id] = neg_reward
            similarity_wrong = (desc1_int[wrong_id] * desc2[k2_desc[wrong_id].squeeze()]).sum(dim=1)
            loss_desc = loss_desc + similarity_wrong.mean()
            quality_desc = correct_idx.sum() / len(ind2)
            quality_desc = torch.ones(1).squeeze() * quality_desc
        if use_geom:
            # geom_reward = (torch.from_numpy(mean_geom).to(desc1_int) - geom_dist_average[0])
            geom_rew = sigmoid_abs(geom_dist)
            geom_rew = geom_rew * (geom_rew > 0)
            sim_dist = torch.from_numpy(geom_rew).to(reward).squeeze()
            reward = reward + sim_dist
            quality = (geom_dist < dist_thres).sum() / numpy.prod(geom_dist.shape)
            reward = reward * (reward >= 0) * quality + reward * (reward < 0)
            quality = torch.ones(1).squeeze() * quality
            # reward = reward + geom_reward

    # import pdb;pdb.set_trace()
    # matches = numpy.stack([numpy.arange(len(points1projected))[wrong_id], k2_desc[wrong_id]])
    # matches = numpy.stack([numpy.arange(len(points1projected)), k2_desc, (k2_desc != ind2).squeeze()])
    # draw_m(img1, img2, matches, points, points2)
    # loss_desc, q = desc_quality_loss(geom_match, desc2_int, desc1_int, points2, points1projected)
    return reward, loss_desc, quality.to(desc1_int), \
           quality_desc.to(desc1_int), \
           ((points1projected + points2[ind2]) / 2.0).round()[correct_idx.nonzero()]


def desc_reward(desc1_int, desc2_int, dist, ind2, k2_desc, use_means):
    similarity_desc = 1 - dist.squeeze()
    if len(k2_desc.shape) == 1:
        k2_desc = k2_desc[..., numpy.newaxis]
        similarity_desc = similarity_desc[..., numpy.newaxis]
    mask = k2_desc == ind2[..., numpy.newaxis]
    # Mask array uses True = 0; False = 1 to compute operations
    # like sum, mean etc.
    wrongs = numpy.ma.masked_array(similarity_desc, mask)
    means = wrongs.mean(axis=1)
    # corrects = numpy.ma.masked_array(similarity_desc, numpy.invert(mask)).mean(axis=1)
    # diff = corrects - means
    # compute distance
    sim_expected = (desc1_int * desc2_int[ind2]).sum(dim=1)
    similarity_rand = torch.stack([random_dist(desc1_int, desc2_int[ind2]) for _ in range(8)]).mean(dim=0)
    reward = (sim_expected - similarity_rand)
    if use_means:
        # todo: add to descriptor loss
        reward = (sim_expected - torch.from_numpy(means).to(sim_expected))
    return reward, sim_expected, similarity_desc[:, 0], similarity_rand


def loss_desc_random_points(average=[0.0], neg_reward=-0.5, **kwargs):
    pass
