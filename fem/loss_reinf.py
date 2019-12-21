from fem.util import numpy_nonzero
import sklearn.neighbors

import torch

from fem import util

import numpy

from fem import hom
from fem.reinf_utils import random_dist


def randomize_points(input):
    max, _ = input.max(dim=0)
    min, _ = input.min(dim=0)
    result = torch.randint_like(input, low=-3, high=3) + input
    result[:, 0] = result[:, 0].clamp(min=min[0], max=max[0])
    result[:, 1] = result[:, 1].clamp(min=min[1], max=max[1])
    return result


def draw_reward(H, desc1, desc2, img1, img2, points1, points1projected, points2):
    import cv2
    from fem.hom import create_grid_batch
    cv2.imshow('img1', img1.cpu().numpy() / 255.0)

    # points1 = util.swap_rows(create_grid_batch(1, 64, 64)[0][:-1]).T.long() * 4
    in_bounds, points1projected = project_points(H, torch.ones((64, 64)), points1)
    points1, points1projected = ensure_2d_points(in_bounds, points1, points1projected)
    points2 = randomize_points(points1projected)
    desc1_int = util.descriptor_interpolate(desc1, 256,
                                            256, points1)
    desc2_int = util.descriptor_interpolate(desc2, 256,
                                        256, points2)

    get_reward(desc1_int, desc2_int, img1, img2, points1, points1projected, points2, 'rew', use_means=True, use_geom=True, dist_thres=4)
    get_reward(desc1_int, desc2_int, img1, img2, points1, points1projected, points2, 'rew2', use_means=True, use_geom=False, dist_thres=4)
    cv2.waitKey(100)
    import pdb;pdb.set_trace()


def get_reward(desc1_int, desc2_int, img1, img2, points1, points1projected, points2, name, use_means, use_geom,
               dist_thres=35):
    import cv2

    reward, loss_desc, quality, k2 = match_desc_reward(points1projected, points2,
                                                       desc1_int, desc2_int,
                                                       dist_thres=dist_thres, use_geom=use_geom,
                                                       img1=img1, img2=img2, use_means=use_means)
    reward = reward / reward.max()
    rew = torch.zeros((256, 256)).to(reward)
    rew[points1[:, 0], points1[:, 1]] = reward
    cv2.imshow(name, rew.cpu().numpy())


def distance_reward(actions, actions2,  average, distance, distance1, logprob, logprob2, newmask1, newmask2):
    if distance is None or distance1 is None:
        rew = - torch.tensor(0.01).to(logprob.device)
        return -rew, rew
    else:
        rew1 = distance_reward_for_points(actions, distance, newmask1)
        rew2 = distance_reward_for_points(actions2, distance1, newmask2)
    # negate because torch does minimization
    # this 'loss' is expected reward with should be maximized
    # todo: use prob?
    l1, rew1 = extract_loss(logprob, newmask1, rew1, average[0])
    l2, rew2 = extract_loss(logprob2, newmask2, rew2, average[0])
    rew = (rew1.mean() + rew2.mean()) * 0.5
    average[0] = 0.95 * average[0] + 0.05 * rew
    loss = 0.5 * (l1 + l2)
    return loss, rew


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


def reward(distance):
    return torch.from_numpy(sigmoid_abs(distance))


def distance_reward_for_points(actions, distance, point_mask):
    distance = distance.reshape(actions.shape)
    rew = reward(distance).to(actions.device)
    # no_poin_weight = point_mask.sum().float() / torch.max(torch.ones(1).squeeze(),
    #                                                      nonpoint_mask.sum())
    # for non-points the further nearest point the better
    non_point_reward = rew * -1 * 0.01
    rew = rew.float() * point_mask.float()# + non_point_reward.float() * nonpoint_mask.float()
    return rew


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
    in_bounds, points1projected = project_points(H, point1_mask, points11)
    points1, points1projected = ensure_2d_points(in_bounds, points11, points1projected)

    # compute descriptors
    # use only points whose projections are in bounds
    desc1 = kwargs['desc1']
    desc1_int = util.descriptor_interpolate(desc1, 256,
                                            256, points1)


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
        # to points1
        # in_bounds, points1pr = project_points(H.inverse(), None, points2)
        # desc1_int = util.descriptor_interpolate(desc1, 256,
        #                                          256, points1pr)
        # _, loss_desc, _, _, _ = match_desc_reward(points2,
        #                                            points2,
        #                                            desc1_int, desc2_int,
        #                                            img1=img1,
        #                                            use_geom=kwargs.get('use_geom', False),
        #                                            use_means=True,
        #                                            img2=img2,
        #                                            points=points1,
        #                                            dist_thres=3,
        #                                            neg_reward=neg_reward)

    deb = True
    deb = False
    if deb:
        import cv2
        cv2.imshow('img1', (img1.cpu() / 256).numpy())
        cv2.waitKey(500)
        drawing.show_points(img1.cpu() / 256, points1.cpu(), 'points1_img1', 2)
        drawing.show_points(img2.cpu() / 256, points1projected.cpu(), 'points1_img2', 2)

    result = dict()
    result['quality'] = quality.squeeze()
    result['quality_desc'] = quality_desc.squeeze()
    rew = reward.mean()

    loss = 0.0
    non_point_mask = (point1_mask == 0)
    logprob1 = kwargs.get('logprob1')
    logprob2 = kwargs.get('logprob2')
    if bool(point1_mask.sum()):
        if kwargs.get('use_geom', False) and numpy.prod(means.shape):
            means = means.long()

            H_inv = numpy.linalg.inv(H.cpu())
            loss_points = -logprob2[means.long()[:,0], means.long()[:,1]].mean()
            in_bounds, coords1 = project_points(H_inv, None, means)
            if numpy.prod(coords1.shape):
                if len(coords1.shape) == 1:
                    coords1 = coords1.unsqueeze(0)
                loss_points = loss_points / 2.0 - logprob2[coords1[:, 0], coords1[:, 1]].mean()
            else:
                loss_points = (-logprob1[numpy_nonzero(point1_mask)] * (reward)).mean()
        else:
            loss_points = (-logprob1[numpy_nonzero(point1_mask)] * (reward)).mean()
            # points2_mask = kwargs['point2_mask']
            # loss_points = -((logprob1[numpy_nonzero(point1_mask)] + logprob2[numpy_nonzero(points2_mask)][k2]) * (reward)).mean()
        loss = loss_points + loss
        points_expected_reward = (torch.exp(logprob1)[numpy_nonzero(point1_mask)] * reward).mean()
        result['loss_points'] = loss_points
        result['expected_reward'] = points_expected_reward
    average[0] = average[0] * 0.99 + 0.01 * rew.detach()

    if loss_desc is not None:
        loss = loss * 0.5 + loss_desc
    if torch.isnan(loss).any():
        import pdb;pdb.set_trace()
    result.update(dict(loss=loss.squeeze(),
                       rew=rew,
                       loss_desc=loss_desc,
                       points=torch.ones(1) * len(points1),
                       average=average[0]))
    return result


def ensure_2d_points(in_bounds, points, points1projected):
    points = points[in_bounds.nonzero()].squeeze()
    if len(points1projected.shape) == 1:
        points1projected = points1projected.unsqueeze(0)
        points = points.unsqueeze(0)
    return points, points1projected


def project_points(H, point_mask, points):
    points1projected = util.compute_new_coords(H,
                                               torch.cat([torch.zeros_like(points),
                                                          points], dim=1)[:, 1:].float()).round().long()
    in_bounds = (
            (points1projected[:, 0] < 256) * (points1projected[:, 0] >= 0)
            * (points1projected[:, 1] >= 0) * (points1projected[:, 1] < 256))
    if point_mask is not None:
        point_mask.flatten()[point_mask.flatten().nonzero().squeeze()] = point_mask.flatten()[
                                                                             point_mask.flatten().nonzero().squeeze()] * in_bounds.to(
            point_mask)
    points1projected = points1projected[in_bounds.nonzero()].squeeze()
    return in_bounds, points1projected


def extract_loss(logprob, mask, reward, average):
    nz = numpy_nonzero(mask)
    l1 = (logprob[nz] * (reward.float()[nz] - average)).mean()
    return l1, reward.float()[nz]



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


def compute_loss_hom(actions, actions2, batch, logprob1,
                     logprob2, desc1, desc2, heatmap1,
                     heatmap2, img1=None, img2=None, average=[0],
                     target=None):

    H = batch['H']
    H_inv = batch['H_inv']
    # img1_reproj = hom.bilinear_sampling(img2.unsqueeze(2), H_inv, h_template=heatmap1.shape[1],
    #                               w_template=heatmap1.shape[2], to_numpy=False, mode='nearest')
    # import matplotlib.pyplot as plt
    # plt.imshow(img1_reproj.cpu().detach())
    # plt.imshow(mask_reproj.cpu().detach())
    # plt.imshow(heatmap2.detach())
    # plt.imshow(heatmap1[1].detach())
    # import cv2
    # cv2.imshow('img2', img2.numpy() / 255.)
    # cv2.imshow('img1', img1.numpy() / 255.)
    # cv2.imshow('img2-reproj', img1_reproj.detach().numpy() / 255.0)
    # cv2.imshow('heatmap2', heatmap2[1].detach().numpy())
    # cv2.imshow("heat2-projected", new_t1.detach().numpy())
    # grid = util.desc_coords_no_homography(32, 32, 8, 8)
    # coords_after_homography = img_mask2[0, grid[1].long(), grid[0].long()] # col, row
    # masknew2 = coords_after_homography.reshape([32, 32]).to(point_mask1)
    # point_mask2, points2masked = masked(actions2, new_size=2000)
    # img_mask2 = _2d_data['mask']
    # nms = PoolingNms(8)

    mask2 = batch['mask'].squeeze()
    # points2nms = nms(heatmap2[1].unsqueeze(0).unsqueeze(0) * mask2).squeeze().nonzero()
    point_mask1, points1masked = masked(actions, new_size=50)
    # both ways don't seem to help
    # use Hinv for points1 -> points2 mapping
    args = dict(H=H_inv,
                points1=points1masked,
                logprob1=logprob1,
                point1_mask=point_mask1,
                logprob2=logprob2,
                desc1=desc1,
                desc2=desc2,
                average=average,
                img1=img1,
                img2=img2,
                target=target)
    result = process(loss_hom(**args), )
    # heat2_reproj = hom.bilinear_sampling(heatmap2[1].unsqueeze(2), H_inv, h_template=heatmap1.shape[1],
    #                               w_template=heatmap1.shape[2], to_numpy=False, mode='nearest').to(heatmap1)
    # # reproj = hom.bilinear_sampling(heatmap2[1].unsqueeze(2), H_inv, h_template=heatmap1.shape[1],
    # #                               w_template=heatmap1.shape[2], to_numpy=False, mode='bilinear').to(heatmap1)
    # import pdb;pdb.set_trace()
    #
    #
    #
    # mask_reproj = hom.bilinear_sampling(mask2.unsqueeze(2), H_inv, h_template=heatmap1.shape[1],
    #                               w_template=heatmap1.shape[2], to_numpy=False, mode='nearest').to(heatmap1)
    # det_diff = -(torch.log(heatmap1[1])[numpy_nonzero(mask_reproj)] * heat2_reproj[numpy_nonzero(mask_reproj)].detach()).mean()
    #

    result['det_diff'] = compute_det_diff(H, H_inv, heatmap1[1].squeeze(), heatmap2[1].squeeze(), mask2)
    result['loss'] = result['loss_points'] + result['det_diff']
    return result


def compute_loss_hom_det(heatmap1, heatmap2, batch, point_mask1,
                     point_mask2, desc1, desc2, heat1_fold, heat2_fold,
                     average=[0]):

    H = batch['H']
    H_inv = batch['H_inv']
    # img1_reproj = hom.bilinear_sampling(img2.unsqueeze(2), H_inv, h_template=heatmap1.shape[1],
    #                               w_template=heatmap1.shape[2], to_numpy=False, mode='nearest')
    # import matplotlib.pyplot as plt
    # plt.imshow(img1_reproj.cpu().detach())
    # plt.imshow(mask_reproj.cpu().detach())
    # plt.imshow(heatmap2.detach())
    # plt.imshow(heatmap1[1].detach())
    # import cv2
    # cv2.imshow('img2', img2.numpy() / 255.)
    # cv2.imshow('img1', img1.numpy() / 255.)
    # cv2.imshow('img2-reproj', img1_reproj.detach().numpy() / 255.0)
    # cv2.imshow('heatmap2', heatmap2[1].detach().numpy())
    # cv2.imshow("heat2-projected", new_t1.detach().numpy())
    # grid = util.desc_coords_no_homography(32, 32, 8, 8)
    # coords_after_homography = img_mask2[0, grid[1].long(), grid[0].long()] # col, row
    # masknew2 = coords_after_homography.reshape([32, 32]).to(point_mask1)
    # point_mask2, points2masked = masked(actions2, new_size=2000)
    # img_mask2 = _2d_data['mask']
    # nms = PoolingNms(8)

    mask2 = batch['mask'].squeeze()
    mask1 = torch.ones_like(heatmap1)

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
    det_diff = compute_det_diff(H, H_inv, heatmap1, heatmap2, mask2.squeeze())
    result['det_diff'] = det_diff.mean()
    result['loss'] = result['loss'] + result['det_diff']
    return result


def compute_det_diff(H, H_inv, heatmap1, heatmap2, mask2):
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
                                      to_numpy=False, mode='nearest').to(heatmap1)

    forward_proj = hom.bilinear_sampling(heatmap1.unsqueeze(2).clone(), H,
                                         h_template=heatmap1.shape[0],
                                         w_template=heatmap1.shape[1],
                                         to_numpy=False, mode='bilinear').to(heatmap1)

    heat2proj = hom.bilinear_sampling(heatmap2.unsqueeze(2).clone(), torch.eye(3).to(heatmap1),
                                      h_template=heatmap2.shape[0],
                                      w_template=heatmap2.shape[1],
                                      to_numpy=False, mode='bilinear').to(heatmap1)
    det_diff = ((reproj[numpy_nonzero(mask2proj)] - heat1proj[numpy_nonzero(mask2proj)]) ** 2 * 2000).mean() + \
               ((forward_proj[numpy_nonzero(mask2)] - heat2proj[numpy_nonzero(mask2)]) ** 2 * 2000).mean()
    return det_diff


def process(*dicts):
    res = dict()
    for key in dicts[0].keys():
        tmp = [d[key] for d in dicts if (key in d) and (d[key] is not None)]
        if len(tmp):
            res[key] = torch.mean(torch.stack(tmp))
    return res

def match_points_reward(points1projected, points2, desc1_int, desc2_int_proj,dist_thres=4):
    import sklearn
    # match geometrically
    # fit(points2)
    tree = sklearn.neighbors.KDTree(points2.cpu(),
                                    leaf_size=6)

    # mapping points1projected -> points2
    # query(points1)
    geom_dist, ind2 = tree.query(points1projected.cpu(), min(len(points1projected), 10))
    ind2 = ind2.squeeze()
    mean_geom = geom_dist[:, 1:].mean(axis=1)
    mean_geom = mean_geom / 20

    geom_dist = geom_dist[:, 0]
    ind2 = numpy.arange(len(desc1_int))



def match_desc_reward(points1projected, points2, desc1_int,
                      desc2,
                      dist_thres=40,
                      img1=None, img2=None, points=None, use_means=False,
                      use_geom=True, geom_dist_average=[0.75], neg_reward=-0.5):
    if numpy.prod(points1projected.shape) == 0:
        return torch.ones((1,)).to(desc1_int) * -0.1, None
    import sklearn
    # match geometrically
    # fit(points2)
    tree = sklearn.neighbors.KDTree(points2.cpu(),
                                    leaf_size=6)


    # mapping points1projected -> points2
    # query(points1)
    geom_dist, ind2 = tree.query(points1projected.cpu(), min(len(points1projected), 10))
    ind2 = ind2[:,0]

    mean_geom = geom_dist[:, 1:].mean(axis=1)
    mean_geom = mean_geom / 20

    geom_dist = geom_dist[:, 0]
    # mapping points1projected -> points2 with descriptors
    # fit(desc2)
    # query(desc1)
    num = 1
    if use_means:
        num = 5
    dist, k2_desc = util.match_descriptors(desc2.cpu().detach(), desc1_int.cpu().detach(), num)

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
        wrong_id = neq.nonzero()[0]
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
    return reward, loss_desc, quality.to(desc1_int), quality_desc.to(desc1_int), ((points1projected + points2[ind2]) / 2.0).round()[correct_idx.nonzero()]


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
