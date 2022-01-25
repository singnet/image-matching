import util
import torch
from torch.nn import functional as F
import numpy
from fem.loss_reinf import match_desc_reward
import drawing


def compute_loss_desc(batch,
                point1_mask, point2_mask,
                desc1, desc2, normalize=True):
    """
    Interpolated discriptors at point_mask locations and match them
    by their 2d coordinates after homographic projection
    """
    H1 = batch['H1_inv']
    H2 = batch['H2_inv']
    points1 = point1_mask.nonzero()
    points2 = point2_mask.nonzero()
    # project points from unmodified images to corresponding
    in_bounds1, points1img1 = util.project_points(H1, point1_mask, points1)
    in_bounds2, points2img2 = util.project_points(H2, point2_mask, points2)

    # project only valid points
    in_bounds1_2, points1img2 = util.project_points(H2, point1_mask, points1[in_bounds1])
    img1 = batch.get('img1')
    img2 = batch.get('img2')
    deb = False
    if deb:
        import cv2
        cv2.imshow('img1', (img1.cpu() / 256).numpy())
        drawing.show_points(img1.cpu().numpy() / 256, points1projected.cpu(), 'points1_img1', 2)

        drawing.show_points(img2.cpu().numpy() / 256, points2projected.cpu(), 'points1_img2', 2)
        cv2.waitKey()
    # now the task is to extract descriptors
    # at projected point locations and match
    desc1_int = util.descriptor_interpolate(desc1, 256, 256,  points1img1[in_bounds1_2], normalize)
    assert len(desc1_int) == len(points1img2)
    desc2_int = util.descriptor_interpolate(desc2, 256, 256, points2img2, normalize)

    if normalize:
        result = match_desc_reward_norm(points1img1, points1img2, points2img2,
                desc1_int, desc2_int, batch['img1'], batch['img2'])
    else:
        result = match_desc_reward_m(points1img2, points2img2, desc1_int, desc2_int, desc2_int_wrong)
    return result

def match_desc_reward_norm(points1img1,
                           points1img2,
                           points2img2,
                           desc1_int,
                           desc2_int,
                           img1,
                           img2):
    reward, loss_desc, quality, quality_desc, means = match_desc_reward(points1img2,
                         points2img2,
                         desc1_int,
                         desc2_int,
                         img1=img1,
                         img2=img2,
                         use_geom=True,
                         use_means=True,
                         dist_thres=4,
                         points=points1img1)

    result = {'loss_desc': loss_desc,
            'quality_desc': quality_desc}
    return result


def match_desc_reward_m(points1img2,
                      points2img2,
                      desc1_int,
                      desc2_int,
                      desc2_int_wrong):
    if numpy.prod(points1img2.shape) == 0:
        return torch.ones((1,)).to(desc1_int) * -0.1, None
    geom_dist, ind2 = util.geom_match(points1img2, points2img2)
    ind2 = ind2[:,0]
    geom_dist = geom_dist[:, 0]

    # mapping points1img2 -> points2 with descriptors
    # fit(desc2)
    # query(desc1)
    num = 1
    dist, k2_desc = util.match_descriptors(desc2_int.cpu().detach(), desc1_int.cpu().detach(), num,
                                           metric='euclidean')
    k2_desc = k2_desc.squeeze()
    assert len(geom_dist) == len(dist)
    assert len(geom_dist) == len(points1img2)

    # minimize descriptor difference
    dist_expected = F.pairwise_distance(desc1_int, desc2_int[ind2], 2)
    # alternative losses - quadratic and exponential
#    loss_expected = (torch.exp(dist_expected.clone()) - 1).mean()
#    loss_expected = F.mse_loss(dist_expected, torch.zeros_like(dist_expected))
    filtered = dist_expected[dist_expected > 0.1]
    loss_expected = torch.zeros(1).squeeze().to(dist_expected)
    if numpy.prod(filtered.shape):
        loss_expected = (filtered).mean()
    loss_desc = loss_expected.clone()
    neq = (k2_desc != ind2)
    wrong_id = neq.nonzero()[0]
    loss_wrong = torch.zeros(1).squeeze().to(loss_desc)
    dist_wrong = torch.zeros(1).squeeze().to(loss_desc)
    if len(wrong_id):
        dist_wrong = F.pairwise_distance(desc1_int[wrong_id], desc2_int[k2_desc[wrong_id].squeeze()])
        # exponential loss works much better than linear
        loss_wrong = (torch.exp(1 / (1 + dist_wrong)) - 1).mean()
        loss_desc += loss_wrong
    result = dict()

    loss_random = torch.zeros(1).squeeze().to(loss_desc)
    dist_random = F.pairwise_distance(desc1_int, desc2_int[ind2 - 1], 2)
    if sum(dist_random.shape):
        loss_random = (torch.exp(1 / (1 + dist_random)) - 1).mean()
        loss_desc += loss_random
        result['loss_rand'] = loss_random

    quality_desc = numpy.invert(neq).sum() / len(ind2)
    quality_desc = torch.ones(1).squeeze() * quality_desc
    if torch.isnan(loss_desc).any():
        import pdb;pdb.set_trace()
    result.update({'loss_desc': loss_desc, 'quality_desc': quality_desc,
            'dist_expect': dist_expected.clone().mean(),
            'dist_wrong': dist_wrong.clone().mean(),
            'loss_desc1': loss_expected.clone().mean(),
            'loss_desc2': loss_wrong,})
    return result

