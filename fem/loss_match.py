from fem import util
import torch


def compute_loss_matcher(heat1, heat2, _3d_data,
        point_mask1, point_mask2, desc1, desc2, glue):
    points1=point_mask1.nonzero()
    points2=point_mask2.nonzero()


    # map point to new image plane with H
    H = _3d_data['H']
    max_h, max_w = point_mask1.shape[0], point_mask1.shape[1]
    assert len(point_mask1.shape) == 2
    geom_dist, ind2, in_bounds = util.homographic_points_mapping_bounds(H, points1, points2, max_h, max_w)
    # matches are geom_dist < 4 * in_bounds

    desc1_int = util.descriptor_interpolate(desc1, 256,
                                            256, points1)

    desc2_int = util.descriptor_interpolate(desc2, 256,
                                            256, points2)

    kwargs = dict(points1=points1,
                  points2=points2,
                  desc1=desc1_int,
                  desc2=desc2_int,
                  conf1=heat1[points1[:,0], points1[:,1]],
                  conf2=heat2[points2[:,0], points2[:,1]])
    P, cost = glue.forward(kwargs)
    matches_id = ((geom_dist.squeeze() < 6) * in_bounds.cpu().numpy()).nonzero()[0].squeeze()
    targets = P[matches_id][range(len(matches_id)), ind2[matches_id].squeeze()]
    print(targets)
    return {'loss': (1 - targets).mean(), 'cost': cost}







