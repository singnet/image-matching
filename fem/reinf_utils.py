import numpy
import torch
from fem import util
from torch.distributions import Categorical
from fem.nonmaximum import PoolingNms1

def match_3d_second_to_first(points2, points1, _3d_data, mask1, mask2, img1=None, img2=None):
    K1 = _3d_data['K1'].cpu()
    K2 = _3d_data['K2'].cpu()
    depth1 = _3d_data['depth1'].cpu()
    depth2 = _3d_data['depth2'].cpu()
    pose1 = _3d_data['pose1'].cpu()
    pose2 = _3d_data['pose2'].cpu()

    deb = False
    points2masked = points2[mask2.flatten().nonzero()].squeeze(1)
    points1masked = points1[mask1.flatten().nonzero()].squeeze(1)
    if deb:
        new_points1 = torch.from_numpy(util.project3d(K1, K2, depth1, points1masked.long(), pose1, pose2)).long()
        drawing.show_points(img2.cpu(), points2masked.cpu(), 'points2_img2', 2)
        drawing.show_points(img1.cpu(), points1masked.cpu(), 'points1_img1', 2)
        drawing.show_points(img2.cpu(), new_points1, 'points1_img2', 2)


    points_mapping = lambda keypoints, keypoints2: \
            util.points_mapping_3dv1(K1.cpu(), K2.cpu(), depth1.cpu(),
                                   pose1.cpu(), pose2.cpu(), keypoints.cpu(), keypoints2.cpu(), 240, 320)

    # compute distance from points1 projected to pose2 to point2
    # use only valid points from points2 since distance from non-points to points is used in reward computation
    if not (numpy.prod(points2masked.shape) and numpy.prod(points1masked.shape)):
        return None, None
    result_dist = numpy.zeros(len(points1), dtype=numpy.float32)
    dist, idx, in_bounds = points_mapping(points1masked,
                                          points2masked)


    # keep distance only for points inside image bounds
    result_dist[mask1.flatten().nonzero().squeeze().cpu()] = dist.squeeze() * in_bounds
    # discard points outside of image out out mask
    new_mask1 = mask1.flatten().clone()
    new_mask1[mask1.flatten().nonzero().squeeze()] = new_mask1[mask1.flatten().nonzero().squeeze()] * in_bounds
    new_mask1 = new_mask1.reshape(mask1.shape)
    return result_dist.reshape(points1.shape[0]), new_mask1



def compute_loss(actions, actions2, _3d_data, logprob, logprob2, img1=None, img2=None, average=[0]):
    coords = unfold_coords(actions)
    coords2 = unfold_coords(actions2)
    point_mask = (actions > 0)
    point_mask2 = (actions2 > 0)

    nonpoint_mask = (point_mask == 0)
    nonpoint_mask2 = (point_mask2 == 0)
    points1 = coords.reshape(30 * 40, 2).long()
    points2 = coords2.reshape(30 * 40, 2).long()

    distance1, newmask2 = match_3d_second_to_first(points1, points2,
                                               swap3d(_3d_data),
                                               img1=img2, img2=img1,
                                               mask1=point_mask2,
                                               mask2=point_mask)

    distance, newmask1 = match_3d_second_to_first(points2, points1,
                                                  _3d_data,
                                               img1=img1, img2=img2,
                                               mask1=point_mask,
                                               mask2=point_mask2)

    loss, rew = loss_reward(actions, actions2, average, distance, distance1, logprob, logprob2,
                            newmask1, newmask2)
    return loss, rew.mean()



def random_dist(desc1_int, desc2_int):
    idx = numpy.arange(0, len(desc2_int))
    numpy.random.shuffle(idx)
    rand_idx = idx - (idx == numpy.arange(0, len(desc2_int)))
    similarity_rand = (desc1_int * desc2_int[rand_idx]).sum(dim=1)
    return similarity_rand


def desc_self_diff_loss(det1, mask):
    # grid now 32 * 32 * 2 with height correspoinding to width in mask
    # grid[1, 0] = 12, 4  grid[31, 0] =  252, 4  etc.
    grid = util.desc_coords_no_homography(32, 32, 8, 8).reshape(2, 32, 32).permute(2, 1, 0)
    grid_flat = grid.reshape(32 * 32, 2).long()
    # take from image mask with grid
    new_mask = mask[grid_flat[:, 0], grid_flat[:, 1]]
    #test_small_mask(mask, new_mask)
    det_resh = det1.reshape(det1.shape[0], numpy.prod(det1.shape[1:])).T
    scaled = det_resh / torch.norm(det_resh, p=2, dim=1).unsqueeze(1)
    similarity_rand = torch.stack([random_dist(scaled, scaled) for _ in range(8)]).mean(dim=0)
    mask = similarity_rand > 0.2
    if mask.sum():
        return similarity_rand[util.numpy_nonzero(mask)].mean()
    return (similarity_rand * mask.to(similarity_rand)).mean()


def test_small_mask(mask, new_mask):
    for i in range(32):
        for j in range(32):
            assert new_mask.reshape(32, 32)[i, j] == mask[i * 8 + 4, j * 8 + 4]


def threshold_nms(keypoints_prob, pool=8, take=50):
    nms = PoolingNms1(pool)
    pooled = nms(keypoints_prob)
    if take:
        good_thresh = torch.sort(pooled.squeeze().reshape(pooled.shape[0],
                                                          numpy.prod(pooled.shape[1:])), dim=1)[0][:, -take:].min(dim=1)[0]
        good_thresh = torch.max(good_thresh, torch.ones_like(good_thresh) * 0.0000001)
        point_mask = pooled.squeeze() >= good_thresh.unsqueeze(1).unsqueeze(2).expand(keypoints_prob.shape[0], 256, 256)
    else:
        point_mask = pooled.squeeze()
    return point_mask


def sample(heatmaps):
    # Categorical uses dimention -1 to sample
    heat = heatmaps.permute(0, 2, 3, 1)
    dist = Categorical(heat)
    actions = dist.sample()
    logprob = dist.log_prob(actions)
    probs = torch.exp(logprob)
    return actions, logprob, probs
