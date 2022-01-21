import torch
import cv2
import sklearn.neighbors
import numpy as np
import sklearn
import numpy
import ignite
import torch.nn.functional as F
from fem.hom import create_grid_batch, apply_h_to_grid


def init_weights(self):
    def init_weights(m):
        if hasattr(m, 'weight') and not isinstance(m, (torch.nn.BatchNorm2d,
                                                       torch.nn.BatchNorm1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    self.apply(init_weights)


def numpy_nonzero(tensor):
    if isinstance(tensor, numpy.ndarray):
        return tensor.nonzero()
    return torch.unbind(tensor.nonzero(), 1)


def desc_coords_no_homography(h, w, factor_height, factor_width):
    """
    Create grid 32x32 and rescale by factors

    :param factor_height: int
    :param factor_width: int
    :return: torch.Tensor
        tensor with descript cell's centers 2 x 1024
        in column, row order
    """
    coords_no_homography = hom.create_grid_batch(1, h, w)[0][:-1]
    coords_no_homography[1] *= factor_height  # 8
    coords_no_homography[1] += factor_height // 2
    coords_no_homography[0] *= factor_width
    coords_no_homography[0] += factor_width // 2  # 8
    return coords_no_homography


def build_distance_matrix_loop_orig_hom(h, w, factor_x, factor_y, H):
    img_h, img_w = h * factor_y, w * factor_x
    coords = []
    # similiarity matrix is orig @ hom
    # build distance matrix len(orig) * len(hom)

    result = numpy.zeros((1024, 1024), dtype=numpy.float)
    for row in range(32):
        for col in range(32):
            for row2 in range(32):
                for col2 in range(32):

                    # original coordinates
                    col_mh, rom_mh = coord_to_homog(H, row, col)
                    # coordinates or a descriptor on homographic image
                    col_m2 = col2 * 8 + 4
                    rom_m2 = row2 * 8 + 4
                    d = ((col_m2 - col_mh) ** 2 + (rom_m2 - rom_mh) ** 2) ** 0.5
                    # result is 1024 x 1024
                    # fist row is distance from hom[0, 0] to all points in original 32x32 map
                    # second row is distance from hom[1, 0] to all points in original 32x32
                    # columnwise orig[0, 0] orig[0,1] orig[0,2]
                    # print("result[{0}, {1}]".format(row * 32 + col, row2 * 32 + col2))
                    # print("dist from {0} to to {1}".format((row, col), (row2, col2)))
                    # print("dist from {0} to to {1}".format((rom_mh, col_mh), (rom_m2, col_m2)))

                    result[row * 32 + col, row2 * 32 + col2] = d
    return result


def coord_to_homog(H, row, col):
    """
    Compute coordinates after homographic transformation
    :param H: Homography 3x3
    :param row: float
    :param col: float
    :return: Tuple[int, int]
    """
    row_m, col_m  = unfold_256_256(row, col)
    hom_coords = H @ [col_m, row_m, 1]
    # translate original point to homographic image
    col_mh, rom_mh = hom_coords[:2] / hom_coords[-1]
    return col_mh, rom_mh


def unfold_256_256(row, col):
    row_m = row * 8 + 4
    col_m = col * 8 + 4
    return row_m, col_m


def build_distance_matrix(h, w, factor_x, factor_y, H_inv):
    img_h, img_w = h * factor_y, w * factor_x
    grid = hom.create_grid_batch(1, img_h, img_w) # it is 1x3*N where 3 = col, row, normalizer
    h_grid = hom.apply_h_to_grid(H_inv, grid.double(), img_h, img_w).float()[0] # it is row x col x 2
    # no-homography coordinates
    coords_no_homography = desc_coords_no_homography(h, w, factor_x, factor_y) # col, row

    coords_after_homography = h_grid[coords_no_homography[1].long(), coords_no_homography[0].long()] # col, row
    coords_no_swapped = swap_rows(coords_no_homography).transpose(1, 0) # row, col
    coords_after_swapped = swap_rows(coords_after_homography.transpose(1, 0)).transpose(1, 0) # row, col
    # distance of original coordinate points to
    # homography applied points
    dist_orig_homog = sklearn.metrics.pairwise_distances(coords_after_swapped,
                                                         coords_no_swapped)
    return dist_orig_homog


def draw_dist_map(dist_mat, idx=0):
    from matplotlib import pyplot as plt
    plt.imshow(dist_mat.reshape((1024, 32, 32))[idx] / dist_mat.max())
    plt.show()


def test(sim, orig, hom, H_inv, dist_map):
    """
    Iterate over distance matrix for (0, 19) and compute distance
    """

    col = 19
    row = 0
    # (0, 1) is 4, 12
    assert (12, 4) == unfold_256_256(1, 0)
    col_mh, row_mh = coord_to_homog(H_inv, row=row, col=col)
    import pdb;pdb.set_trace()
    for row2 in range(32):
        for col2 in range(32):
            # compare orig[1, 0] with hom[0,0] hom[0,1]... hom[1, 0]..
            row_2m, col_2m = unfold_256_256(row2, col2)
            assert (sim[0][row * 32 + col][row2 * 32 + col2] - hom[0, :, row2, col2] @ orig[0,:, row, col]) < 0.0001
            print("distance from ({0}, {1}) to ({2}, {3})".format(row, col, row2, col2))
            print("distance from ({0}, {1}) to ({2}, {3})".format(row_mh, col_mh, row_2m, col_2m))

            d = ((col_2m - col_mh) ** 2 + (row_2m - row_mh) ** 2) ** 0.5
            print(d)
            value = (sim * dist_map)[0].reshape((1024, 32, 32))[row * 32 + col][row2, col2]
            if d <= 7.0:
                assert abs(value) > 0.0001
            else:
                assert abs(value) <= 0.0000001


def reshape_dist(dist_mat_s, h, nearest, w):
    dist_mat_s = dist_mat_s
    dist_mat_s = torch.from_numpy(
        dist_mat_s.astype(numpy.float32)).to(
        nearest.device)
    return dist_mat_s


def output_transform_precission_recall(output,
                                       print_=False):
    """
    Extract and flatten layers corresponding to keypoints
    """
    target = output['det_target']
    if not isinstance(output['det_target'], torch.Tensor):
        target = torch.from_numpy(target)
    prob = output['det_prob']
    if not isinstance(prob, torch.Tensor):
        prob = torch.from_numpy(prob)
    assert (prob.min() >= 0.0)
    assert len(prob.shape) == 3

    shape = target.shape
    batch = shape[0]
    item_length = numpy.prod(shape[-2:])
    y_pred = prob.reshape((batch, item_length)).contiguous() > 0.2
    y = target.reshape((batch, item_length)).contiguous() > 0.2
    if print_:
        nonz = y.nonzero()
        selected_pred = y_pred[nonz[:, 0], nonz[:, 1]]
        selected_y = y[nonz[:, 0], nonz[:, 1]]
        prec = (selected_pred == selected_y).sum().float() / numpy.prod(selected_pred.shape)
        print(prec)
    return y_pred.long(), y.long()


def on_iteration_completed(engine, print_every=100):
    iteration = engine.state.iteration
    if iteration % print_every == 0:
        epoch = engine.state.epoch
        loss = engine.state.output.get('loss', '')
        lr = engine.state.output.get('lr', '')
        print("Iteration: {}, Loss: {}, lr: {}".format(iteration, loss, lr))
        print("metrics {0}".format({k: float(v) for
                                    k, v in engine.state.metrics.items()}))


def add_metrics(engine, average=False, print_every=100):
    """
    Add metrics to the engine, and iteration completed callbacks

    Metrics will process engine.state.output to compute precession, recall
    :param engine: ignite.Engine
    :param average: bool
        if True compute running average of precesion and recall
    :return:
    """
    prec = ignite.metrics.Precision(output_transform=output_transform_precission_recall, average=average)
    recall = ignite.metrics.Recall(output_transform=output_transform_precission_recall, average=average)
    prec.attach(engine, 'precision')
    recall.attach(engine, 'recall')
    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, lambda eng: prec.completed(eng, 'precision'))
    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, lambda eng: recall.completed(eng, 'recall'))
    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, lambda eng: on_iteration_completed(eng,
                                                                                                          print_every=print_every))


def add_moving_average(engine, name, decay=0.99):
    metric = ignite.metrics.RunningAverage(output_transform=lambda x: x[name],
                                           alpha=decay)
    def callback(engine):
        out = engine.state.output
        if name in out:
            metric.iteration_completed(engine)
            metric.completed(engine, 'average_' + name)

    engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED,
                             callback)


def match_descriptors(desc1_keypoints, desc2_keypoints, num=1):
    """
    Match descriptors

    :param desc1_keypoints: numpy.array
        N * descriptor length
    :param desc2_keypoints: numpy.array
        N * descriptor length
    :return: indices of matches of desc2_keypoints in desc1_keypoints
    """
    tree_cos = sklearn.neighbors.NearestNeighbors(n_neighbors=num,
                                                  leaf_size=10,
                                                  metric='cosine')
    tree_cos.fit(desc1_keypoints)
    dist_cos, ind = tree_cos.kneighbors(desc2_keypoints, min(num, len(desc1_keypoints)))
    assert len(dist_cos) == len(ind)
    return dist_cos, ind


def match_wrapper(desc1, desc2):
    """
    wrapper around match_descriptors - returns same result in different format
    :param desc1: numpy.array
        N * descriptor length
    :param desc2: numpy.array
        N * descriptor length
    :return: numpy.array
        len(desc2) * 3
    """
    matches1 = numpy.zeros((3, len(desc2)))
    dist, ind = match_descriptors(desc1, desc2)
    dist = dist.squeeze()
    ind = ind.squeeze()
    for i in range(len(desc2)):
        # index in desc1
        matches1[0, i] = ind[i]
        # index in desc2
        matches1[1, i] = i
        matches1[2, i] = dist[i]
    return matches1


def swap_rows(x):
    if isinstance(x, torch.Tensor):
        y = torch.zeros_like(x)
    else:
        y = numpy.zeros_like(x)
    y[[1, 0]] = x
    if numpy.prod(y.shape):
        assert y[0].max() == x[1].max()
    return y


def interpolate(H, W, coarse_desc, samp_pts, normalize=True, align_corners=True):
    """

    :param H: int
      height
    :param W: int
      width
    :param coarse_desc: numpy.array
        descriptors of shape (batch * D * height * width)
    :param samp_pts: numpy.array
        points of shape (2 * num points)
         with 0th dimension being width(e.g. x)
    :return:
    """
    desc = grid_sample(H, W, coarse_desc, samp_pts, align_corners=align_corners)
    # desc = F.grid_sample(coarse_desc, samp_pts)
    if normalize:
        desc_normal = F.normalize(desc, dim=1, p=2).squeeze()
        if len(desc_normal.shape) == 1:
            desc_normal = desc_normal.unsqueeze(1)
        desc = desc_normal
    return desc


def pts_normalize(coords, dim):
    return (coords / (dim / 2)) - 1.


def grid_sample_old(H, W, coarse_desc, samp_pts, align_corners=True):
    if torch.numel(samp_pts):
       assert samp_pts[0, :].max() < W
       assert samp_pts[1, :].max() < H

    samp_pts[0, :] = (samp_pts[0, :] / (W / 2.)) - 1.
    samp_pts[1, :] = (samp_pts[1, :] / (H / 2.)) - 1.
    samp_pts = samp_pts.transpose(0, 1).contiguous()
    samp_pts = samp_pts.view(1, 1, -1, 2)
    samp_pts = samp_pts.float()
    samp_pts = samp_pts.to(coarse_desc.device)
    desc = F.grid_sample(coarse_desc, samp_pts, align_corners=align_corners)
    return desc


def grid_sample(H, W, coarse_desc, samp_pts, align_corners=True):
    #if torch.numel(samp_pts):
    #    assert samp_pts[0, :].max() < W
    #    assert samp_pts[1, :].max() < H
    tmp0 = pts_normalize(samp_pts[0, :], W)
    tmp1 = pts_normalize(samp_pts[1, :], H)
    tmp = torch.stack([tmp0, tmp1])
    tmp = tmp.transpose(0, 1).contiguous()
    tmp = tmp.view(1, 1, -1, 2)
    tmp = tmp.to(coarse_desc)
    desc = F.grid_sample(coarse_desc, tmp, align_corners=align_corners)
    return desc


def descriptor_interpolate(desc, rows, cols, points, normalize=True, align_corners=True):
    """
    Interpolate descriptors for given points

    :param desc:
    :param rows:
    :param cols:
    :param points: torch.Tensor
        (points, row-and-column)
    :return:
    """
    if len(points.shape) == 1:
        points = points.unsqueeze(0)
    # else:
    #     assert points.shape[0] >= points.shape[1]
    if isinstance(points, torch.Tensor):
        p = points.float().clone().transpose(1, 0)
    else:
        p = torch.from_numpy(points.astype(numpy.float32).copy().transpose(1, 0))
    x_y_points = swap_rows(p)
    tmp = interpolate(rows, cols, desc.unsqueeze(0), x_y_points, normalize=normalize, align_corners=align_corners)
    return tmp.transpose(1, 0)


def calculate_h(pts_init, pts_pert):
    H, _ = cv2.findHomography(pts_init, pts_pert)
    H_inv, _ = cv2.findHomography(pts_pert, pts_init)
    H = H.astype(numpy.float32)
    return H, H_inv


def desc_quality_no_transform(desc1_keypoints, desc2_keypoints, keypoints, keypoints2):
    import sklearn.neighbors
    tree = sklearn.neighbors.KDTree(keypoints,
                                    leaf_size=6)
    # mapping keypoints2 -> keypoints by coordinates
    dist_points, ind_points = tree.query(keypoints2)
    max_distance = 5
    keep2 = (dist_points.squeeze() < max_distance).nonzero()
    keep1 = ind_points.squeeze()[keep2]
    desc1left = desc1_keypoints[keep1]
    desc2left = desc2_keypoints[keep2]
    if not (len(desc1left) and len(desc2left)):
        return 0.0
    dist, ind = match_descriptors(desc1left, desc2left)
    ideal_match = numpy.arange(len(keep1))
    return ((ind.flatten() == ideal_match).sum() / len(ideal_match),
            keypoints[keep1][ind.squeeze()],
            keypoints2[keep2])


def compute_new_coords(H, t1):
    t2 = torch.cat((t1, torch.ones(len(t1), dtype=t1.dtype).unsqueeze(1).to(t1.device)), dim=1)
    h0 = H.squeeze()
    grid = create_grid_batch(1, 256, 256, t2.device)
    h_grid = apply_h_to_grid(h0, grid.double(), 256, 256).float()
    new_coords = h_grid[0, t2[:, 1].round().long(), t2[:, 2].round().long()]
    new_coords = torch.stack((new_coords[:, 1], new_coords[:, 0])).transpose(0, 1)
    return new_coords


def get_pts3d(depth, K, grid):
    pts = numpy.ones((3, grid.shape[0]), dtype=numpy.float32)

    d = depth.flatten()
    d = numpy.zeros(grid.shape[0], dtype=numpy.float32)
    pts[0, :] = grid[:, 0]
    pts[1, :] = grid[:, 1]

    for i in range(grid.shape[0]):
        d[i] = depth[grid[i, 1], grid[i, 0]]

    Kinv = numpy.linalg.inv(K)

    ptsn = numpy.matmul(Kinv, pts)
    Xn = ptsn[0, :]
    Yn = ptsn[1, :]

    #  f = W / 2 / tan(FOV/2) = W/2, but since we multiplied points by Kinv it's 1
    f = 1

    pts3d = numpy.ones((3, grid.shape[0]), dtype=np.float32)
    pts3d[0, :] = (d * Xn.flatten()) / f
    pts3d[1, :] = (d * Yn.flatten()) / f
    pts3d[2, :] = d.flatten()

    return pts3d


'''
 Transfer points from view1 to view2 using ground truth depth map and camera parameters
 Input parameters:
 K1, K2 - intrinsic matrices 3x3
 pts2d - 2d int points on the first image (pts1[:, 0] - x value; pts1[:, 1] - y value)
 depth1, depth2 - depth maps
 pose1, pose2 - cameras extrinsic parameters

 Return:
 pts2d_repr - reprojected points
 pts3dh_1_to_2 - homogeneous 3d points of the first camera within the second camera coordinates
'''


def reproject_points_between_views(pts2d, K1, K2, depth1, pose1, pose2):
    pts3d = get_pts3d(depth1, K1, pts2d)
    pts3dh = numpy.ones((4, pts3d.shape[1]))
    pts3dh[0:3, :] = pts3d

    pts3dh_to_world = numpy.matmul(pose1, pts3dh)
    pose2_inv = numpy.linalg.inv(pose2)
    pts3dh_1_to_2 = numpy.matmul(pose2_inv, pts3dh_to_world)

    # Project points on camera2
    xn = (K2[0, 0] * pts3dh_1_to_2[0, :] / pts3dh_1_to_2[2, :]) + K2[0, 2]
    # xn = xn.astype(np.int32)
    yn = (K2[1, 1] * pts3dh_1_to_2[1, :] / pts3dh_1_to_2[2, :]) + K2[1, 2]
    # yn = yn.astype(np.int32)

    pts2d_repr = numpy.stack([xn, yn], axis=1)

    return pts2d_repr, pts3dh_1_to_2


def homographic_points_mapping(H_inv, keypoints, keypoints2):
    new_coords1 = compute_new_coords(H_inv,
                                         torch.cat([torch.zeros_like(keypoints),
                                                    keypoints], dim=1)[:, 1:].float()).round().long()
    tree = sklearn.neighbors.KDTree(new_coords1.cpu(),
                                    leaf_size=6)
    # mapping keypoints2 -> new_coords
    geom_dist, ind2 = tree.query(keypoints2.detach().cpu())
    return geom_dist, ind2


def homographic_points_mapping_bounds(H_inv, keypoints, keypoints2, max_h, max_w):
    new_coords1 = compute_new_coords(H_inv,
                                     torch.cat([torch.zeros_like(keypoints),
                                                keypoints], dim=1)[:, 1:].float()).round().long()
    tree = sklearn.neighbors.KDTree(keypoints2.cpu(),
                                    leaf_size=6)
    # mapping keypoints2 -> new_coords
    geom_dist, ind2 = tree.query(new_coords1.detach().cpu())
    in_bounds = (
            (new_coords1[:, 0] < max_h) * (new_coords1[:, 0] >= 0)
            * (new_coords1[:, 1] >= 0) * (new_coords1[:, 1] < max_w))
    assert len(in_bounds) == len(geom_dist)
    return geom_dist, ind2, in_bounds


def points_mapping_3d(K1, K2, depth1, pose1, pose2, keypoints, keypoints2):
    """
    Transform keypoints to imageplane of pose2, then match using nearest neighbours
    keypoints2 to keypoints with new coords, return idx alongside with distance
    """
    new_coords1 = project3d(K1, K2, depth1, keypoints, pose1, pose2)
    # build search tree for keypoints
    tree = sklearn.neighbors.KDTree(new_coords1,
                                    leaf_size=6)

    # mapping keypoints2 -> new_coords
    geom_dist, ind2 = tree.query(keypoints2)
    return geom_dist, ind2


def points_mapping_3dv1(K1, K2, depth1, pose1, pose2, keypoints, keypoints2, max_h, max_w):
    """
    Transform keypoints to imageplane of pose2, then match using nearest neighbours
    keypoints with new coords to keypoints2
    :return Tuple[tensor, tensor, tensor]
        distance, indices, array with 1 = projected point inside image, 0 = outside
    """
    new_coords1 = project3d(K1, K2, depth1, keypoints, pose1, pose2)
    # build search tree for keypoints
    tree = sklearn.neighbors.KDTree(keypoints2,
                                    leaf_size=6)

    # mapping keypoints2 -> new_coords
    geom_dist, ind2 = tree.query(new_coords1)
    in_bounds = (
            (new_coords1[:, 0] < max_h) * (new_coords1[:, 0] >= 0)
            * (new_coords1[:, 1] >= 0) * (new_coords1[:, 1] < max_w))

    return geom_dist, ind2, in_bounds


def project3d(K1, K2, depth1, keypoints, pose1, pose2):
    # keypoints are rows, cols, change to cols rows as expected by the function
    kp1 = swap_rows(keypoints.transpose(1, 0)).transpose(1, 0)
    new_coords1, _ = reproject_points_between_views(pts2d=kp1,
                                                    K1=K1, K2=K2,
                                                    depth1=depth1,
                                                    pose1=pose1,
                                                    pose2=pose2)
    # swap rows back
    new_coords1 = swap_rows(new_coords1.transpose(1, 0)).transpose(1, 0).round().astype(numpy.int32)
    return new_coords1


def geom_match(points1, points2, num=2):

    if isinstance(points2, torch.Tensor):
        points2 = points2.cpu()
        points1 = points1.cpu()

    # match geometrically
    # fit(points2)
    tree = sklearn.neighbors.KDTree(points2,
                            leaf_size=6)


    # mapping points1projected -> points2
    # query(points1)
    geom_dist, ind2 = tree.query(points1, min(len(points2), len(points1), num))
    return geom_dist, ind2


def project_points(H, point_mask, points):
    points1projected = compute_new_coords(H,
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
    if numpy.prod(points1projected.shape):
        if len(points1projected.shape) == 1:
            points1projected = points1projected.unsqueeze(0)
    return in_bounds, points1projected


def project_points2(H, point_mask, points, max_h, max_w):
    pt1 = torch.cat([points, torch.ones([points.shape[0], 1]).to(points)], dim=1)
    pt1_proj = (H.to(pt1) @ pt1.T).T
    points1projected = pt1_proj[:, :2] / pt1_proj[:, -1].unsqueeze(1)
    in_bounds = gen_in_bounds_mask(max_h, max_w, points1projected)
    return in_bounds, points1projected


def gen_in_bounds_mask(max_h, max_w, points1projected):
    in_bounds = (
            (points1projected[:, 0] < max_h) * (points1projected[:, 0] >= 0)
            * (points1projected[:, 1] >= 0) * (points1projected[:, 1] < max_w))
    return in_bounds


def get_points_in_bounds(in_bounds, points, points1projected):
    points = points[in_bounds.nonzero()].squeeze()
    if len(points1projected.shape) == 1:
        points1projected = points1projected.unsqueeze(0)
        points = points.unsqueeze(0)
    return points, points1projected


def remove_module(state_dict):
    result = dict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            k = k.split('module.')[1]
        result[k] = v
    return result


def mean(lst):
    if len(lst) == 0:
        return 0
    return torch.mean(torch.stack(lst))


def iterative_mean(prev_mean, t, observation):
    """
    Iterative mean

    see https://www.heikohoffmann.de/htmlthesis/node134.html
    """
    return prev_mean + (1  / (t + 1)) * (observation - prev_mean)

