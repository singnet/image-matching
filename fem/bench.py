import numpy
import torch
from fem import util
import cv2


def get_points_desc(fe, sp, device, img, thresh):
    if fe is not None:
        pts_2, desc_2, heatmap_2 = fe.run(img.astype('float32').squeeze() / 255.)
    else:
        with torch.no_grad():
            pts_2, desc_2 = sp.to(device).points_desc(torch.from_numpy(img).to(device), threshold=thresh)
        pts_2 = pts_2.T
        desc_2 = desc_2[0].T.cpu().detach().numpy()
        pts_2 = numpy.concatenate([util.swap_rows(pts_2[:2]), pts_2[2, :][numpy.newaxis, :]])
    return pts_2, desc_2


def preprocess(img_2_src, to_gray=cv2.COLOR_RGB2GRAY):
    img_2 = cv2.cvtColor(img_2_src, to_gray)
    timg2 = numpy.expand_dims(numpy.expand_dims(img_2.astype('float32'), axis=0), axis=0)
    return timg2



def ensure_3d_image(img_1_src):
    if len(img_1_src.shape) == 2:
        img_1c = numpy.repeat(numpy.expand_dims(img_1_src.astype('uint8'), axis=2), 3, axis=2).squeeze().transpose(1, 2, 0)
    else:
        img_1c = img_1_src
    return img_1c


def draw_matches(matches, pts_1, pts_2, img_1_src, img_2):
    IMG_SIZE_MAX = [max(img_1_src.shape[0], img_2.shape[0]), max(img_1_src.shape[1], img_2.shape[1])]
    img_output = numpy.zeros(shape=(2 * IMG_SIZE_MAX[0], img_1_src.shape[1] + img_2.shape[1], 3), dtype=numpy.uint8)
    img_1c = ensure_3d_image(img_1_src)
    img_2 = ensure_3d_image(img_2)
    img_output[IMG_SIZE_MAX[0]:IMG_SIZE_MAX[0] + img_1_src.shape[0], :img_1_src.shape[1], :] = img_1c.squeeze()
    img_output[IMG_SIZE_MAX[0]:IMG_SIZE_MAX[0] + img_2.shape[0], img_1_src.shape[1]:, :] = img_2
    import drawing
    img_pair = drawing.draw_matches(matches, pts_1, pts_2, img_output[IMG_SIZE_MAX[0]:])
    return img_pair


def replication_ratio(pts1_proj, pts2, threshold):
    """
    find how many of pts2 are replicated in pts1_proj given threshold

    :param pts2: numpy.array with shape 2xN
    :param pts1_proj: numpy.array with shape 2xM
    :param threshold: threshold for good correspondence
    :return:
    """
    assert type(pts2) == type(pts1_proj)
    from fem.util import geom_match
    dist, ind = geom_match(pts1_proj[:2, :].T, pts2[:2, :].T, num=1)
    result = (dist <= threshold).sum() / pts1_proj.shape[1]
    return result


def coverage(img1, mask, matches, pt1):
    if mask is None:
        mask = numpy.ones_like(img1)
    coverage_mask = numpy.zeros_like(img1)
    is_error = matches[2]
    pt_round = numpy.round(pt1.T).astype(numpy.int16)
    for i in range(len(is_error)):
        if not is_error[i]:
            point = tuple(pt_round[i])
            cv2.circle(coverage_mask, point, 25, (255, 255, 255), cv2.FILLED)
    intersect = coverage_mask * mask * 255
    coverage = (intersect > 0).sum() / (mask > 0).sum()
    return coverage, coverage_mask, intersect


def harmonic(*args):
    return len(args) / numpy.sum(1 / x for x in args)

