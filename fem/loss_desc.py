import util
from fem.loss_reinf import match_desc_reward
import drawing


def compute_loss_desc(batch,
                point1_mask, point2_mask,
                desc1, desc2):
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
    desc1_int = util.descriptor_interpolate(desc1, 256, 256,  points1img1[in_bounds1_2])
    assert len(desc1_int) == len(points1img2)
    desc2_int = util.descriptor_interpolate(desc2, 256, 256, points2img2)
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
    result = dict(loss_desc=loss_desc,
                  quality_desc=quality_desc)
    return result
