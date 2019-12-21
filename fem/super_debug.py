import cv2
import numpy
import sklearn.neighbors


def draw_desc_interpolate(img_source, img_h, keypoints, keypoints2,
                          desc1_keypoints, desc2_keypoints, idx=0, skip_matches=False):



    # tree = sklearn.neighbors.KDTree(desc1_keypoints.detach(),
    #                                 leaf_size=6)
    # dist, ind = tree.query(desc2_keypoints.detach())

    img_pair = numpy.hstack([img_source, img_h]).astype(numpy.uint8)
    img_pair = numpy.stack([img_pair, img_pair, img_pair], axis=2).squeeze()
    pts1 = numpy.vstack([keypoints[:, 1], keypoints[:, 0]])
    pts2 = numpy.vstack([keypoints2[:, 1], keypoints2[:, 0]])
    tree_cos = sklearn.neighbors.NearestNeighbors(n_neighbors=1,
                                                  leaf_size=10,
                                                  metric='cosine')
    tree_cos.fit(desc1_keypoints)
    dist_cos, ind = tree_cos.kneighbors(desc2_keypoints, 1)
    ind_desc1 = numpy.arange(len(desc1_keypoints))[ind]
    matches = numpy.stack([ind_desc1.flatten(),
                           numpy.arange(len(ind))])

    draw_matches(idx, img_pair, matches, pts1, pts2, skip_matches=skip_matches)
    return pts1, pts2, matches


def draw_matches(idx, img_pair, matches, pts1, pts2, skip_matches=False):
    import drawing
    img5 = drawing.draw_matches(matches,
                                pts1=pts1,
                                pts2=pts2,
                                imgpair=img_pair,
                                skip_match=skip_matches)
    cv2.namedWindow(str(idx), cv2.WINDOW_NORMAL)
    cv2.imshow(str(idx), img5)
    cv2.resizeWindow(str(idx), img5.shape[1] * 2, img5.shape[0] * 2)
    cv2.waitKey(1000)
