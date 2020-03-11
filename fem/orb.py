import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from fem.reinf_utils import threshold_nms_dense
from fem import drawing


class Orb:
    def __init__(self, nfeatures=3000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def compute(self, img):
        """
        Compute orb keypoints
        :param img: numpy.array
        :return: numpy.array
        """
        # find the keypoints with ORB
        kp = self.orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = self.orb.compute(img, kp)
        result = img.copy()
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(img, kp, result, color=(0, 255, 0), flags=0)
        img3 = np.zeros(img.shape).astype(np.float32)
        i = len(kp) + 1.0
        for k in kp:
            img3[np.int(k.pt[1]), np.int(k.pt[0])] = i / 3000.
            i -= 1
        mask = threshold_nms_dense(torch.from_numpy(img3).unsqueeze(0).unsqueeze(0), pool=16, take=None)
        return mask


if __name__ == '__main__':
    img = cv2.imread('/mnt/fileserver/shared/datasets/at-on-at-data/COCO/val2014/COCO_val2014_000000000073.jpg',0)
    orb = Orb()
    points = orb.compute(img)
    drawing.show_points(img, points, 'mask')
    import pdb;pdb.set_trace()
