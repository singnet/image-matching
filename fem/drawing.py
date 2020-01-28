import random
import cv2
import numpy


def draw_points(pts, img, iscolor=True):
    r = 0
    g = 255
    b = 0
    for i in range(pts.shape[1]):
        pt = (int(round(pts[0,i])), int(round(pts[1, i])))
        if iscolor:
            r = random.randint(0, 32767) % 256
            g = random.randint(0, 32767) % 256
            b = 0 if (r + g > 255) else ( 255 - (r + g))

        color = (b, g, r)

        cv2.circle(img, pt, 2, color=color, thickness=-1)
    return img


def show_points(img, points, name, scale=1, save_path=None):
    if len(img.shape) == 2:
        img = numpy.ascontiguousarray(numpy.stack([img, img, img]).transpose(1, 2, 0))
    for pt in points:
        pt = pt[1], pt[0]
        cv2.circle(img, tuple(pt), 2, color=(0, 255, 0), thickness=2)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    cv2.imshow(name, img)
    if save_path:
        cv2.imwrite(save_path, img)
    cv2.waitKey(500)


def draw_matches(matches, pts1, pts2, imgpair, iscolor=True, skip_match=False):
    # matches[0, :] = m_idx1
    # matches[1, :] = m_idx2
    # matches[2, :] = scores
    prng2 = numpy.random.RandomState(42)

    r = 0
    g = 255
    b = 0
    for i in range(matches.shape[1]):
        pt1 = (int(round(pts1[0, int(matches[0,i])])),
               int(round(pts1[1, int(matches[0,i])])))
        pt2 = (int( round(pts2[0, int(matches[1, i])])  + imgpair.shape[1] / 2 ),
               int( round(pts2[1, int(matches[1, i])]) ))

        if iscolor:
            r = prng2.randint(0, 32767) % 256
            g = prng2.randint(0, 32767) % 256
            b = 0 if (r + g > 255) else ( 255 - (r + g))
            if matches.shape[0] == 3:
                if matches[2, i]:
                    r = 255
                    g = 0
                    b = 0
                else:
                    r = 0
                    b = 255
                    g = 255

        color = (b, g, r)

        cv2.circle(imgpair, pt1,  2, color=color, thickness=-1)
        cv2.circle(imgpair, pt2, 2, color=color, thickness=-1)
        if not skip_match:
            cv2.line(imgpair, pt1, pt2, color, thickness=1)

    return imgpair


def draw_with_points(self, image, points, name="vasya"):
    img = image.copy()
    for j in range(points.shape[0]):
        p = points[j]
        cv2.circle(img, (p[1], p[0]), radius=3, thickness=-1, color=(255, 255, 255))
    cv2.imshow(name, img)
    cv2.waitKey(1000)
