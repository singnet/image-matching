import numpy
import cv2

X_ = 0
Y_ = 1


def calcSingleProjectionAirsim(point_to_project, H, depth,
                               K, Kinv, projection):
    xi = point_to_project.pt.x
    yi = point_to_project.pt.y

    pK = K.reshape(numpy.prod(K.shape))
    pKinv = Kinv.reshape(numpy.prod(Kinv.shape))
    pDepth = depth

    # Reconstruct 3d point in the first camera frame
    # f = W / 2 / tan(FOV/2) = W/2, but since we multiplied points by Kinv it's 1
    float Xn = pKinv[0]*point_to_project.pt.x + pKinv[1]*point_to_project.pt.y + pKinv[2];
    float Yn = pKinv[3]*point_to_project.pt.x + pKinv[4]*point_to_project.pt.y + pKinv[5];
    float Z1 = pDepth[yi*depth.cols + xi];
    float X1 = Z1*Xn;
    float Y1 = Z1*Yn;

    // Move 3d point to the second camera frame
    const float* pH = H.ptr<float>();
    float X2 = pH[0]*X1 + pH[1]*Y1 + pH[2]*Z1 + pH[3];
    float Y2 = pH[4]*X1 + pH[5]*Y1 + pH[6]*Z1 + pH[7];
    float Z2 = pH[8]*X1 + pH[9]*Y1 + pH[10]*Z1 + pH[11];

    //Project point on camera2
    float xn = pK[0] * X2 / Z2 + pK[2];
    float yn = pK[4] * Y2 / Z2 + pK[5];


    projection[X_] = xn;
    projection[Y_] = yn;


def calcProjectionsAirsim(src, H, depth, K, Kinv):

    if len(src):
        assert( not H.empty() and H.shape[1] == 4 and H.shape[0] == 4);
        dst = numpy.zeros_like(src)
        for i in range(0, len(src)):
            calcSingleProjectionAirsim(src[i], H, depth, K, Kinv, dst[i]);




def calculateMatchingPrecisionAndRecallAirsim(**kwargs):
    width, height = kwargs.get('width'), kwargs.get('height')
    H1to2 = kwargs.get('H1to2')
    depth1 = kwargs.get('depth1')
    depth2 = kwargs.get('depth2')
    keypoints1 = kwargs.get('points1')
    keypoints2 = kwargs.get('points2')
    matches1 = kwargs.get('matches1')
    matches2 = kwargs.get('matches2')
    nTruePositiveCount: int = 0
    err_thresh: float = kwargs.get('err_thresh')

    fPrecision:float = 0.0
    fRecall:float = 0.0

    assert (matches1.size() == matches2.size())
    nMatches:int = len(matches1)

    K = numpy.eye(3).astype(numpy.float32)
    pK = K.reshape(9)
    pK[4] = pK[0] = width/2
    pK[2] = width/2
    pK[5] = height/2

    _, Kinv = cv2.invert(K)

    matches1t = calcProjectionsAirsim(keypoints1, H1to2, depth1, K, Kinv);
    calcProjectionsAirsim(matches1, H1to2, depth1, K, Kinv, matches1t);

