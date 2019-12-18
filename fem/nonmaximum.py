from torch import nn
import numpy as np


class HeatmapNMS(nn.Module):
    pass


class CoordsNMS:
    pass


class PoolingNms1(HeatmapNMS):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_size,
                                       stride=self.kernel_size,
                                       return_indices=True)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=self.kernel_size,
                                            stride=self.kernel_size)
    def forward(self, x, img=None):

        size = x.shape
        pooled, indices = self.pool1(x)
        x = self.unpool1(pooled, indices, output_size=size)
        return x


class PoolingNms(HeatmapNMS):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_size,
                                       stride=self.kernel_size,
                                       return_indices=True)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=self.kernel_size,
                                            stride=self.kernel_size)

        self.pool2 = nn.MaxPool2d(kernel_size=self.kernel_size + 2,
                                       stride=self.kernel_size + 2,
                                       return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=self.kernel_size + 2,
                                            stride=self.kernel_size + 2)

        self.pool3 = nn.MaxPool2d(kernel_size=self.kernel_size + 3,
                                       stride=(self.kernel_size + 3, self.kernel_size  + 3),
                                       return_indices=True)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=self.kernel_size + 3,
                                       stride=(self.kernel_size + 3, self.kernel_size + 3),
                                        )

    def forward(self, x, img=None):

        size = x.shape
        pooled, indices = self.pool1(x)
        x = self.unpool1(pooled, indices, output_size=size)

        pooled, indices = self.pool2(x)
        x = self.unpool2(pooled, indices, output_size=size)
        pooled, indices = self.pool3(x)
        x = self.unpool3(pooled, indices, output_size=size)

        return x

    def show_distance(self, x, img=None, name=None):
        import numpy
        points = (x > 0.3).nonzero()[:, 2:4]
        print(len(points))
        import sklearn.metrics
        dist = numpy.eye(len(points)) * 1000 + sklearn.metrics.pairwise_distances(points, points)
        print(dist.min())
        from drawing import show_points
        if img is not None:
            show_points(img, points, name, scale=2)


class MagicNMS(CoordsNMS):

    def __init__(self, border=4, nms_dist=8):
        super().__init__()
        self.nms_dist = nms_dist
        self.border_remove = border  # Remove points this close to the border.

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def __call__(self, H, W, pts):
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts

