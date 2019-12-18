import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np

import airsim
import json
import math
from collections import namedtuple



class AirsimIntVarDataset(Dataset):

    def __init__(self, dir_day, dir_night, poses_file, transform=None, frame_offset = 3):
        self.img_dir_1 = dir_day + '/img/'
        self.img_dir_2 = dir_night + '/img/'
        self.depth_dir_1 = dir_day + '/depth_persp/'
        self.depth_dir_2 = dir_night + '/depth_persp/'
        # list_dir = os.listdir(self.img_dir_day)
        # self.img_list = [name for name in list_dir if os.path.isfile(self.img_dir + '/' + name)]
        self.transform = transform
        with open(poses_file, 'r') as f:
            self.poses = json.load(f)

        self.img_num = len(self.poses)-frame_offset
        self.frame_offset = frame_offset

    def __len__(self):
        return self.img_num

    def get_intrinsic(self, W, H):
        K = np.eye(3, dtype=np.float32)
        f1 = W / 2
        # f2 = H / 2
        K[0, 0] = f1
        K[1, 1] = f1
        K[0, 2] = W / 2
        K[1, 2] = H / 2
        return K


    def build_rot_mat(self, roll, pitch, yaw):
        # In the camera coordinate frame: Pitch along X (to the right), Yaw along Y (down), Roll along Z (fwd)
        # In the world coordinate frame: Pitch along Y (to the right), Yaw along Z (down), Roll along X (fwd)
        # We should swap
        angle = pitch
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(angle), -math.sin(angle)],
                       [0, math.sin(angle), math.cos(angle)]])

        angle = yaw
        Ry = np.array([[math.cos(angle), 0, math.sin(angle)],
                       [0, 1, 0],
                       [-math.sin(angle), 0, math.cos(angle)]])

        angle = roll
        Rz = np.array([[math.cos(angle), -math.sin(angle), 0],
                       [math.sin(angle), math.cos(angle), 0],
                       [0, 0, 1]])
        R = np.matmul(Ry, np.matmul(Rx, Rz))

        return R

    def DepthConversion(self, PointDepth, f):
        H = PointDepth.shape[0]
        W = PointDepth.shape[1]
        i_c = np.float(H) / 2 - 1
        j_c = np.float(W) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, W - 1, num=W), np.linspace(0, H - 1, num=H))
        DistanceFromCenter = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** (0.5)
        PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f) ** 2) ** (0.5)
        return PlaneDepth

    def build_camera_to_world(self, position, orientation):
        # Camera coordinate frame (Z forward, X right, Y down)
        # doesn't correspond to the world frame (X forward, Y right, Z down)
        # Thus we should swap axes: XYZ -> YZX
        M = np.eye(4, dtype=np.float32)
        M[0:3, 3] = np.array([position['y_val'], position['z_val'], position['x_val']])

        (pitch, roll, yaw) = airsim.to_eularian_angles(namedtuple('Struct', orientation.keys())(*orientation.values()))
        R = self.build_rot_mat(roll=roll, pitch=pitch, yaw=yaw)

        # quat = np.array([orientation['x_val'], orientation['y_val'], orientation['z_val'], orientation['w_val']])
        # R = self.build_rot_mat_from_quat(quat)

        M[0:3, 0:3] = R

        return M

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID1 = self.poses[index]['id']
        ID2 = self.poses[index + self.frame_offset]['id']
        img1 = cv2.imread(self.img_dir_1 + '/' + ID1 + '.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.img_dir_2 + '/' + ID2 + '.png', cv2.IMREAD_GRAYSCALE)
        depth1 = airsim.read_pfm(self.depth_dir_1 + ID1 + '.pfm')[0]
        depth1 = np.flip(depth1, 0)
        depth2 = airsim.read_pfm(self.depth_dir_2 + ID2 + '.pfm')[0]
        depth2 = np.flip(depth2, 0)

        H1, W1 = img1.shape[:2]
        H2, W2 = img2.shape[:2]

        K1 = self.get_intrinsic(W1, H1)
        K2 = self.get_intrinsic(W2, H2)

        depth1 = self.DepthConversion(depth1, W1 / 2)
        depth2 = self.DepthConversion(depth2, W2 / 2)

        pose1 = self.build_camera_to_world(position=self.poses[index]['position'],
                                           orientation=self.poses[index]['orientation'])
        pose2 = self.build_camera_to_world(position=self.poses[index+self.frame_offset]['position'],
                                           orientation=self.poses[index+self.frame_offset]['orientation'])

        sample = {}

        # else:
        #     sample = {'input_img': img, 'template_img': img, 'p': np.ones( (8,), dtype=np.float32)}

        sample['img1'] = img1
        sample['img2'] = img2
        sample['depth1'] = depth1
        sample['depth2'] = depth2
        sample['K1'] = K1
        sample['K2'] = K2
        sample['pose1'] = pose1
        sample['pose2'] = pose2
        pose2_inv = np.linalg.inv(pose2)
        sample['H'] = np.matmul(pose2_inv, pose1)

        if self.transform:
            sample = self.transform(sample)

        return sample


class AirsimWithTarget(AirsimIntVarDataset):
    def __init__(self, dir_day, dir_night, poses_file, transform=None, frame_offset = 3):
        AirsimIntVarDataset.__init__(self, dir_day, dir_night, poses_file, transform=transform, frame_offset=frame_offset)
        self.points_dir_1 = dir_day + '/points/'
        self.points_dir_2 = dir_night + '/points/'

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID1 = self.poses[index]['id']
        ID2 = self.poses[index + self.frame_offset]['id']
        img1 = cv2.imread(self.img_dir_1 + '/' + ID1 + '.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.img_dir_2 + '/' + ID2 + '.png', cv2.IMREAD_GRAYSCALE)
        depth1 = airsim.read_pfm(self.depth_dir_1 + ID1 + '.pfm')[0]
        depth1 = np.flip(depth1, 0)
        depth2 = airsim.read_pfm(self.depth_dir_2 + ID2 + '.pfm')[0]
        depth2 = np.flip(depth2, 0)
        target1 = np.load(self.points_dir_1 + '/' + ID1 + '.npy')
        target2 = np.load(self.points_dir_2 + '/' + ID2 + '.npy')
        H1, W1 = img1.shape[:2]
        H2, W2 = img2.shape[:2]

        K1 = self.get_intrinsic(W1, H1)
        K2 = self.get_intrinsic(W2, H2)

        depth1 = self.DepthConversion(depth1, W1 / 2)
        depth2 = self.DepthConversion(depth2, W2 / 2)

        pose1 = self.build_camera_to_world(position=self.poses[index]['position'],
                                           orientation=self.poses[index]['orientation'])
        pose2 = self.build_camera_to_world(position=self.poses[index+self.frame_offset]['position'],
                                           orientation=self.poses[index+self.frame_offset]['orientation'])

        sample = {}

        # else:
        #     sample = {'input_img': img, 'template_img': img, 'p': np.ones( (8,), dtype=np.float32)}

        sample['img1'] = img1
        sample['img2'] = img2
        sample['depth1'] = depth1
        sample['depth2'] = depth2
        sample['K1'] = K1
        sample['K2'] = K2
        sample['pose1'] = pose1
        sample['pose2'] = pose2
        sample['points1'] = target1
        sample['points2'] = target2
        pose2_inv = np.linalg.inv(pose2)
        sample['H'] = np.matmul(pose2_inv, pose1)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img1 = sample['img1']
        img2 = sample['img2']
        depth1 = sample['depth1']
        depth2 = sample['depth2']
        K1 = sample['K1']
        K2 = sample['K2']
        pose1 = sample['pose1']
        pose2 = sample['pose2']
        H = sample['H']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_img1 = img1.transpose((2, 0, 1))
        input_img2 = img2.transpose((2, 0, 1))
        return {
                'img1': torch.from_numpy(input_img1),
                'img2': torch.from_numpy(input_img2),
                'depth1': torch.from_numpy(depth1),
                'depth2': torch.from_numpy(depth2),
                'K1': torch.from_numpy(K1),
                'K2': torch.from_numpy(K2),
                'pose1': torch.from_numpy(pose1),
                'pose2': torch.from_numpy(pose2),
                'H': torch.from_numpy(H)
                }
