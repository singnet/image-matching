import torch
import numpy
import cv2
import pickle
import os
import numpy as np
from airsim_dataset import AirsimIntVarDataset
from torch.utils.data import DataLoader

from eval import village, fantasy_village
from goodpoint import GoodPoint
from nonmaximum import PoolingNms
from bench import get_points_desc
from lib import DBoW
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vocabulary")
    parser.add_argument("dataset")
    parser.add_argument('--reuse', action='store_true', default=False,
            help='reuse keypoints from first img',
            required=False)
    return parser.parse_args()


def get_datasets():
    frame_offset = 0
    batch_size = 1

    from fem.noise import AdditiveGaussian, AdditiveShade, SaltPepper
    gaussian = AdditiveGaussian(var=30)
    shade = AdditiveShade(kernel_size_range=[45, 85],
                          transparency_range=(-0.25, .45))
    salt_pepper = SaltPepper()

    transforms = [salt_pepper]
    transforms = []
    transforms = [gaussian, shade, salt_pepper]
    print('using noise ', transforms)
    def noisy(sample):
        sample['img1'] = sample['img1'][np.newaxis, ...]
        for tr in transforms:
            sample['img1'] = tr(sample['img1'])
        sample['img1'] = sample['img1'].squeeze()
        return sample

    dataset_village = AirsimIntVarDataset(**village, frame_offset=frame_offset, transform=noisy)
    dataset_fantasy_village = AirsimIntVarDataset(**fantasy_village, frame_offset=frame_offset, transform=noisy)
    village_loader = DataLoader(dataset_village, batch_size=batch_size, shuffle=False, num_workers=1)
    fantasy_loader = DataLoader(dataset_fantasy_village, batch_size=batch_size, shuffle=False, num_workers=1)
    return village_loader, fantasy_loader


def get_model(device):
    weight = "./snapshots/super3400.pt"
    nms = PoolingNms(8)
    sp = GoodPoint(dustbin=0,
                   activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval()
    sp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    return sp


def features_to_list(desc):
    assert (desc.shape[1] in (256, 32))
    return [x[np.newaxis, ...] for x in desc]


def norm_average_rank(N, R):
    N_rel = len(R)
    return 1 / (N * N_rel) * (sum(R) - N_rel * (N_rel - 1) / 2)


def loop(extract, db, loader, dataset_name, reuse):
    night_features = []
    day_features = []
    day_path = '{0}_day.pkl'.format(dataset_name)
    night_path = '{0}_night.pkl'.format(dataset_name)
    db_indexes = []
    if os.path.exists(day_path):
        print('loading precomputed features')
        day_features = pickle.load(open(day_path, 'rb'))
        night_features = pickle.load(open(night_path, 'rb'))
        for desc1 in day_features:
            lst1 = features_to_list(desc1)
            idx = db.add(lst1)
            db_indexes.append(idx)
    else:
        for i_batch, sample in enumerate(loader):
            img_1 = sample['img1'].numpy()
            img_2 = sample['img2'].numpy()

            pts1, desc1 = extract(img_1)
            points2 = None
            if reuse:
                points2 = pts1
            pts2, desc2 = extract(img_2, points2)
            day_features.append(desc1)
            night_features.append(desc2)
            lst1 = features_to_list(desc1)
            idx = db.add(lst1)
            db_indexes.append(idx)
            if not i_batch % 100:
                print('processing ', i_batch)
        #with open(day_path, 'wb') as f:
        #    pickle.dump(day_features, f)
        #with open(night_path, 'wb') as f:
        #    pickle.dump(night_features, f)
    res = []
    for i, desc2 in enumerate(night_features):
        query_results = db.query(features_to_list(desc2), len(day_features) + 1)
        ids = [x.Id for x in query_results]
        expected  = [x for x in range(i - 4, i + 5) if x >= 0]
        indexes = []
        for x in expected:
            if x in ids:
                indexes.append(ids.index(x) + 1)
            else:
                indexes.append(numpy.random.randint(1, len(day_features) + 1))
        rank = norm_average_rank(len(day_features), indexes)
        res.append(rank)
    print('min(rank)', np.min(res))
    print('max(rank)', np.max(res))
    print("rank ", np.mean(res))

def test_orb_sep(args):
    """
    extract points separatly for day and night images
    """
    vocab = DBoW.ORBVocabulary()
    vocab.loadFromTextFile(args.vocabulary)
    db = DBoW.ORBDatabase(vocab, False, 0)
    print('   depth=', vocab.getDepthLevels(), "k=", vocab.getBranchingFactor())

    orb = cv2.ORB_create()
    village_loader, fantasy_loader = get_datasets()

    loaders = dict(village=village_loader,
                   fantasy=fantasy_loader)

    def function(image, points=None):
        assert image.max() > 10
        image = image.squeeze().astype(numpy.uint8)
        kp = orb.detect(image, None)
        if points is not None:
            kp = points
        # compute the descriptors with ORB
        kp, des = orb.compute(image, kp)
        return kp, des

    loop(function, db, loaders[args.dataset], args.dataset + '_orb', reuse=args.reuse)

def test_good_sep(args):
    """
    extract points separatly for day and night images
    """
    device = 'cuda'
    vocab = DBoW.GOODVocabulary()
    vocab.load_json(args.vocabulary)
    print('   depth=', vocab.getDepthLevels(), "k=", vocab.getBranchingFactor())

    db = DBoW.GOODDatabase(vocab, False, 0)
    thresh = 0.035

    village_loader, fantasy_loader = get_datasets()
    loaders = dict(village=village_loader,
                   fantasy=fantasy_loader)

    model = get_model(device)

    def function(img, points=None):
        if points is not None:
            points = points.T[:, :2]
        points, desc = get_points_desc(device=device,
                        img=img[np.newaxis, ...],
                        thresh=thresh,
                        fe=None, sp=model, points_precomputed=points)
        return points, desc.T
    loop(function, db, loaders[args.dataset], args.dataset + '_good', reuse=args.reuse)


if __name__ == '__main__':
    args = parse_args()
    test_good_sep(args)
#    test_orb_sep(args)

