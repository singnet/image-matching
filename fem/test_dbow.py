import torch
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
    return parser.parse_args()


def get_datasets():
    frame_offset = 0
    batch_size = 1
    dataset_village = AirsimIntVarDataset(**village, frame_offset=frame_offset)
    dataset_fantasy_village = AirsimIntVarDataset(**fantasy_village, frame_offset=frame_offset)

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
    assert (desc.shape[0] == 256)
    return [x[np.newaxis, ...] for x in desc.T]


def norm_average_rank(N, R):
    N_rel = len(R)
    return 1 / (N * N_rel) * (sum(R) - N_rel * (N_rel - 1) / 2)


def loop(model, db, loader, thresh, device, dataset_name):
    night_features = []
    day_features = []
    day_path = '{0}_day.pkl'.format(dataset_name)
    night_path = '{0}_night.pkl'.format(dataset_name)
    if os.path.exists(day_path):
        print('loading precomputed features')
        day_features = pickle.load(open(day_path, 'rb'))
        night_features = pickle.load(open(night_path, 'rb'))
        for desc1 in day_features:
            lst1 = features_to_list(desc1)
            db.add(lst1)
    else:
        for i_batch, sample in enumerate(loader):
            img_1 = sample['img1'].numpy()
            img_2 = sample['img2'].numpy()

            pts1, desc1 = get_points_desc(device=device,
                        img=img_1[np.newaxis,...],
                        thresh=thresh,
                        fe=None, sp=model)
            pts2, desc2 = get_points_desc(device=device,
                        img=img_2[np.newaxis,...],
                        thresh=thresh,
                        fe=None, sp=model)
            day_features.append(desc1)
            night_features.append(desc2)
            lst1 = features_to_list(desc1)
            db.add(lst1)
            if not i_batch % 100:
                print('processing ', i_batch)
        with open(day_path, 'wb') as f:
            pickle.dump(day_features, f)
        with open(night_path, 'wb') as f:
            pickle.dump(night_features, f)
    import pdb;pdb.set_trace()
    res = []
    for i, desc2 in enumerate(night_features):
        query_results = db.query(features_to_list(desc2), len(day_features) + 1)
        ids = [x.Id for x in query_results]
        expected  = [x for x in range(i - 3, i + 3) if x >= 0]
        indexes = [ids.index(x) + 1 for x in expected if x in ids]
        rank = norm_average_rank(len(day_features), indexes)
        print(rank)
        res.append(rank)
    print(np.mean(res))


def main():
    device = 'cuda'
    args = parse_args()
    vocab = DBoW.GOODVocabulary()
    vocab.load_json(args.vocabulary)

    db = DBoW.GOODDatabase(vocab, False, 0)
    thresh = 0.035

    village_loader, fantasy_loader = get_datasets()

    model = get_model(device)
    loop(model, db, village_loader, thresh, device, 'village')


if __name__ == '__main__':
    main()
