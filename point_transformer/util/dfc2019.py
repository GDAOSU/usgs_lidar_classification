import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare, data_prepare_xyzIR

from pathlib import Path
import pickle

class DFC2019(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        
        dataset_dir = Path(data_root)
        train_ids_path = dataset_dir.parent / "train_ids.txt"
        val_ids_path = dataset_dir.parent / "val_ids.txt"
        test_ids_path = dataset_dir.parent / "test_ids.txt"
        
        assert train_ids_path.exists(), f"{train_ids_path} does not exist"
        assert val_ids_path.exists(), f"{val_ids_path} does not exist"
        assert test_ids_path.exists(), f"{test_ids_path} does not exist"
        
        if split== 'train':
            self.data_list = np.loadtxt(train_ids_path, dtype=str)
        elif split == 'val':
            self.data_list = np.loadtxt(val_ids_path, dtype=str)
        elif split == 'test':
            self.data_list = np.loadtxt(test_ids_path, dtype=str)

        intensity_90_percentiles = []
        
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = dataset_dir / f"{item}.pkl"
                pts, label = pickle.load(open(data_path, "rb"))[:2] # pts xyzIR: (N, 5), label: (N,)
                data = np.concatenate([pts, label.reshape(-1, 1)], axis=1)
                sa_create("shm://{}".format(item), data)
                
            data = SA.attach("shm://{}".format(item))
            intensity_90_percentiles.append(np.nanpercentile(data[:, 3], 90))
            
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))
        # print(intensity_90_percentiles)
        self.feat_normalizer = np.array([np.median(intensity_90_percentiles), 3.]).reshape(-1, 2).astype(np.float32)
        print(f"intensity_divisor: {self.feat_normalizer[0,0]}")
        print(f"return_divisor: {self.feat_normalizer[0,1]}")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:5], data[:, 5]
        feat /= self.feat_normalizer
        coord, feat, label = data_prepare_xyzIR(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop

if __name__ == '__main__':
    import torch
    from util.data_util import collate_fn
    train_data = DFC2019(split='train', data_root="dataset/dfc2019/trainval", voxel_size=0.5, voxel_max=80000, transform=None, shuffle_index=False, loop=1)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, sampler=None, drop_last=True, collate_fn=collate_fn)
    
    print("len TrainLoader", len(train_loader))
    
    features = [] 
    labels = set()
    for i, (coord, feat, target, offset) in enumerate(train_loader):
        # print(f"coord: {coord.shape}")
        
        features.append(feat)
        labels.update(np.unique(target))
        # print(f"{i} >> feat: {feat.shape}")
        # print(np.minimum(feat, axis=0), np.maximum(feat, axis=0))
        # print(np.percentile(feat, 90, axis=0))
        # print(np.percentile(feat, 100, axis=0))
        # print(f"target: {target.shape}")
        # print(f"offset: {offset.shape}")
    features = torch.cat(features, dim=0)
    print(f"0%: {np.percentile(features, 0, axis=0)}")
    print(f"25%: {np.percentile(features, 25, axis=0)}")
    print(f"50%: {np.percentile(features, 50, axis=0)}")
    print(f"75%: {np.percentile(features, 75, axis=0)}")
    print(f"90%: {np.percentile(features, 90, axis=0)}")
    print(f"95%: {np.percentile(features, 95, axis=0)}")
    print(f"100%: {np.percentile(features, 100, axis=0)}")
    
    print(f"labels: {labels}")
        
        
    