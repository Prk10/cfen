import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZSceneGraphDataset(Dataset):
    """
    Minimal dataset loader for per-image .npz files.

    Each .npz must contain:
      - features: float32 [num_boxes, feat_dim]
      - obj_labels: int64 [num_boxes]
      - rel_pairs: int64 [num_pairs, 2]
      - rel_labels: int64 [num_pairs]
    """
    def __init__(self, data_dir, split='train'):
        # For simplicity in this baseline, we just split by filename hash.
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if split in ['train', 'val']:
            self.files = [f for i, f in enumerate(self.files) if (i % 5 == (0 if split=='val' else 1))]
        else:
            self.files = [f for i, f in enumerate(self.files) if (i % 5 == 0)]
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir} for split={split}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        feats = torch.from_numpy(arr['features']).float()
        obj_labels = torch.from_numpy(arr['obj_labels']).long()
        rel_pairs = torch.from_numpy(arr['rel_pairs']).long()
        rel_labels = torch.from_numpy(arr['rel_labels']).long()
        return feats, obj_labels, rel_pairs, rel_labels

class SyntheticSGGDataset(Dataset):
    def __init__(self, num_images=200, feat_dim=1024, num_obj_classes=151, num_rel_classes=51, seed=1337):
        rng = np.random.RandomState(seed)
        self.samples = []
        for _ in range(num_images):
            n_boxes = rng.randint(5, 20)
            n_pairs = rng.randint(max(1, n_boxes-3), max(2, n_boxes*(n_boxes-1)//4))
            features = rng.randn(n_boxes, feat_dim).astype('float32')
            obj_labels = rng.randint(0, num_obj_classes, size=(n_boxes,), dtype='int64')
            # rel pairs indices
            subj_idx = rng.randint(0, n_boxes, size=(n_pairs,))
            obj_idx = rng.randint(0, n_boxes, size=(n_pairs,))
            mask = subj_idx != obj_idx
            subj_idx = subj_idx[mask]
            obj_idx = obj_idx[mask]
            n_pairs = len(subj_idx)
            rel_pairs = np.stack([subj_idx, obj_idx], axis=1).astype('int64')
            rel_labels = rng.randint(0, num_rel_classes, size=(n_pairs,), dtype='int64')
            self.samples.append((features, obj_labels, rel_pairs, rel_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, o, p, r = self.samples[idx]
        return torch.from_numpy(f), torch.from_numpy(o), torch.from_numpy(p), torch.from_numpy(r)

def collate_fn(batch):
    # batch of variable-sized graphs; use batch_size=1 in training for simplicity
    return batch[0]
