import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import h5py
class NPZSceneGraphDataset(Dataset):
    def __init__(self, data_dir, h5_path, image_data_json, split='train'):
        """
        Args:
            data_dir: Folder containing .npz files
            h5_path: Path to 'VG-SGG-with-attri.h5' (Contains official split mask)
            image_data_json: Path to 'image_data.json' (Maps index to Image ID)
            split: 'train', 'val', or 'test'
        """
        self.split = split
        self.files = []
        
        print(f"Initializing dataset for split: {split}")
        
        
        with open(image_data_json, 'r') as f:
            img_data = json.load(f)
        
        idx_to_id = {i: item['image_id'] for i, item in enumerate(img_data)}

        
        # Standard SGG H5 coding: 0=Train, 1=Val (rarely used), 2=Test
        print(f"Loading split configuration from {h5_path}...")
        with h5py.File(h5_path, 'r') as f_h5:
            if 'split' in f_h5:
                split_mask = f_h5['split'][:] # Load into memory
            else:
                raise RuntimeError("H5 file missing 'split' key. Cannot determine train/test sets.")

        
        # Paper Standard: Train=0, Test=2. 
        # We handle 'val' by carving 5k out of 'train' (standard practice)
        target_split_code = 2 if split == 'test' else 0 

        # 4. Build the valid file list
        all_npz = set(os.path.basename(x) for x in glob.glob(os.path.join(data_dir, '*.npz')))
        valid_files = []

        for h5_idx, code in enumerate(split_mask):
            if code == target_split_code:
                # Get the image ID for this index
                img_id = idx_to_id[h5_idx]
                filename = f"{img_id}.npz"
                
                # Verify we actually generated this .npz file
                if filename in all_npz:
                    valid_files.append(os.path.join(data_dir, filename))
        
        # Handle Train/Val Split (Standard 5k Holdout)
        # If we are in 'train' or 'val', we are looking at code 0 (Training set).
        # We must manually split this into Train (minus 5k) and Val (last 5k).
        if split in ['train', 'val']:
            # Sort to ensure deterministic split every time
            # We sort by H5 Index (which is implicit in the list order we just built)
            # but to be safe, let's sort by filename (Image ID)
            valid_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            
            VAL_SIZE = 5000
            
            if len(valid_files) > VAL_SIZE:
                if split == 'val':
                    self.files = valid_files[-VAL_SIZE:] # Last 5000 are Val
                else: # train
                    self.files = valid_files[:-VAL_SIZE] # The rest are Train
            else:
                
                print("Warning: Dataset too small for 5k split. Using simple 80/20.")
                split_point = int(len(valid_files) * 0.8)
                if split == 'train':
                    self.files = valid_files[:split_point]
                else:
                    self.files = valid_files[split_point:]
        else:
            
            self.files = valid_files

        print(f"Final Dataset Split '{split}': {len(self.files)} images loaded.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            arr = np.load(self.files[idx])
            feats = torch.from_numpy(arr['features']).float()
            obj_labels = torch.from_numpy(arr['obj_labels']).long()
            rel_pairs = torch.from_numpy(arr['rel_pairs']).long()
            rel_labels = torch.from_numpy(arr['rel_labels']).long()
            return feats, obj_labels, rel_pairs, rel_labels, self.files[idx]
        except Exception as e:
            return None
'''
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
'''
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
