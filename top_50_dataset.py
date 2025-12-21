import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import h5py
import json
import os
import glob
from tqdm import tqdm

BASE_DIR = '/Users/prk/Documents/AL_ML/cfen/dataset_specifics'
IMG_DIR = os.path.join(BASE_DIR, 'VG_100K')
H5_PATH = os.path.join(BASE_DIR, 'VG-SGG-with-attri.h5')
IMG_DATA_JSON = os.path.join(BASE_DIR, 'image_data.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'NPZ_DatasetVG_100K')
json_path = os.path.join(BASE_DIR, 'image_data.json')
h5_path = os.path.join(BASE_DIR, 'VG-SGG-with-attri.h5')


# Stops generation after 50 images 
#NUM_IMAGES_TO_PROCESS = 50 
NUM_IMAGES_TO_PROCESS = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading Faster R-CNN (ResNet50-FPN) backbone...")
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.model.eval()
        self.backbone = self.model.backbone
        self.box_head = self.model.roi_heads.box_head
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward(self, images, boxes):
        image_batch = torch.stack(images)
        image_batch = F.normalize(image_batch, mean=self.normalize_mean, std=self.normalize_std)
        features = self.backbone(image_batch)
        
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        original_image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        box_features = box_roi_pool(features, boxes, original_image_sizes)
        final_features = self.box_head(box_features)
        return final_features

def process_all_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing images on {device}...")
    
    extractor = FeatureExtractor().to(device)

    # Identify all images in the directory
    image_files = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"Found {len(image_files)} images in {IMG_DIR}")

    if len(image_files) == 0:
        print("No images found! Check your IMG_DIR path.")
        return

    # Load Metadata Tables
    print("Loading Metadata...")
    with open(IMG_DATA_JSON, 'r') as f:
        img_data = json.load(f)
    
    
    id_to_index = {item['image_id']: i for i, item in enumerate(img_data)}

    
    f_h5 = h5py.File(H5_PATH, 'r')

    success_count = 0
    
    for img_path in tqdm(image_files, desc="Generating .npz"):
        
        #for phase 1 the number of images is 50. for processing the entire dataset, num_images_to_process is changed to none
        if NUM_IMAGES_TO_PROCESS is not None and success_count >= NUM_IMAGES_TO_PROCESS:
            print(f"Phase 1 limit reached ({NUM_IMAGES_TO_PROCESS} images). Stopping.")
            break
        

        try:
            filename = os.path.basename(img_path)
            target_id = int(filename.split('.')[0])
            save_path = os.path.join(OUTPUT_DIR, f"{target_id}.npz")

            if target_id not in id_to_index:
                # print(f"Warning: Image ID {target_id} not found.") 
                continue
                
            img_h5_index = id_to_index[target_id]

            # Load Image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device)

            # Get H5 Data
            first_box = f_h5['img_to_first_box'][img_h5_index]
            last_box = f_h5['img_to_last_box'][img_h5_index]
            
            if first_box < 0:
                continue

            boxes_center = f_h5['boxes_1024'][first_box : last_box + 1]
            obj_labels = f_h5['labels'][first_box : last_box + 1].flatten()

            first_rel = f_h5['img_to_first_rel'][img_h5_index]
            last_rel = f_h5['img_to_last_rel'][img_h5_index]
            
            if first_rel >= 0:
                rel_pairs = f_h5['relationships'][first_rel : last_rel + 1]
                
                
                rel_pairs = rel_pairs - first_box 
                

                rel_labels = f_h5['predicates'][first_rel : last_rel + 1].flatten()
            else:
                rel_pairs = np.zeros((0, 2))
                rel_labels = np.zeros((0,))

            # Coordinate Conversion
            h_raw, w_raw = img_rgb.shape[:2]
            scale = 1024.0 / max(h_raw, w_raw)
            
            boxes_abs = []
            for cx, cy, w, h in boxes_center:
                x1 = (cx - w/2) / scale
                y1 = (cy - h/2) / scale
                x2 = (cx + w/2) / scale
                y2 = (cy + h/2) / scale
                boxes_abs.append([x1, y1, x2, y2])
            
            boxes_tensor = torch.tensor(boxes_abs, dtype=torch.float32).to(device)

            # Extract Features
            with torch.no_grad():
                features = extractor([img_tensor], [boxes_tensor])
            
            # Save
            np.savez(save_path, 
                     features=features.cpu().numpy(),
                     obj_labels=obj_labels,
                     rel_pairs=rel_pairs,
                     rel_labels=rel_labels)
            success_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    f_h5.close()
    print(f"Processing Complete. Successfully saved {success_count} .npz files.")
"""
class NPZSceneGraphDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        # Load ALL files if dataset is small (Phase 1 check)
        if len(self.files) < 100:
            print(f"DEBUG: Dataset small ({len(self.files)} files), loading ALL for verification.")
        else:
            
            if split == 'val':
                self.files = [f for i, f in enumerate(self.files) if i % 5 == 0]
            elif split == 'train':
                self.files = [f for i, f in enumerate(self.files) if i % 5 != 0]
        
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir} for split={split}.")

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
            print(f"Error loading {self.files[idx]}: {e}")
            return None
"""

class NPZSceneGraphDataset(Dataset):
    def __init__(self, data_dir, h5_path, image_data_json, split='train'):
        
        self.split = split
        self.files = []
        
        print(f"Initializing dataset for split: {split}")
        
        
        with open(image_data_json, 'r') as f:
            img_data = json.load(f)
        
        idx_to_id = {i: item['image_id'] for i, item in enumerate(img_data)}

        
        print(f"Loading split configuration from {h5_path}...")
        with h5py.File(h5_path, 'r') as f_h5:
            if 'split' in f_h5:
                split_mask = f_h5['split'][:] 
            else:
                raise RuntimeError("H5 file missing 'split' key. Cannot determine train/test sets.")

        # Paper Standard: Train=0, Test=2. 
        # We handle 'val' by carving 5k out of 'train' (standard practice)
        target_split_code = 2 if split == 'test' else 0 

        
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
        
def verify_dataset():
    print("\n--- Verifying All Generated Files ---")
    try:
        dataset = NPZSceneGraphDataset(OUTPUT_DIR, image_data_json=json_path, h5_path=h5_path, split='test')
        print(f"Dataset initialized with {len(dataset)} files.")
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        valid_count = 0
        for batch in loader:
            if batch is None: continue
            feats, objs, pairs, rels, filename = batch
            
            feats = feats.squeeze(0)
            
            if feats.dim() == 2 and feats.shape[1] == 1024:
                valid_count += 1
            else:
                print(f"Shape Mismatch in {filename}: {feats.shape}")

        print(f"Verification Successful: {valid_count}/{len(dataset)} files loaded correctly.")
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    #process_all_images()
    verify_dataset()