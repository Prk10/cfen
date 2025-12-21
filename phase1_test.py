import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import load_config, update_config, parse_args
from utils.seed import set_seed
from utils.metrics import recall_at_k, mean_recall_at_k
from data.sgg_dataset import NPZSceneGraphDataset, SyntheticSGGDataset, collate_fn
from models.cfen import CFEN


import glob
import numpy as np
class SimpleDebugDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        try:
            arr = np.load(self.files[idx])
            return (
            torch.from_numpy(arr['features']).float(),
            torch.from_numpy(arr['obj_labels']).long(),
            torch.from_numpy(arr['rel_pairs']).long(),
            torch.from_numpy(arr['rel_labels']).long(),
            self.files[idx] # Includes filename
            )
        except: return None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = update_config(cfg, args.overrides)

    set_seed(cfg["TRAIN"]["SEED"])

    
    if cfg["DATASET"]["USE_SYNTHETIC"]:
        train_ds = SyntheticSGGDataset(
            num_images=int(cfg["DATASET"]["NUM_SYNTHETIC_IMAGES"]),
            feat_dim=int(cfg["DATASET"]["FEAT_DIM"]),
            num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
            num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
            seed=cfg["TRAIN"]["SEED"],
        )
        val_ds = SyntheticSGGDataset(
            num_images=max(50, int(cfg["DATASET"]["NUM_SYNTHETIC_IMAGES"] // 5)),
            feat_dim=int(cfg["DATASET"]["FEAT_DIM"]),
            num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
            num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
            seed=cfg["TRAIN"]["SEED"] + 1,
        )
    else:
        """
        # Load the .npz files extracted from Faster R-CNN (ResNeXt-101-FPN equivalent)
        data_dir = cfg["DATASET"]["DATA_DIR"]
        #train_ds = NPZSceneGraphDataset(data_dir, split="train")
        #val_ds = NPZSceneGraphDataset(data_dir, split="val")
        json_path = os.path.join(cfg["BASE_DIR"], 'image_data.json')
        h5_path = os.path.join(cfg["BASE_DIR"], 'VG-SGG-with-attri.h5')
        train_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="train")
        val_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="val")
        """

        

        # Manually grab all files on disk
        all_files = sorted(glob.glob(os.path.join(cfg["DATASET"]["DATA_DIR"], "*.npz")))
        
        # Manually split 80/20
        split_point = int(len(all_files) * 0.8)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]
        
        print(f"DEBUG OVERRIDE: Loaded {len(train_files)} Train, {len(val_files)} Val images.")

        train_ds = SimpleDebugDataset(train_files)
        val_ds = SimpleDebugDataset(val_files)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )
    
    # Validation is typically done with batch_size=1 to avoid graph batching complexity
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = CFEN(
        feat_dim=int(cfg["MODEL"]["FEAT_DIM"]),
        num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
        num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
        dm_lambda=float(cfg["MODEL"]["DM_LAMBDA"]),  
        ema_alpha=float(cfg["MODEL"]["EMA_ALPHA"]), 
        fusion=str(cfg["MODEL"]["FUSION"]),
    ).to(device)

    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["TRAIN"]["LR"],
        momentum=cfg["TRAIN"]["MOMENTUM"],
        weight_decay=cfg["TRAIN"]["WEIGHT_DECAY"]
    )

    os.makedirs(cfg["OUTPUT"]["CKPT_DIR"], exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg["TRAIN"]["EPOCHS"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['TRAIN']['EPOCHS']}")
        
        for feats, obj_labels, rel_pairs, rel_labels, _ in pbar:

            if rel_pairs.shape[0] == 0:
                continue
            feats = feats.to(device)         # [N_boxes, D]
            obj_labels = obj_labels.to(device)
            rel_pairs = rel_pairs.to(device)
            rel_labels = rel_labels.to(device)

            
            out = model(feats, obj_labels, rel_pairs, rel_labels, update_ema=True)
            
            
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{out['loss_ce'].item():.4f}",
                "dm": f"{out['loss_dm'].item():.4f}"
            })
            global_step += 1

        
        if epoch % cfg["TRAIN"]["VAL_INTERVAL"] == 0:
            model.eval()
            r50_total, mr50_total, n_batches = 0.0, 0.0, 0
            
            with torch.no_grad():
                for feats, obj_labels, rel_pairs, rel_labels, _ in val_loader:
                    feats = feats.to(device)
                    obj_labels = obj_labels.to(device)
                    rel_pairs = rel_pairs.to(device)
                    rel_labels = rel_labels.to(device)
                    
                    # Forward without EMA update for validation
                    out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
                    logits = out["logits"]

                    predictions = torch.argmax(logits, dim=1)
                    """
                    print(f"\n--- Debug Step {global_step} ---")
                    print(f"Ground Truth: {rel_labels[:10].cpu().numpy()}") # Show first 10
                    print(f"Predictions : {predictions[:10].cpu().numpy()}") # Show first 10
                    print(f"Unique Preds: {torch.unique(predictions).cpu().numpy()}") # Is it only predicting 1 class?
                    """
                    
                    
                    r50_total += recall_at_k(logits, rel_labels, k=cfg["EVAL"]["TOPK"])
                    mr50_total += mean_recall_at_k(
                        logits, rel_labels, k=cfg["EVAL"]["TOPK"], 
                        num_classes=cfg["DATASET"]["NUM_REL_CLASSES"]
                    )
                    n_batches += 1
                    
            print(f"[Val] Epoch {epoch} | R@{cfg['EVAL']['TOPK']}: {r50_total / max(1,n_batches):.4f} | mR@{cfg['EVAL']['TOPK']}: {mr50_total / max(1,n_batches):.4f}")

        
        ckpt_path = os.path.join(cfg["OUTPUT"]["CKPT_DIR"], f"cfen_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg
        }, ckpt_path)

if __name__ == "__main__":
    main()