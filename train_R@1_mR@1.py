import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config, update_config, parse_args
from utils.seed import set_seed
from utils.metrics import recall_at_k, mean_recall_at_k
from data.sgg_dataset import NPZSceneGraphDataset, SyntheticSGGDataset, collate_fn
from models.cfen import CFEN

def calculate_sgg_metrics(logits, rel_labels, k_list=[50, 100]):
    """
    Calculates SGG Triplet Recall matches for a single image (batch_size=1).
    Returns:
        hits_k: dict {k: total_hits}
        gt_count: total ground truth relations in this image
        hits_per_class_k: dict {k: {class_id: hits}}
        gt_per_class: dict {class_id: count}
    """
    # 1. Identify Ground Truth Triplets (indices and labels)
    # rel_labels shape: [num_pairs]
    # We ignore background (label 0)
    gt_indices = (rel_labels > 0).nonzero(as_tuple=True)[0]
    gt_count = len(gt_indices)
    
    gt_per_class = {}
    gt_triplets = set()
    
    for idx in gt_indices:
        pair_idx = idx.item()
        label = rel_labels[idx].item()
        gt_triplets.add((pair_idx, label))
        gt_per_class[label] = gt_per_class.get(label, 0) + 1

    if gt_count == 0:
        return {k: 0 for k in k_list}, 0, {k: {} for k in k_list}, {}

    # 2. Flatten Predictions for Ranking
    # logits shape: [num_pairs, num_classes] (e.g., [N, 51])
    # We slice [:, 1:] to remove background class 0. New shape [N, 50]
    scores_fg = logits[:, 1:] 
    num_pairs, num_fg_classes = scores_fg.shape
    
    # Flatten to [N * 50]
    flat_scores = scores_fg.flatten()
    
    # We only need the top max(k) scores to determine hits
    max_k = max(k_list)
    current_k = min(flat_scores.shape[0], max_k)
    
    topk_vals, topk_inds = torch.topk(flat_scores, current_k)
    
    # Map flat indices back to (pair_idx, class_label)
    # Note: class_label must be shifted by +1 because we sliced off class 0
    topk_pair_indices = (topk_inds // num_fg_classes).tolist()
    topk_labels = (topk_inds % num_fg_classes + 1).tolist()
    
    top_triplets = list(zip(topk_pair_indices, topk_labels))

    # 3. Compute Hits for each K
    hits_k = {k: 0 for k in k_list}
    hits_per_class_k = {k: {} for k in k_list}

    for k in k_list:
        # Get top K triplets
        current_top = set(top_triplets[:k])
        
        # Check intersections with GT
        # Intersection gives us the "Hits"
        matches = gt_triplets.intersection(current_top)
        hits_k[k] = len(matches)
        
        # Record hits per class (for mR@K)
        for (_, label) in matches:
            hits_per_class_k[k][label] = hits_per_class_k[k].get(label, 0) + 1
            
    return hits_k, gt_count, hits_per_class_k, gt_per_class

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = update_config(cfg, args.overrides)

    set_seed(cfg["TRAIN"]["SEED"])

    # Dataset
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
        data_dir = cfg["DATASET"]["DATA_DIR"]
        json_path = os.path.join(cfg["BASE_DIR"], 'image_data.json')
        h5_path = os.path.join(cfg["BASE_DIR"], 'VG-SGG-with-attri.h5')
        train_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="train")
        val_ds = NPZSceneGraphDataset(data_dir, image_data_json=json_path, h5_path=h5_path, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=True, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["NUM_WORKERS"],
        collate_fn=collate_fn, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
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

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 1. Load Model Weights
            model.load_state_dict(checkpoint['model_state'])
            
            # 2. Load Optimizer State (important for momentum/decay)
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # 3. Update Start Epoch
            # We add 1 because the saved epoch was the last fully completed one
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    os.makedirs(cfg["OUTPUT"]["CKPT_DIR"], exist_ok=True)

    global_step = 0
    for epoch in range(start_epoch, cfg["TRAIN"]["EPOCHS"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['TRAIN']['EPOCHS']}")
        for feats, obj_labels, rel_pairs, rel_labels, _ in pbar:
            feats = feats.to(device)
            obj_labels = obj_labels.to(device)
            rel_pairs = rel_pairs.to(device)
            rel_labels = rel_labels.to(device)

            # Forward
            out = model(feats, obj_labels, rel_pairs, rel_labels, update_ema=True)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["TRAIN"]["GRAD_CLIP_NORM"])
            optimizer.step()

            if global_step % cfg["TRAIN"]["LOG_INTERVAL"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{out['loss_ce'].item():.4f}",
                    "dm": f"{out['loss_dm'].item():.4f}"
                })
            global_step += 1

        # Validation
        if epoch % cfg["TRAIN"]["VAL_INTERVAL"] == 0:
            model.eval()
            
            # Global Accumulators
            total_hits = {50: 0, 100: 0}
            total_gt_count = 0
            
            # Per-Class Accumulators (class_id -> count)
            class_hits = {50: {}, 100: {}}
            class_gt_count = {}
            
            with torch.no_grad():
                for i, (feats, obj_labels, rel_pairs, rel_labels, _) in enumerate(tqdm(val_loader, desc="Validating")):
                    feats = feats.to(device)
                    obj_labels = obj_labels.to(device)
                    rel_pairs = rel_pairs.to(device)
                    rel_labels = rel_labels.to(device)
                    
                    # Forward pass
                    out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
                    logits = out["logits"]
                    
                    # Compute SGG Metrics for this image
                    # Note: We pass the full logits and labels. The helper handles flattening/sorting.
                    b_hits, b_gt, b_class_hits, b_class_gt = calculate_sgg_metrics(
                        logits, rel_labels, k_list=[50, 100]
                    )
                    
                    # Accumulate Global
                    total_gt_count += b_gt
                    for k in [50, 100]:
                        total_hits[k] += b_hits[k]
                    
                    # Accumulate Per-Class
                    for cls, count in b_class_gt.items():
                        class_gt_count[cls] = class_gt_count.get(cls, 0) + count
                        
                    for k in [50, 100]:
                        for cls, hits in b_class_hits[k].items():
                            class_hits[k][cls] = class_hits[k].get(cls, 0) + hits

            # --- Final Calculation ---
            
            # 1. R@K (Micro Recall / Global Recall)
            r50 = total_hits[50] / max(1, total_gt_count)
            r100 = total_hits[100] / max(1, total_gt_count)
            
            # 2. mR@K (Macro Recall / Mean Recall per Class)
            mr_values = {50: [], 100: []}
            for cls in class_gt_count:
                gt_cnt = class_gt_count[cls]
                if gt_cnt > 0:
                    mr_values[50].append(class_hits[50].get(cls, 0) / gt_cnt)
                    mr_values[100].append(class_hits[100].get(cls, 0) / gt_cnt)
            
            mr50 = sum(mr_values[50]) / len(mr_values[50]) if mr_values[50] else 0.0
            mr100 = sum(mr_values[100]) / len(mr_values[100]) if mr_values[100] else 0.0

            print(f"\n[Val] Epoch {epoch} SGG Metrics:")
            print(f"  R@50:  {r50:.4f} | R@100:  {r100:.4f}")
            print(f"  mR@50: {mr50:.4f} | mR@100: {mr100:.4f}\n")

        """
        if epoch % cfg["TRAIN"]["VAL_INTERVAL"] == 0:
            model.eval()
            r50_total, mr50_total, n_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for i, (feats, obj_labels, rel_pairs, rel_labels, _) in enumerate(val_loader):
                    feats = feats.to(device)
                    obj_labels = obj_labels.to(device)
                    rel_pairs = rel_pairs.to(device)
                    rel_labels = rel_labels.to(device)
                    out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
                    logits = out["logits"]
                    if i == 0:
                      print(f"\n[DEBUG] Epoch {epoch} Raw Data Inspection:")
                      preds = torch.argmax(logits, dim=1)
                      
                      # 1. Check Label Distribution
                      print(f"Ground Truth Labels (First 20): {rel_labels[:20].cpu().numpy()}")
                      print(f"Predicted Labels    (First 20): {preds[:20].cpu().numpy()}")
                      
                      # 2. Check Background Dominance
                      n_bg = (rel_labels == 0).sum().item()
                      n_fg = (rel_labels > 0).sum().item()
                      print(f"Batch Stat: {n_bg} Backgrounds vs {n_fg} Foreground Relationships")
                
                      mask = rel_labels > 0
                      if mask.sum() > 0:
                          masked_acc = (preds[mask] == rel_labels[mask]).float().mean().item()
                          print(f"Accuracy on FOREGROUND ONLY: {masked_acc:.4f}")
                      else:
                          print("Accuracy on FOREGROUND: N/A (No foreground in this batch)")
                    mask = rel_labels > 0 
                    
                    if mask.sum() > 0:
                        valid_logits = logits[mask]
                        valid_labels = rel_labels[mask]

                        r1 = recall_at_k(valid_logits, valid_labels, k=1)
    
                        
                        r5 = recall_at_k(valid_logits, valid_labels, k=5)
                        
                        
                        mr1 = mean_recall_at_k(
                            valid_logits, 
                            valid_labels, 
                            k=1, # Change this to 1
                            num_classes=cfg["DATASET"]["NUM_REL_CLASSES"]
                        )
                        
                        # Accumulate
                        r50_total += r1 # Reusing variable name, but storing R@1
                        mr50_total += mr1
                        
                        

                        r50_total += recall_at_k(valid_logits, valid_labels, k=cfg["EVAL"]["TOPK"])
                        mr50_total += mean_recall_at_k(
                            valid_logits, 
                            valid_labels, 
                            k=cfg["EVAL"]["TOPK"], 
                            num_classes=cfg["DATASET"]["NUM_REL_CLASSES"]
                        )
                        
                    n_batches += 1
            divisor = max(1, n_batches)
            print(f"[Val] Epoch {epoch} | R@{cfg['EVAL']['TOPK']}: {r50_total / max(1,n_batches):.4f} | mR@{cfg['EVAL']['TOPK']}: {mr50_total / max(1,n_batches):.4f}")
        """
        # Save checkpoint
        ckpt_path = os.path.join(cfg["OUTPUT"]["CKPT_DIR"], f"cfen_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg
        }, ckpt_path)

if __name__ == "__main__":
    main()
