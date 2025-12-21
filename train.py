import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config, update_config, parse_args
from utils.seed import set_seed
from utils.metrics import recall_at_k, mean_recall_at_k
from data.sgg_dataset import NPZSceneGraphDataset, SyntheticSGGDataset, collate_fn
from models.cfen import CFEN

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

    os.makedirs(cfg["OUTPUT"]["CKPT_DIR"], exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg["TRAIN"]["EPOCHS"] + 1):
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
            r50_total, mr50_total, n_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for feats, obj_labels, rel_pairs, rel_labels, _ in val_loader:
                    feats = feats.to(device)
                    obj_labels = obj_labels.to(device)
                    rel_pairs = rel_pairs.to(device)
                    rel_labels = rel_labels.to(device)
                    out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
                    logits = out["logits"]
                    r50_total += recall_at_k(logits, rel_labels, k=cfg["EVAL"]["TOPK"])
                    mr50_total += mean_recall_at_k(logits, rel_labels, k=cfg["EVAL"]["TOPK"], num_classes=cfg["DATASET"]["NUM_REL_CLASSES"])
                    n_batches += 1
            print(f"[Val] Epoch {epoch} | R@{cfg['EVAL']['TOPK']}: {r50_total / max(1,n_batches):.4f} | mR@{cfg['EVAL']['TOPK']}: {mr50_total / max(1,n_batches):.4f}")

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
