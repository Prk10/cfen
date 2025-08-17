import os
import torch
from torch.utils.data import DataLoader
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

    if cfg["DATASET"]["USE_SYNTHETIC"]:
        ds = SyntheticSGGDataset(
            num_images=max(50, int(cfg["DATASET"]["NUM_SYNTHETIC_IMAGES"] // 5)),
            feat_dim=int(cfg["DATASET"]["FEAT_DIM"]),
            num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
            num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
            seed=cfg["TRAIN"]["SEED"] + 1,
        )
    else:
        ds = NPZSceneGraphDataset(cfg["DATASET"]["DATA_DIR"], split="val")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["NUM_WORKERS"], collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CFEN(
        feat_dim=int(cfg["MODEL"]["FEAT_DIM"]),
        num_obj_classes=int(cfg["DATASET"]["NUM_OBJ_CLASSES"]),
        num_rel_classes=int(cfg["DATASET"]["NUM_REL_CLASSES"]),
        dm_lambda=float(cfg["MODEL"]["DM_LAMBDA"]),
        ema_alpha=float(cfg["MODEL"]["EMA_ALPHA"]),
        fusion=str(cfg["MODEL"]["FUSION"]),
    ).to(device)

    # Find latest checkpoint
    ckpt_dir = cfg["OUTPUT"]["CKPT_DIR"]
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')] if os.path.exists(ckpt_dir) else []
    if ckpts:
        latest = sorted(ckpts)[-1]
        print(f"Loading checkpoint: {latest}")
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        print("No checkpoints found. Evaluating randomly initialized model.")

    model.eval()
    Rk, mRk, n = 0.0, 0.0, 0
    with torch.no_grad():
        for feats, obj_labels, rel_pairs, rel_labels in loader:
            feats = feats.to(device)
            obj_labels = obj_labels.to(device)
            rel_pairs = rel_pairs.to(device)
            rel_labels = rel_labels.to(device)
            out = model(feats, obj_labels, rel_pairs, None, update_ema=False)
            logits = out["logits"]
            Rk += recall_at_k(logits, rel_labels, k=cfg["EVAL"]["TOPK"])
            mRk += mean_recall_at_k(logits, rel_labels, k=cfg["EVAL"]["TOPK"], num_classes=cfg["DATASET"]["NUM_REL_CLASSES"])
            n += 1
    print(f"[Eval] R@{cfg['EVAL']['TOPK']}: {Rk / max(1,n):.4f} | mR@{cfg['EVAL']['TOPK']}: {mRk / max(1,n):.4f}")

if __name__ == "__main__":
    main()
