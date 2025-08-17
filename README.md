# CFEN Baseline (PyTorch)

This repository provides a **reproducible baseline** and a faithful implementation of the **Causal Features Enhancement Network (CFEN)** components for Scene Graph Generation (SGG), focusing on:
- A clean baseline relation head using precomputed object features
- CFEN **fact** and **counterfactual** branches
- **Class-generic** feature moving averages (EMA)
- **DM Loss** and fusion of logits
- Training and evaluation scaffolding
- A **synthetic dataset option** so you can verify your pipeline end-to-end without downloading VG

> This code is designed to be *practical* for reproduction. You can later plug in real precomputed features (e.g., from VG150) as `.npz` files per image.

## Quickstart (with synthetic data)
```bash
# Create and activate an environment (example with conda)
conda create -n cfen python=3.10 -y
conda activate cfen

# Install deps
pip install -r requirements.txt

# Train with synthetic data (tiny demo run)
python train.py --config configs/default.yaml
```

## Using your own (real) data
Place `.npz` files in a folder, each containing:
- `features` (float32) shape: [num_boxes, feat_dim]
- `obj_labels` (int64) shape: [num_boxes]
- `rel_pairs` (int64) shape: [num_pairs, 2] indices into boxes
- `rel_labels` (int64) shape: [num_pairs]

Then run:
```bash
python train.py --config configs/default.yaml DATASET.USE_SYNTHETIC false DATASET.DATA_DIR /path/to/npz_dir DATASET.NUM_OBJ_CLASSES 151 DATASET.NUM_REL_CLASSES 51 MODEL.FEAT_DIM 1024
```

## Evaluate
```bash
python evaluate.py --config configs/default.yaml
```

## Notes
- The implementation uses a simple MLP relation head for clarity. You can swap in more complex context encoders (e.g., BiLSTM) easily.
- The CFEN parts (counterfactual branch, EMA class-generic features, DM loss, fusion) follow the spirit of the paper and are implemented modularly so you can iterate.
