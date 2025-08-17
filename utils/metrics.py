import torch
from collections import defaultdict

@torch.no_grad()
def recall_at_k(logits, gt_labels, k=50):
    # logits: [N_pairs, C], gt_labels: [N_pairs]
    if logits.numel() == 0:
        return 0.0
    topk = logits.topk(min(k, logits.size(1)), dim=1).indices  # [N_pairs, k]
    correct = (topk == gt_labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()

@torch.no_grad()
def mean_recall_at_k(logits, gt_labels, k=50, num_classes=None, ignore_bg=True, bg_index=0):
    if logits.numel() == 0:
        return 0.0
    pred_topk = logits.topk(min(k, logits.size(1)), dim=1).indices
    cls_hits = defaultdict(list)
    for i in range(gt_labels.numel()):
        g = gt_labels[i].item()
        if ignore_bg and g == bg_index:
            continue
        hit = (pred_topk[i] == g).any().item()
        cls_hits[g].append(hit)
    if not cls_hits:
        return 0.0
    recalls = [sum(v)/len(v) for v in cls_hits.values() if len(v) > 0]
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)
