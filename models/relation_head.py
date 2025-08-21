import torch
import torch.nn as nn

class RelationHead(nn.Module):
    """
    Simple MLP relation classifier operating on concatenated subject/object features.
    Can be swapped  with a BiLSTM/TreeLSTM context encoder later.
    """
    def __init__(self, feat_dim: int, num_rel_classes: int):
        super().__init__()
        hidden = max(512, feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_rel_classes)
        )

    def forward(self, subj_feats, obj_feats):
        x = torch.cat([subj_feats, obj_feats], dim=-1)  # [N_pairs, 2*D]
        return self.net(x)  # [N_pairs, C_rel]
