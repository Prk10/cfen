import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_head import RelationHead, CFENRelationHead

class CFEN(nn.Module):
    """
    Causal Features Enhancement Network (baseline implementation)
    - Fact branch: uses instance-specific features
    - Counterfactual branch: uses class-generic (EMA) features
    - L_sp = L_f - L_cf (object-specific influence)
    - Fusion (sum): L_f + L_sp
    - DM Loss: encourages L_sp to emphasize the true predicate

    This module expects per-image inputs (batch_size=1 in training loop).
    """
    def __init__(self, feat_dim, num_obj_classes, num_rel_classes, dm_lambda=0.4, ema_alpha=5e-4, fusion='sum'):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.dm_lambda = dm_lambda
        self.ema_alpha = ema_alpha
        self.fusion = fusion

        # Relation heads (share weights between branches for faithfulness/simplicity)
        #self.rel_head = RelationHead(feat_dim, num_rel_classes)

        self.rel_head = CFENRelationHead(feat_dim, num_rel_classes)

        # Class-generic feature memory (EMA)
        self.register_buffer("class_mean", torch.zeros(num_obj_classes, feat_dim))  # [C_obj, D]
        self.register_buffer("class_inited", torch.zeros(num_obj_classes, dtype=torch.bool))

    @torch.no_grad()
    def _ema_update(self, feats, obj_labels):
        # feats: [N_boxes, D], obj_labels: [N_boxes]
        if feats.numel() == 0:
            return
        for c in obj_labels.unique():
            c = c.item()
            mask = (obj_labels == c)
            if not mask.any():
                continue
            f_mean = feats[mask].mean(dim=0)  # [D]
            if not self.class_inited[c]:
                self.class_mean[c] = f_mean
                self.class_inited[c] = True
            else:
                self.class_mean[c] = (1.0 - self.ema_alpha) * self.class_mean[c] + self.ema_alpha * f_mean

    def forward(self, feats, obj_labels, rel_pairs, rel_labels=None, update_ema=False):
        """
        feats: [N_boxes, D]
        obj_labels: [N_boxes]
        rel_pairs: [N_pairs, 2] (subject_idx, object_idx)
        rel_labels: [N_pairs] (optional, required for loss)
        """
        device = feats.device
        if update_ema:
            self._ema_update(feats, obj_labels)

        # Gather subject/object features
        subj_idx = rel_pairs[:, 0]
        obj_idx  = rel_pairs[:, 1]
        subj_feats = feats[subj_idx]  # [N_pairs, D]
        obj_feats  = feats[obj_idx]   # [N_pairs, D]

        # Fact branch
        #L_f = self.rel_head(subj_feats, obj_feats)  # [N_pairs, C_rel]
        L_f = self.rel_head(subj_feats, obj_feats, full_image_feats=feats)

        # Counterfactual branch: replace with class-generic (EMA) features
        subj_labels = obj_labels[subj_idx]
        obj_labels_ = obj_labels[obj_idx]

        # For classes not yet initialized, fallback to zeros (or instance feats)
        subj_gen = self.class_mean[subj_labels]  # [N_pairs, D]
        obj_gen  = self.class_mean[obj_labels_]  # [N_pairs, D]

        #L_cf = self.rel_head(subj_gen, obj_gen)
        full_image_gen_feats = self.class_mean[obj_labels] 
        L_cf = self.rel_head(subj_gen, obj_gen, full_image_feats=full_image_gen_feats)

        # Object-specific influence
        L_sp = L_f - L_cf

        # Fusion
        if self.fusion == 'sum':
            logits = L_f + L_sp
        elif self.fusion == 'fact_only':
            logits = L_f
        else:
            logits = L_f + L_sp  # default

        outputs = {
            "logits": logits,
            "L_f": L_f,
            "L_cf": L_cf,
            "L_sp": L_sp,
        }

        # Loss
        loss = None
        if rel_labels is not None:
            ce = F.cross_entropy(L_f, rel_labels)  # CE on fact branch
            l_sp_log_probs = F.log_softmax(L_sp, dim=1)
            target_dist = F.one_hot(rel_labels, num_classes=self.num_rel_classes).float()
            dm = F.kl_div(l_sp_log_probs, target_dist, reduction='batchmean')

            loss = ce + self.dm_lambda * dm
            outputs.update({"loss": loss, "loss_ce": ce, "loss_dm": dm})

        return outputs
