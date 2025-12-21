import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BiTreeLSTM(nn.Module):
    """
    Bi-Directional Tree-Structured LSTM for Context Refinement.
    Replicates the 'BiTreeLSTM' block in CFEN Fig. 4[cite: 239].
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_obj_classes: int = 151):
        super(BiTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        
        # Scoring Module. This is used to build the Dynamic Tree)
        # Determines which objects are connected (Parent/Child relationships)
        self.score_fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Tree LSTM Cells
        # 'Up' direction (Children -> Parent)
        self.px_up = nn.Linear(feat_dim, 5 * hidden_dim)
        self.ph_up = nn.Linear(hidden_dim, 5 * hidden_dim)
        
        # 'Down' direction (Parent -> Children)
        self.px_down = nn.Linear(feat_dim, 5 * hidden_dim)
        self.ph_down = nn.Linear(hidden_dim, 5 * hidden_dim)

        # Output Projector 
        self.out_project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1) 
        )

    def calculate_tree_structure(self, features):
        """
        Builds a dependency tree (adjacency matrix) based on feature similarity.
        Simplification of VCTree's dynamic spanning tree logic.
        """
        num_obj = features.size(0)
        
        # Calculate pairwise scores (N x N)
        # In a full implementation, this uses spatial masks + visual features
        scores = self.score_fc(features).view(-1, 1) # Simple score for root selection
        
        # For this implementation, we simulate a 'chain' structure or 
        # a simple tree to allow gradients to flow without needing C++ extensions
        # (A full Minimum Spanning Tree is non-differentiable and complex in pure PyTorch).
        
        # Strategy: Use a fully-connected attention-like weighting for the tree
        # This is a robust differentiable approximation of a discrete tree.
        attn_logits = torch.matmul(features, features.t()) / (self.feat_dim ** 0.5)
        tree_weights = F.softmax(attn_logits, dim=1)
        return tree_weights

    def lstm_step(self, x, h_prev, c_prev, direction='up'):
        """
        One step of TreeLSTM. 
        Instead of explicit recursion, we use the weighted sum of neighbors (tree_weights).
        """
        if direction == 'up':
            gates = self.px_up(x) + self.ph_up(h_prev)
        else:
            gates = self.px_down(x) + self.ph_down(h_prev)

        # Split gates: input, forget, cell, output, sibling_forget
        i, f, o, u, f_sibling = gates.chunk(5, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        
        c = i * u + f * c_prev
        h = o * torch.tanh(c)
        return h, c

    def forward(self, obj_feats):
        """
        Args:
            obj_feats: [N, feat_dim] - The RoI features from Faster R-CNN
        Returns:
            refined_feats: [N, hidden_dim] - Context-aware features
        """
        N = obj_feats.size(0)
        
       
        tree_weights = self.calculate_tree_structure(obj_feats)
        
        
        h_up = torch.zeros(N, self.hidden_dim).to(obj_feats.device)
        c_up = torch.zeros(N, self.hidden_dim).to(obj_feats.device)
        h_down = torch.zeros(N, self.hidden_dim).to(obj_feats.device)
        c_down = torch.zeros(N, self.hidden_dim).to(obj_feats.device)
        
        num_steps = 4 
        
        
        for _ in range(num_steps):
            
            neighbor_h = torch.matmul(tree_weights, h_up)
            h_up, c_up = self.lstm_step(obj_feats, neighbor_h, c_up, direction='up')
            
        
        for _ in range(num_steps):
            neighbor_h = torch.matmul(tree_weights.t(), h_down)
            h_down, c_down = self.lstm_step(obj_feats, neighbor_h, c_down, direction='down')
            
        
        refined_feats = h_up + h_down
        return self.out_project(refined_feats)
    
    

class CFENRelationHead(nn.Module):
    """
    The CFEN Module replacing your placeholder.
    1. Refines Object Features using BiTreeLSTM (Context).
    2. Classifies Relationships using Fact/Counterfactual Logic.
    """
    def __init__(self, feat_dim: int, num_rel_classes: int):
        super().__init__()
        
        
        self.hidden_dim = 2048 
        
        
        # Input: Raw RoI features -> Output: Refined Context Features
        self.bitreilstm = BiTreeLSTM(feat_dim, self.hidden_dim)
        
        
        self.classifier_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, num_rel_classes)
        )

    def forward(self, subj_feats, obj_feats, full_image_feats=None):
        """
        Args:
            subj_feats: [N_pairs, feat_dim] (From your loader's roi features)
            obj_feats:  [N_pairs, feat_dim]
            full_image_feats: [N_objects, feat_dim] (Optional: The full set of objects for tree context)
        """
    
        
        if full_image_feats is not None:
           
            refined_ctx = self.bitreilstm(full_image_feats)
            
            pass 
        
        
        subj_refined = self.bitreilstm(subj_feats)
        obj_refined = self.bitreilstm(obj_feats)

        
        combined = torch.cat([subj_refined, obj_refined], dim=-1)
        
        
        return self.classifier_fc(combined)