"""
Fusion Head: combines outputs from multiple branches.
Supports: late_concat_mlp, gated_fusion, inject_then_head.
Auto-adapts to missing branches.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class FusionHead(nn.Module):
    """
    Multi-branch fusion head.

    Dynamically adapts to available branch outputs.
    """

    def __init__(
        self,
        fusion_type: str = 'late_concat_mlp',
        branch_dims: Dict[str, int] = None,
        hidden_dim: int = 256,
        out_dim: int = 1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.branch_dims = branch_dims or {}
        self.hidden_dim = hidden_dim

        total_dim = sum(self.branch_dims.values())

        if fusion_type == 'late_concat_mlp':
            self.head = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        elif fusion_type == 'gated_fusion':
            self.gates = nn.ModuleDict()
            self.projections = nn.ModuleDict()
            for name, dim in self.branch_dims.items():
                self.projections[name] = nn.Linear(dim, hidden_dim)
                self.gates[name] = nn.Sequential(
                    nn.Linear(dim, 1),
                    nn.Sigmoid(),
                )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )

        elif fusion_type == 'inject_then_head':
            self.head = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, branch_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            branch_outputs: dict of {branch_name: tensor [B, dim]}

        Returns:
            score: [B, 1] or [B]
        """
        if self.fusion_type == 'late_concat_mlp':
            # Concatenate available branches
            feats = []
            for name in self.branch_dims:
                if name in branch_outputs:
                    feats.append(branch_outputs[name])
                else:
                    # Zero-fill missing branch
                    B = list(branch_outputs.values())[0].shape[0]
                    feats.append(torch.zeros(B, self.branch_dims[name],
                                             device=list(branch_outputs.values())[0].device))
            combined = torch.cat(feats, dim=-1)
            return self.head(combined).squeeze(-1)

        elif self.fusion_type == 'gated_fusion':
            projected = []
            total_gate = 0
            for name in self.branch_dims:
                if name in branch_outputs:
                    feat = branch_outputs[name]
                    gate = self.gates[name](feat)
                    proj = self.projections[name](feat)
                    projected.append(gate * proj)
                    total_gate = total_gate + gate
            if projected:
                combined = sum(projected) / (total_gate + 1e-8)
            else:
                B = 1
                combined = torch.zeros(B, self.hidden_dim)
            return self.head(combined).squeeze(-1)

        elif self.fusion_type == 'inject_then_head':
            feats = []
            for name in self.branch_dims:
                if name in branch_outputs:
                    feats.append(branch_outputs[name])
                else:
                    B = list(branch_outputs.values())[0].shape[0]
                    feats.append(torch.zeros(B, self.branch_dims[name],
                                             device=list(branch_outputs.values())[0].device))
            combined = torch.cat(feats, dim=-1)
            return self.head(combined).squeeze(-1)
