"""
RAPE (Resolution-Adaptive Position Encoding) and ScaleToken.
Injects resolution-awareness into the model.
"""

import torch
import torch.nn as nn
import math


class ScaleToken(nn.Module):
    """
    Learnable scale tokens for multi-resolution awareness.
    Maps resolution buckets to learnable tokens that are prepended/added.
    """

    def __init__(self, embed_dim: int = 256, num_buckets: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_buckets = num_buckets

        # Resolution buckets: 480p, 720p, 1080p, 4K
        self.bucket_boundaries = [480, 720, 1080, 2160]
        self.scale_tokens = nn.Embedding(num_buckets, embed_dim)

    def _get_bucket(self, height: int) -> int:
        """Map height to bucket index."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if height <= boundary:
                return i
        return self.num_buckets - 1

    def forward(self, height, features=None) -> torch.Tensor:
        """
        Get scale embeddings, optionally adding to features.

        Args:
            height: int or [B] tensor — video height for bucket selection
            features: optional [B, D] or [B, N, D] — if given, adds scale token to features

        Returns:
            If features is None: [B, embed_dim] scale token embeddings
            If features given: features with scale token added
        """
        if isinstance(height, torch.Tensor):
            # Batched: height is [B] tensor
            buckets = torch.tensor(
                [self._get_bucket(int(h.item())) for h in height],
                device=height.device,
            )
            tokens = self.scale_tokens(buckets)  # [B, embed_dim]
        else:
            bucket = self._get_bucket(int(height))
            tokens = self.scale_tokens(
                torch.tensor(bucket, device=self.scale_tokens.weight.device)
            ).unsqueeze(0)  # [1, embed_dim]

        if features is None:
            return tokens

        if features.dim() == 2:
            return features + tokens
        elif features.dim() == 3:
            return features + tokens.unsqueeze(1)
        return features


class RAPE(nn.Module):
    """
    Resolution-Adaptive Position Encoding.
    Generates position encodings that adapt to the input resolution.
    """

    def __init__(self, embed_dim: int = 256, max_h: int = 2160, max_w: int = 3840):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable frequency components
        self.freq = nn.Parameter(torch.randn(embed_dim // 2) * 0.01)
        self.phase = nn.Parameter(torch.randn(embed_dim // 2) * 0.01)

    def forward(self, height, width, device=None) -> torch.Tensor:
        """
        Generate resolution-adaptive encoding.

        Args:
            height: int, float, or [B] tensor with spatial height
            width: int, float, or [B] tensor with spatial width
            device: target device

        Returns:
            pe: [B, embed_dim] or [1, embed_dim] position encoding
        """
        if device is None:
            device = self.freq.device

        # Normalize resolution — handle both scalar and tensor inputs
        if isinstance(height, torch.Tensor):
            h_norm = height.float().to(device) / 2160.0  # [B]
            w_norm = width.float().to(device) / 3840.0   # [B]
            # [B, D//2]
            enc_sin = torch.sin(h_norm.unsqueeze(1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0))
            enc_cos = torch.cos(w_norm.unsqueeze(1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0))
            pe = torch.cat([enc_sin, enc_cos], dim=1)  # [B, embed_dim]
        else:
            h_norm = float(height) / 2160.0
            w_norm = float(width) / 3840.0
            pos = torch.tensor([h_norm, w_norm], device=device, dtype=torch.float32)
            enc_sin = torch.sin(pos[0] * self.freq + self.phase)
            enc_cos = torch.cos(pos[1] * self.freq + self.phase)
            pe = torch.cat([enc_sin, enc_cos], dim=0).unsqueeze(0)

        return pe


class ResolutionConditioner(nn.Module):
    """
    FiLM-style Resolution Conditioner.

    Maps input resolution (height, width) to scale & bias vectors that
    *directly modulate* backbone feature representations (e.g. 1024-dim).

    Key insight: Instead of concatenating a tiny 48-dim side feature that
    gets bypassed in repro_mlp mode, we modulate the main feature stream
    *before* FR interaction and the scoring head, ensuring resolution
    information always participates in the prediction.

    Architecture:
        (h_norm, w_norm) → [2] → MLP → (scale [feat_dim], bias [feat_dim])
        feat_out = feat * (1 + scale) + bias

    The (1 + scale) formulation ensures the identity mapping at initialization
    (scale init ≈ 0, bias init ≈ 0), so the model starts from the same
    point as without conditioning and can smoothly learn to use resolution.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_dim: int = 256,
        ref_h: float = 2160.0,
        ref_w: float = 3840.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.ref_h = ref_h
        self.ref_w = ref_w

        # Shared encoder: (h_norm, w_norm) → hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Separate heads for scale and bias
        self.scale_head = nn.Linear(hidden_dim, feat_dim)
        self.bias_head = nn.Linear(hidden_dim, feat_dim)

        # Init: scale → 0, bias → 0 → identity at start
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(
        self,
        height: torch.Tensor,
        width: torch.Tensor,
        features: torch.Tensor = None,
    ) -> dict:
        """
        Compute resolution-conditioned scale and bias, optionally apply to features.

        Args:
            height: [B] tensor of video heights
            width:  [B] tensor of video widths
            features: optional [B, D] or [B*P, D] features to modulate

        Returns:
            dict with keys:
                'scale': [B, feat_dim] — scale vector (add 1 before multiplying)
                'bias':  [B, feat_dim] — bias vector
                'features': modulated features (only if input features provided)
        """
        device = height.device
        h_norm = height.float().to(device) / self.ref_h  # [B]
        w_norm = width.float().to(device) / self.ref_w    # [B]
        res_input = torch.stack([h_norm, w_norm], dim=-1)  # [B, 2]

        hidden = self.encoder(res_input)           # [B, hidden_dim]
        scale = self.scale_head(hidden)            # [B, feat_dim]
        bias = self.bias_head(hidden)              # [B, feat_dim]

        result = {'scale': scale, 'bias': bias}

        if features is not None:
            B = scale.shape[0]
            if features.shape[0] != B and features.shape[0] % B == 0:
                # features is [B*P, D] or [B*T, D] — expand scale/bias
                repeat = features.shape[0] // B
                scale = scale.unsqueeze(1).expand(B, repeat, -1).reshape(-1, self.feat_dim)
                bias = bias.unsqueeze(1).expand(B, repeat, -1).reshape(-1, self.feat_dim)
            result['features'] = features * (1.0 + scale) + bias

        return result
