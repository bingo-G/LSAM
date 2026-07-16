"""
Semantic Branch: ConvNeXtV2 / PE / SwinV2 + downsampled global semantic.

Supports two modes via `head_mode`:
  - 'projection' (default/legacy): Linear(bb_dim→out_dim)+GELU, feeds into FusionHead
  - 'repro_mlp': standalone 1024→512→128→1 MLP head with Sigmoid (PE-repro style)

When `head_mode='repro_mlp'`, the branch can produce a final score directly,
enabling repro-level performance inside the HMF framework.

FR interaction is performed at **backbone feature level** (bb_dim space) before the head,
so both projection and repro_mlp modes can benefit from FR information.
"""

import logging
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ..backbones import build_backbone

logger = logging.getLogger('hmf_vqa.semantic_branch')


# ---------------------------------------------------------------------------
# ReNormalize: ImageNet normalization → PE normalization (0.5, 0.5)
# ---------------------------------------------------------------------------
class ReNormalize(nn.Module):
    """
    Convert source-normalized tensor into target-normalized tensor:
      x_tgt = x_src * (std_src / std_tgt) + (mean_src - mean_tgt * std_src / std_tgt)
    """

    def __init__(
        self,
        mean_src=(0.485, 0.456, 0.406),
        std_src=(0.229, 0.224, 0.225),
        mean_tgt=(0.5, 0.5, 0.5),
        std_tgt=(0.5, 0.5, 0.5),
    ):
        super().__init__()
        mean_src_t = torch.tensor(mean_src, dtype=torch.float32)
        std_src_t = torch.tensor(std_src, dtype=torch.float32)
        mean_tgt_t = torch.tensor(mean_tgt, dtype=torch.float32)
        std_tgt_t = torch.tensor(std_tgt, dtype=torch.float32)

        self.register_buffer(
            "weight", (std_src_t / std_tgt_t).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "bias",
            (mean_src_t - mean_tgt_t * (std_src_t / std_tgt_t)).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W]"""
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# CrossAttention + AttentionPoolingBlock (temporal pooling)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Cross-attention block used by temporal pooling."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        bsz, nq, dim = x_q.shape
        nk = x_kv.shape[1]

        q = self.q(x_q).reshape(bsz, nq, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, nq, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AttentionPoolingBlock(nn.Module):
    """Temporal attention pooling from [B, T, D] to [B, D]."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.mean(dim=1, keepdim=True)
        q = self.norm_q(q)
        kv = self.norm_kv(x)
        out = self.cross_attn(q, kv)
        return out.squeeze(1)


class SemanticBranch(nn.Module):
    """
    Semantic branch using ConvNeXtV2, PE, or SwinV2 on resized frames.

    Ablation switches (via semantic_cfg dict):
      - apply_renorm: bool, add ReNormalize layer (PE needs this)
      - temporal_pool: 'tsm_mean' | 'attention' | 'mean'
      - head_mode: 'projection' | 'repro_mlp'
      - score_sigmoid: bool, add Sigmoid at MLP output (repro_mlp only)
      - attn_heads: int, attention pooling heads
      - head_hidden1/head_hidden2: int, MLP hidden dims (repro_mlp only)
      - dropout: float (repro_mlp only)
      - patch_reduce: 'mean' | 'median' | 'max' (multi-patch reduction)
      - multipatch_input: bool, accept 6D [B,P,3,T,H,W] input

    FR interaction:
      Performed at backbone feature level (bb_dim space) before head.
      Both projection and repro_mlp modes can use FR interaction.
      Modes: 'diff_prod' (cat[dis, ref, |diff|, prod] → 4*bb_dim → bb_dim)
             'diff_only' (dis - ref, no extra params)
    """

    def __init__(
        self,
        backbone_name: str = 'convnextv2_tiny',
        temporal: str = 'tsm',
        out_dim: int = 256,
        pretrained: bool = False,
        pe_weights: str = None,
        fr_interaction: str = 'diff_prod',
        semantic_cfg: dict = None,
    ):
        super().__init__()
        self.fr_interaction = fr_interaction
        scfg = semantic_cfg or {}

        # ---- Backbone (unified factory) ----
        img_size = int(scfg.get('_target_size', 224))
        rope_res_scale = bool(scfg.get('rope_resolution_scale', False))
        pec = bool(scfg.get('patch_embed_cond', False))
        res_tok = bool(scfg.get('resolution_token', False))
        sdr = float(scfg.get('stochastic_depth_rate', 0.0))
        self.backbone = build_backbone(
            variant=backbone_name,
            pretrained=pretrained,
            pe_weights=pe_weights,
            img_size=img_size,
            rope_resolution_scale=rope_res_scale,
            patch_embed_cond=pec,
            resolution_token=res_tok,
            stochastic_depth_rate=sdr,
        )
        self.rope_resolution_scale = rope_res_scale
        self.patch_embed_cond = pec
        self.resolution_token = res_tok

        bb_dim = self.backbone.out_dim
        self.bb_dim = bb_dim

        # ---- TOPIQ multi-layer FR (PE only) ----
        # Use PE encoder layers [1, 4, 7, 10] (1-based) by default,
        # run FR interaction per layer, then concat and project back to bb_dim.
        layer_ids_1based = scfg.get('topiq_layer_ids', [1, 4, 7, 10])
        self.topiq_layer_ids = [max(0, int(i) - 1) for i in layer_ids_1based]
        self.use_topiq_multilayer = bool(
            scfg.get('topiq_multilayer', True)
            and self.fr_interaction == 'topiq_deep'
            and hasattr(self.backbone, 'forward_intermediate_layers')
            and len(self.topiq_layer_ids) > 0
        )
        if self.use_topiq_multilayer:
            self.topiq_layer_merge = nn.Sequential(
                nn.Linear(bb_dim * len(self.topiq_layer_ids), bb_dim),
                nn.GELU(),
            )

        # ---- ReNormalize (ablation: default True for PE, False for others) ----
        is_pe = backbone_name.startswith('pe') or backbone_name.startswith('PE')
        self.apply_renorm = bool(scfg.get('apply_renorm', is_pe))
        if self.apply_renorm:
            # Determine source normalization from colorspace mode
            cs = str(scfg.get('_colorspace', 'bt709_imagenet')).lower()
            if cs.endswith('_clip'):
                src_mean = (0.48145466, 0.4578275, 0.40821073)
                src_std = (0.26862954, 0.26130258, 0.27577711)
            else:
                src_mean = (0.485, 0.456, 0.406)
                src_std = (0.229, 0.224, 0.225)
            self.renorm = ReNormalize(
                mean_src=src_mean, std_src=src_std,
                mean_tgt=(0.5, 0.5, 0.5), std_tgt=(0.5, 0.5, 0.5),
            )
        else:
            self.renorm = nn.Identity()

        # ---- Temporal pooling mode ----
        self.temporal_pool_mode = str(scfg.get('temporal_pool', 'tsm_mean')).lower()
        self.tsm_fraction = 1 / 8

        if self.temporal_pool_mode == 'attention':
            attn_heads = int(scfg.get('attn_heads', 8))
            self.attention_pool = AttentionPoolingBlock(dim=bb_dim, num_heads=attn_heads)

        # ---- Temporal adaptor (Round 9 + Round 11) ----
        self.temporal_adaptor_mode = str(scfg.get('temporal_adaptor', 'none')).lower()
        if self.temporal_adaptor_mode == 'self_attn':
            from ..fusion.temporal_adaptor import TemporalSelfAttentionAdaptor
            self.temporal_adaptor = TemporalSelfAttentionAdaptor(dim=bb_dim, num_heads=8)
        elif self.temporal_adaptor_mode == 'conv1d':
            from ..fusion.temporal_adaptor import TemporalConv1DAdaptor
            self.temporal_adaptor = TemporalConv1DAdaptor(dim=bb_dim, kernel_size=3)
        elif self.temporal_adaptor_mode == 'diff_fusion':
            from ..fusion.temporal_adaptor import TemporalDiffFusion
            self.temporal_adaptor = TemporalDiffFusion(dim=bb_dim)
        elif self.temporal_adaptor_mode == 'burst_two_level':
            from ..fusion.temporal_adaptor import BurstTwoLevelAggregator
            self.temporal_adaptor = BurstTwoLevelAggregator(dim=bb_dim, n_bursts=4, frames_per_burst=2)
        elif self.temporal_adaptor_mode == 'segment_gated':
            from ..fusion.temporal_adaptor import SegmentGatedFusion
            self.temporal_adaptor = SegmentGatedFusion(dim=bb_dim, n_segments=2, frames_per_seg=4)
        elif self.temporal_adaptor_mode == 'shared_temporal_mixer':
            from ..fusion.temporal_adaptor import SharedTemporalMixer
            self.temporal_adaptor = SharedTemporalMixer(dim=bb_dim, max_frames=32)
        else:
            self.temporal_adaptor = None

        # ---- Head mode ----
        self.head_mode = str(scfg.get('head_mode', 'projection')).lower()

        if self.head_mode == 'repro_mlp':
            # PE-repro style: bb_dim → h1 → h2 → 1 with optional sigmoid
            drop = float(scfg.get('dropout', 0.1))
            h1 = int(scfg.get('head_hidden1', 512))
            h2 = int(scfg.get('head_hidden2', 128))
            score_sigmoid = bool(scfg.get('score_sigmoid', True))
            layers = [
                nn.Linear(bb_dim, h1),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(h1, h2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(h2, 1),
            ]
            if score_sigmoid:
                layers.append(nn.Sigmoid())
            self.head = nn.Sequential(*layers)
            # In repro_mlp mode, out_dim is bb_dim (feature) for fusion compatibility
            # But the branch also produces its own score
            self.out_dim = bb_dim
            self._init_repro_head_weights()
        else:
            # Legacy projection mode
            self.head = nn.Sequential(
                nn.Linear(bb_dim, out_dim),
                nn.GELU(),
            )
            self.out_dim = out_dim

        # ---- FR interaction at backbone feature level (bb_dim space) ----
        # This allows both projection and repro_mlp modes to use FR interaction.
        if self.fr_interaction == 'diff_prod':
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 4, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'concat_mlp':
            # Two-layer MLP on [dis, ref, |diff|] — richer than diff_prod with deeper network
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 3, bb_dim),
                nn.GELU(),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'cosine_diff':
            # Cosine similarity channel-wise + absolute diff, compressed back to bb_dim
            # Provides scale-invariant comparison
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 2, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'diff_affine':
            # FiLM-style: ref generates scale & bias to modulate dis features
            self.fr_scale = nn.Sequential(
                nn.Linear(bb_dim, bb_dim),
                nn.Sigmoid(),
            )
            self.fr_bias = nn.Linear(bb_dim, bb_dim)
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 2, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'topiq_like':
            # Full TOPIQ-style: [dis, ref, diff] concat + diff-gated weighting + 2-layer MLP
            self.fr_gate = nn.Sequential(
                nn.Linear(bb_dim, bb_dim // 4),
                nn.GELU(),
                nn.Linear(bb_dim // 4, 1),
                nn.Sigmoid(),
            )
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 3, bb_dim),
                nn.GELU(),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'concat_mlp_deep':
            # Deeper 3-layer MLP on [dis, ref, |diff|] — more capacity than concat_mlp
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 3, bb_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'diff_affine_res':
            # diff_affine + residual: FiLM modulation with skip connection
            self.fr_scale = nn.Sequential(
                nn.Linear(bb_dim, bb_dim),
                nn.Sigmoid(),
            )
            self.fr_bias = nn.Linear(bb_dim, bb_dim)
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 2, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'topiq_deep':
            # topiq_like with wider gate (D/2 instead of D/4) + 3-layer MLP
            self.fr_gate = nn.Sequential(
                nn.Linear(bb_dim, bb_dim // 2),
                nn.GELU(),
                nn.Linear(bb_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.fr_fusion = nn.Sequential(
                nn.Linear(bb_dim * 3, bb_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'ensemble_vote':
            # Ensemble of concat_mlp + diff_affine, average their outputs
            # Path A: concat_mlp style
            self.fr_fusion_a = nn.Sequential(
                nn.Linear(bb_dim * 3, bb_dim),
                nn.GELU(),
                nn.Linear(bb_dim, bb_dim),
                nn.GELU(),
            )
            # Path B: diff_affine style
            self.fr_scale_b = nn.Sequential(
                nn.Linear(bb_dim, bb_dim),
                nn.Sigmoid(),
            )
            self.fr_bias_b = nn.Linear(bb_dim, bb_dim)
            self.fr_fusion_b = nn.Sequential(
                nn.Linear(bb_dim * 2, bb_dim),
                nn.GELU(),
            )
        elif self.fr_interaction == 'diff_only':
            pass  # Simple subtraction, no learnable parameters
        else:
            raise ValueError(
                f"Unknown FR interaction '{self.fr_interaction}'. "
                f"Supported: diff_prod, concat_mlp, cosine_diff, diff_affine, "
                f"topiq_like, concat_mlp_deep, diff_affine_res, topiq_deep, ensemble_vote, diff_only"
            )

        # ---- Multi-patch support ----
        self.multipatch_input = bool(scfg.get('multipatch_input', False))
        self.patch_reduce = str(scfg.get('patch_reduce', 'mean')).lower()
        self.patch_forward_chunk_size = int(scfg.get('patch_forward_chunk_size', 0))

        # ---- Feature Whitening (ZCA-like decorrelation) ----
        # When enabled, applies a learnable whitening transform after backbone output
        # to decorrelate feature dimensions, improving domain generalization.
        self.feature_whitening = bool(scfg.get('feature_whitening', False))
        if self.feature_whitening:
            self.whiten_norm = nn.LayerNorm(bb_dim, elementwise_affine=False)
            self.whiten_proj = nn.Linear(bb_dim, bb_dim, bias=False)
            # Initialize as identity for stable start
            nn.init.eye_(self.whiten_proj.weight)

        # ---- MSS (Multi-Scale Spatial) fusion ----
        # When enabled, receives both global resize features and local GMS patch features,
        # fusing them via cross-attention (global as query, local patches as key/value).
        self.use_mss = bool(scfg.get('mss', False))
        self.mss_fusion_mode = str(scfg.get('mss_fusion', 'cross_attn')).lower()
        if self.use_mss:
            attn_heads = int(scfg.get('attn_heads', 8))
            if self.mss_fusion_mode == 'cross_attn':
                # Global features attend to local patch features
                self.mss_norm_q = nn.LayerNorm(bb_dim)
                self.mss_norm_kv = nn.LayerNorm(bb_dim)
                self.mss_cross_attn = CrossAttention(
                    dim=bb_dim, num_heads=attn_heads, qkv_bias=True
                )
                self.mss_merge = nn.Sequential(
                    nn.Linear(bb_dim * 2, bb_dim),
                    nn.GELU(),
                )
            elif self.mss_fusion_mode == 'concat':
                # Simple concat + project: [global, mean_local] → bb_dim
                self.mss_merge = nn.Sequential(
                    nn.Linear(bb_dim * 2, bb_dim),
                    nn.GELU(),
                )
            elif self.mss_fusion_mode == 'add':
                # Weighted addition with learnable scale
                self.mss_local_scale = nn.Parameter(torch.ones(1) * 0.5)
            else:
                raise ValueError(f"Unknown MSS fusion mode '{self.mss_fusion_mode}'. "
                                 "Supported: cross_attn, concat, add")

        # ---- Cross-Clip Temporal Interaction (multi_clip_4x8 continued training) ----
        cross_clip_temporal_name = str(scfg.get('cross_clip_temporal', 'none')).lower().strip()
        self.cross_clip_temporal_module = None
        self.num_temporal_clips = int(scfg.get('num_temporal_clips', 4))
        if cross_clip_temporal_name not in ('none', ''):
            from ..fusion.cross_clip_temporal import build_cross_clip_temporal
            self.cross_clip_temporal_module = build_cross_clip_temporal(
                name=cross_clip_temporal_name,
                dim=bb_dim,
                num_clips=self.num_temporal_clips,
            )
            logger.info(
                "[SemanticBranch] Cross-clip temporal module: %s (dim=%d, clips=%d, params=%.2fM)",
                cross_clip_temporal_name, bb_dim, self.num_temporal_clips,
                sum(p.numel() for p in self.cross_clip_temporal_module.parameters()) / 1e6,
            )

    def _init_repro_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_tsm(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if self.temporal_pool_mode != 'tsm_mean' or T <= 1:
            return x
        BT, D = x.shape
        B = BT // T
        x = x.view(B, T, D)
        fold = int(D * self.tsm_fraction)
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        return out.view(BT, D)

    def _apply_temporal_interaction(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Apply TSM + optional temporal adaptor.

        TSM is always applied first (when mode=tsm_mean), then the adaptor
        (self_attn or conv1d) operates on the TSM-shifted features.

        Args:
            x: [B*T, D] flattened temporal features
            T: number of temporal frames

        Returns:
            [B*T, D] with temporal interaction applied
        """
        # Step 1: TSM shift (zero-param, always applied in tsm_mean mode)
        x = self._apply_tsm(x, T)

        # Step 2: Temporal adaptor (learnable, applied on top of TSM)
        if self.temporal_adaptor is not None and T > 1:
            BT, D = x.shape
            B = BT // T
            x_3d = x.view(B, T, D)       # [B, T, D]
            x_3d = self.temporal_adaptor(x_3d)  # [B, T, D]
            x = x_3d.reshape(BT, D)       # [B*T, D]

        return x

    def _reduce_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce patch dimension: [B, P, D] -> [B, D]."""
        if self.patch_reduce == 'max':
            return x.max(dim=1).values
        if self.patch_reduce == 'median':
            return x.median(dim=1).values
        return x.mean(dim=1)

    def _reduce_patch_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Reduce patch scores: [B, P] -> [B]."""
        if self.patch_reduce == 'max':
            return scores.max(dim=1).values
        if self.patch_reduce == 'median':
            return scores.median(dim=1).values
        return scores.mean(dim=1)

    def _apply_fr_interaction(self, dis_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        """
        Apply FR interaction at backbone feature level.

        Args:
            dis_feat: [*, bb_dim] distorted features
            ref_feat: [*, bb_dim] reference features (same shape as dis_feat)

        Returns:
            fused: [*, bb_dim] FR-fused features
        """
        if self.fr_interaction == 'diff_only':
            return dis_feat - ref_feat
        elif self.fr_interaction == 'diff_prod':
            diff = torch.abs(dis_feat - ref_feat)
            prod = dis_feat * ref_feat
            return self.fr_fusion(torch.cat([dis_feat, ref_feat, diff, prod], dim=-1))
        elif self.fr_interaction == 'concat_mlp':
            diff = torch.abs(dis_feat - ref_feat)
            return self.fr_fusion(torch.cat([dis_feat, ref_feat, diff], dim=-1))
        elif self.fr_interaction == 'cosine_diff':
            # Cosine similarity per-element (normalized dot) + absolute diff
            cos_sim = torch.nn.functional.cosine_similarity(
                dis_feat, ref_feat, dim=-1, eps=1e-8
            ).unsqueeze(-1).expand_as(dis_feat) * dis_feat
            diff = torch.abs(dis_feat - ref_feat)
            return self.fr_fusion(torch.cat([cos_sim, diff], dim=-1))
        elif self.fr_interaction == 'diff_affine':
            # FiLM-style: ref conditions dis via learned scale & bias
            diff = torch.abs(dis_feat - ref_feat)
            scale = self.fr_scale(ref_feat)  # [0,1] gating
            bias = self.fr_bias(ref_feat)
            modulated = dis_feat * scale + bias
            return self.fr_fusion(torch.cat([modulated, diff], dim=-1))
        elif self.fr_interaction == 'topiq_like':
            # Full TOPIQ-style: diff-gated concat + deep MLP
            diff = torch.abs(dis_feat - ref_feat)
            gate = self.fr_gate(diff)  # [*, 1], sigmoid → scalar per sample
            concat = torch.cat([dis_feat, ref_feat, diff], dim=-1)
            concat = concat * gate  # broadcast gate across channels
            return self.fr_fusion(concat)
        elif self.fr_interaction == 'concat_mlp_deep':
            # Deeper 3-layer MLP on [dis, ref, |diff|]
            diff = torch.abs(dis_feat - ref_feat)
            return self.fr_fusion(torch.cat([dis_feat, ref_feat, diff], dim=-1))
        elif self.fr_interaction == 'diff_affine_res':
            # FiLM + residual: modulated + skip from original dis
            diff = torch.abs(dis_feat - ref_feat)
            scale = self.fr_scale(ref_feat)
            bias = self.fr_bias(ref_feat)
            modulated = dis_feat * scale + bias
            fused = self.fr_fusion(torch.cat([modulated, diff], dim=-1))
            return fused + dis_feat  # residual connection
        elif self.fr_interaction == 'topiq_deep':
            # TOPIQ with wider gate (D/2) + deeper 3-layer MLP
            diff = torch.abs(dis_feat - ref_feat)
            gate = self.fr_gate(diff)  # [*, 1]
            concat = torch.cat([dis_feat, ref_feat, diff], dim=-1)
            concat = concat * gate
            return self.fr_fusion(concat)
        elif self.fr_interaction == 'ensemble_vote':
            # Average of concat_mlp path + diff_affine path
            diff = torch.abs(dis_feat - ref_feat)
            # Path A: concat_mlp
            out_a = self.fr_fusion_a(torch.cat([dis_feat, ref_feat, diff], dim=-1))
            # Path B: diff_affine
            scale_b = self.fr_scale_b(ref_feat)
            bias_b = self.fr_bias_b(ref_feat)
            modulated_b = dis_feat * scale_b + bias_b
            out_b = self.fr_fusion_b(torch.cat([modulated_b, diff], dim=-1))
            return (out_a + out_b) * 0.5
        else:
            return dis_feat

    def _apply_resolution_modulation(
        self,
        feat: torch.Tensor,
        resolution_scale_bias: dict = None,
    ) -> torch.Tensor:
        """
        Apply FiLM-style resolution modulation to backbone features.

        Args:
            feat: [N, bb_dim] backbone output features
            resolution_scale_bias: dict with 'scale' [B, bb_dim] and 'bias' [B, bb_dim]
                from ResolutionConditioner. If None, no-op.

        Returns:
            Modulated features [N, bb_dim]
        """
        if resolution_scale_bias is None:
            return feat

        scale = resolution_scale_bias['scale']  # [B, D]
        bias = resolution_scale_bias['bias']     # [B, D]

        N = feat.shape[0]
        B = scale.shape[0]

        if N != B and N % B == 0:
            # feat is [B*T, D] or [B*P, D] or [B*P*T, D] — expand scale/bias
            repeat = N // B
            scale = scale.unsqueeze(1).expand(B, repeat, -1).reshape(N, -1)
            bias = bias.unsqueeze(1).expand(B, repeat, -1).reshape(N, -1)

        return feat * (1.0 + scale) + bias

    def _needs_res_hw(self) -> bool:
        """Check if backbone needs res_hw for Scheme F/H."""
        return self.patch_embed_cond or self.resolution_token

    def _encode_frames(self, frames: torch.Tensor, resolution_scale: float = 1.0, res_hw: tuple = None) -> torch.Tensor:
        """Encode [N, 3, H, W] → [N, bb_dim] with optional renormalization and whitening."""
        frames = self.renorm(frames)
        if self.rope_resolution_scale and abs(resolution_scale - 1.0) > 1e-6:
            feat = self.backbone(frames, resolution_scale=resolution_scale, res_hw=res_hw)
        elif self._needs_res_hw() and res_hw is not None:
            feat = self.backbone(frames, res_hw=res_hw)
        else:
            feat = self.backbone(frames)
        # Apply feature whitening if enabled
        if self.feature_whitening:
            feat = self.whiten_proj(self.whiten_norm(feat))
        return feat

    def _encode_frames_multilayer(self, frames: torch.Tensor, resolution_scale: float = 1.0, res_hw: tuple = None) -> list[torch.Tensor]:
        """Encode [N, 3, H, W] -> list[[N, bb_dim]] from selected PE layers."""
        frames = self.renorm(frames)
        feats = self.backbone.forward_intermediate_layers(
            frames,
            layer_indices=self.topiq_layer_ids,
            norm=True,
            resolution_scale=resolution_scale,
            res_hw=res_hw,
        )
        # Apply feature whitening to each layer if enabled
        if self.feature_whitening:
            feats = [self.whiten_proj(self.whiten_norm(f)) for f in feats]
        return feats

    def _apply_topiq_multilayer_interaction(
        self,
        dis_feat_layers: list[torch.Tensor],
        ref_feat_layers: list[torch.Tensor],
    ) -> torch.Tensor:
        """Apply FR interaction per layer, then concat and project back to bb_dim."""
        fused_layers = [
            self._apply_fr_interaction(dis_l, ref_l)
            for dis_l, ref_l in zip(dis_feat_layers, ref_feat_layers)
        ]
        return self.topiq_layer_merge(torch.cat(fused_layers, dim=-1))

    def _temporal_aggregate(self, feat: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Temporal aggregation: [B*T, D] → [B, D].

        Mode:
          - tsm_mean: TSM shift + mean
          - attention: cross-attention pooling
          - mean: simple mean
        """
        if self.temporal_pool_mode == 'attention':
            feat_bt = feat.reshape(B, T, -1)  # [B, T, D]
            return self.attention_pool(feat_bt)  # [B, D]
        elif self.temporal_pool_mode == 'tsm_mean':
            feat = self._apply_tsm(feat, T)
            return feat.reshape(B, T, -1).mean(dim=1)
        else:
            return feat.reshape(B, T, -1).mean(dim=1)

    # ---- MSS (Multi-Scale Spatial) helpers ----
    def _encode_patches_to_feat(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Encode GMS patches [P, 3, T, H, W] → [P, D] (per-patch backbone + temporal pool).
        """
        P, C, T, H, W = patches.shape
        frames = patches.permute(0, 2, 1, 3, 4).reshape(P * T, C, H, W)  # [P*T, 3, H, W]
        feat = self._encode_frames(frames)  # [P*T, D]
        return self._temporal_aggregate(feat, P, T)  # [P, D]

    def _fuse_mss(self, global_feat: torch.Tensor, local_feats: torch.Tensor) -> torch.Tensor:
        """
        Fuse global and local features via MSS fusion.

        Args:
            global_feat:  [B, D]   — from resize path
            local_feats:  [B, P, D] — from GMS patches
        Returns:
            fused: [B, D]
        """
        if self.mss_fusion_mode == 'cross_attn':
            # Global as query [B,1,D], patches as key/value [B,P,D]
            q = self.mss_norm_q(global_feat.unsqueeze(1))   # [B, 1, D]
            kv = self.mss_norm_kv(local_feats)               # [B, P, D]
            attn_out = self.mss_cross_attn(q, kv).squeeze(1)  # [B, D]
            # Merge global + attended local
            fused = self.mss_merge(torch.cat([global_feat, attn_out], dim=-1))  # [B, D]
            return fused
        elif self.mss_fusion_mode == 'concat':
            local_mean = local_feats.mean(dim=1)  # [B, D]
            fused = self.mss_merge(torch.cat([global_feat, local_mean], dim=-1))
            return fused
        elif self.mss_fusion_mode == 'add':
            local_mean = local_feats.mean(dim=1)  # [B, D]
            return global_feat + self.mss_local_scale * local_mean
        else:
            return global_feat

    def forward_mss(
        self,
        dis_global: torch.Tensor,
        dis_patches: torch.Tensor,
        ref_global: torch.Tensor = None,
        ref_patches: torch.Tensor = None,
        num_frames: int = 1,
    ) -> torch.Tensor:
        """
        MSS forward: encode global resize + local GMS patches, fuse, then FR interaction.

        Args:
            dis_global:  [B*T, 3, H, W] — resize frames (flattened temporal)
            dis_patches: [B, P, 3, T, H, W] — GMS patches
            ref_global:  same as dis_global (optional, FR mode)
            ref_patches: same as dis_patches (optional, FR mode)
            num_frames:  T for temporal aggregation

        Returns:
            [B, D] fused feature (or [B, out_dim] if projection head)
        """
        B = dis_patches.shape[0]
        P = dis_patches.shape[1]

        # 1. Encode global resize path → [B, D]
        if ref_global is not None and self.use_topiq_multilayer:
            dis_layers = self._encode_frames_multilayer(dis_global)
            ref_layers = self._encode_frames_multilayer(ref_global)
            dis_layers = [self._apply_temporal_interaction(x, num_frames) for x in dis_layers]
            ref_layers = [self._apply_temporal_interaction(x, num_frames) for x in ref_layers]
            global_feat = self._apply_topiq_multilayer_interaction(dis_layers, ref_layers)  # [B, D]
        else:
            dis_feat = self._encode_frames(dis_global)
            dis_feat = self._apply_temporal_interaction(dis_feat, num_frames)
            if ref_global is not None:
                ref_feat = self._encode_frames(ref_global)
                ref_feat = self._apply_temporal_interaction(ref_feat, num_frames)
                dis_feat = self._apply_fr_interaction(dis_feat, ref_feat)
            BT = dis_feat.shape[0]
            T = max(1, num_frames)
            B_g = BT // T
            global_feat = dis_feat.reshape(B_g, T, -1).mean(dim=1)  # [B, D]

        # 2. Encode local GMS patches → [B, P, D]
        # Flatten patches: [B, P, 3, T, H, W] → [B*P, 3, T, H, W]
        _, _, C, T_p, H_p, W_p = dis_patches.shape
        dis_flat = dis_patches.reshape(B * P, C, T_p, H_p, W_p)
        local_feats = self._encode_and_pool_clip(dis_flat)  # [B*P, D]

        # FR interaction per-patch (if ref available)
        if ref_patches is not None:
            ref_flat = ref_patches.reshape(B * P, C, T_p, H_p, W_p)
            ref_local = self._encode_and_pool_clip(ref_flat)  # [B*P, D]
            local_feats = self._apply_fr_interaction(local_feats, ref_local)

        local_feats = local_feats.reshape(B, P, -1)  # [B, P, D]

        # 3. Fuse global + local
        fused = self._fuse_mss(global_feat, local_feats)  # [B, D]

        if self.head_mode == 'repro_mlp':
            return fused

        out = self.head(fused)
        return out

    def forward(
        self,
        dis_frames: torch.Tensor,
        ref_frames: torch.Tensor = None,
        num_frames: int = 1,
        resolution_scale_bias: dict = None,
        resolution_scale: float = 1.0,
        res_hw: tuple = None,
        num_clips: int = None,
    ) -> torch.Tensor:
        """
        Args:
            dis_frames: [B*T, 3, H, W] or (if multipatch_input) [B, P, 3, T, H, W]
            ref_frames: optional, same shape
            num_frames: T for temporal aggregation (used when input is [B*T, 3, H, W])
            resolution_scale_bias: optional dict with 'scale'/'bias' from ResolutionConditioner
            resolution_scale: RoPE position scale factor for Scheme E (1.0=default)
            res_hw: (height, width) for Scheme F/H (patch_embed_cond / resolution_token)

        Returns:
            When head_mode='projection': [B*T, out_dim] (or [B, out_dim] if multipatch)
            When head_mode='repro_mlp': [B*T, bb_dim] features (score via forward_with_score)
        """
        if self.multipatch_input and dis_frames.dim() == 6:
            return self._forward_multipatch(dis_frames, ref_frames,
                                           resolution_scale_bias=resolution_scale_bias,
                                           resolution_scale=resolution_scale,
                                           res_hw=res_hw,
                                           num_clips=num_clips)

        # Standard 4D path: [B*T, 3, H, W]
        if ref_frames is not None and self.use_topiq_multilayer:
            dis_layers = self._encode_frames_multilayer(dis_frames, resolution_scale=resolution_scale, res_hw=res_hw)
            ref_layers = self._encode_frames_multilayer(ref_frames, resolution_scale=resolution_scale, res_hw=res_hw)
            # Apply FiLM to each layer before TSM
            dis_layers = [self._apply_resolution_modulation(x, resolution_scale_bias) for x in dis_layers]
            ref_layers = [self._apply_resolution_modulation(x, resolution_scale_bias) for x in ref_layers]
            dis_layers = [self._apply_temporal_interaction(x, num_frames) for x in dis_layers]
            ref_layers = [self._apply_temporal_interaction(x, num_frames) for x in ref_layers]
            dis_feat = self._apply_topiq_multilayer_interaction(dis_layers, ref_layers)
        else:
            dis_feat = self._encode_frames(dis_frames, resolution_scale=resolution_scale, res_hw=res_hw)
            dis_feat = self._apply_resolution_modulation(dis_feat, resolution_scale_bias)
            dis_feat = self._apply_temporal_interaction(dis_feat, num_frames)

            # FR interaction at backbone feature level (before head)
            if ref_frames is not None:
                ref_feat = self._encode_frames(ref_frames, resolution_scale=resolution_scale, res_hw=res_hw)
                ref_feat = self._apply_resolution_modulation(ref_feat, resolution_scale_bias)
                ref_feat = self._apply_temporal_interaction(ref_feat, num_frames)
                dis_feat = self._apply_fr_interaction(dis_feat, ref_feat)

        if self.head_mode == 'repro_mlp':
            # Return FR-fused backbone features for fusion; score is computed separately
            return dis_feat

        out = self.head(dis_feat)
        return out

    def _forward_multipatch(
        self,
        dis_patches: torch.Tensor,
        ref_patches: torch.Tensor = None,
        resolution_scale_bias: dict = None,
        resolution_scale: float = 1.0,
        res_hw: tuple = None,
        num_clips: int = None,
    ) -> torch.Tensor:
        """
        Multi-patch forward: [B, P, 3, T, H, W] → [B, D] features + optional score.
        This path is used when semantic sampler produces multi-patch (gmsavg stack mode).
        Supports FR interaction when ref_patches is provided.
        """
        B, P, C, T, H, W = dis_patches.shape
        # Flatten to [B*P, C, T, H, W]
        x_flat = dis_patches.reshape(B * P, C, T, H, W)

        # Optionally chunk for memory
        if self.patch_forward_chunk_size > 0 and x_flat.shape[0] > self.patch_forward_chunk_size:
            feat_list = []
            for s in range(0, x_flat.shape[0], self.patch_forward_chunk_size):
                chunk = x_flat[s:s + self.patch_forward_chunk_size]
                feat_list.append(self._encode_and_pool_clip(chunk, resolution_scale=resolution_scale, res_hw=res_hw))
            feat = torch.cat(feat_list, dim=0)  # [B*P, D]
        else:
            feat = self._encode_and_pool_clip(x_flat, resolution_scale=resolution_scale, res_hw=res_hw)  # [B*P, D]

        # Apply FiLM resolution modulation after backbone encoding
        feat = self._apply_resolution_modulation(feat, resolution_scale_bias)

        # FR interaction at backbone feature level (per-patch)
        if ref_patches is not None:
            r_flat = ref_patches.reshape(B * P, C, T, H, W)
            if self.use_topiq_multilayer:
                if self.patch_forward_chunk_size > 0 and x_flat.shape[0] > self.patch_forward_chunk_size:
                    dis_layer_buf = [[] for _ in self.topiq_layer_ids]
                    ref_layer_buf = [[] for _ in self.topiq_layer_ids]
                    for s in range(0, x_flat.shape[0], self.patch_forward_chunk_size):
                        d_chunk = x_flat[s:s + self.patch_forward_chunk_size]
                        r_chunk = r_flat[s:s + self.patch_forward_chunk_size]
                        d_layers = self._encode_and_pool_clip_multilayer(d_chunk, resolution_scale=resolution_scale, res_hw=res_hw)
                        r_layers = self._encode_and_pool_clip_multilayer(r_chunk, resolution_scale=resolution_scale, res_hw=res_hw)
                        for i, (dl, rl) in enumerate(zip(d_layers, r_layers)):
                            dis_layer_buf[i].append(dl)
                            ref_layer_buf[i].append(rl)
                    dis_layers = [torch.cat(v, dim=0) for v in dis_layer_buf]
                    ref_layers = [torch.cat(v, dim=0) for v in ref_layer_buf]
                else:
                    dis_layers = self._encode_and_pool_clip_multilayer(x_flat, resolution_scale=resolution_scale, res_hw=res_hw)
                    ref_layers = self._encode_and_pool_clip_multilayer(r_flat, resolution_scale=resolution_scale, res_hw=res_hw)
                # Apply FiLM to each layer
                dis_layers = [self._apply_resolution_modulation(x, resolution_scale_bias) for x in dis_layers]
                ref_layers = [self._apply_resolution_modulation(x, resolution_scale_bias) for x in ref_layers]
                feat = self._apply_topiq_multilayer_interaction(dis_layers, ref_layers)
            else:
                if self.patch_forward_chunk_size > 0 and r_flat.shape[0] > self.patch_forward_chunk_size:
                    ref_feat_list = []
                    for s in range(0, r_flat.shape[0], self.patch_forward_chunk_size):
                        chunk = r_flat[s:s + self.patch_forward_chunk_size]
                        ref_feat_list.append(self._encode_and_pool_clip(chunk, resolution_scale=resolution_scale, res_hw=res_hw))
                    ref_feat = torch.cat(ref_feat_list, dim=0)
                else:
                    ref_feat = self._encode_and_pool_clip(r_flat, resolution_scale=resolution_scale, res_hw=res_hw)  # [B*P, D]
                # Apply FiLM to ref features too
                ref_feat = self._apply_resolution_modulation(ref_feat, resolution_scale_bias)
                feat = self._apply_fr_interaction(feat, ref_feat)

        # Reshape to [B, P, D] and reduce patches
        feat = feat.reshape(B, P, -1)
        feat = self._reduce_patches(feat)  # [B, D]

        # Cross-clip temporal interaction (multi_clip_4x8 mode)
        if self.cross_clip_temporal_module is not None and num_clips is not None and num_clips > 1:
            B_real = feat.shape[0] // num_clips
            feat = feat.view(B_real, num_clips, -1)  # [B_real, K, D]
            feat = self.cross_clip_temporal_module(feat)  # [B_real, D]

        return feat

    def _encode_and_pool_clip(self, clip: torch.Tensor, resolution_scale: float = 1.0, res_hw: tuple = None) -> torch.Tensor:
        """
        Encode a 5D clip [N, 3, T, H, W] → [N, D].
        Applies per-frame backbone + temporal pooling.
        """
        N, C, T, H, W = clip.shape
        frames = clip.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        frame_feat = self._encode_frames(frames, resolution_scale=resolution_scale, res_hw=res_hw)  # [N*T, D]
        return self._temporal_aggregate(frame_feat, N, T)  # [N, D]

    def _encode_and_pool_clip_multilayer(self, clip: torch.Tensor, resolution_scale: float = 1.0, res_hw: tuple = None) -> list[torch.Tensor]:
        """Encode a 5D clip [N, 3, T, H, W] -> list[[N, D]] for selected PE layers."""
        N, C, T, H, W = clip.shape
        frames = clip.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        frame_feats = self._encode_frames_multilayer(frames, resolution_scale=resolution_scale, res_hw=res_hw)
        return [self._temporal_aggregate(f, N, T) for f in frame_feats]

    def forward_with_score(
        self,
        dis_frames: torch.Tensor,
        ref_frames: torch.Tensor = None,
        num_frames: int = 1,
        resolution_scale_bias: dict = None,
        resolution_scale: float = 1.0,
        res_hw: tuple = None,
        num_clips: int = None,
    ) -> dict:
        """
        Full forward with score computation (repro_mlp mode).
        Returns dict with 'feat' and 'score'.
        Supports FR interaction at backbone feature level.

        For multipatch input [B, P, 3, T, H, W], returns per-patch and reduced scores.
        """
        if self.head_mode != 'repro_mlp':
            feat = self.forward(dis_frames, ref_frames, num_frames,
                               resolution_scale_bias=resolution_scale_bias,
                               resolution_scale=resolution_scale,
                               res_hw=res_hw,
                               num_clips=num_clips)
            return {'feat': feat, 'score': None}

        if self.multipatch_input and dis_frames.dim() == 6:
            B, P, C, T, H, W = dis_frames.shape
            x_flat = dis_frames.reshape(B * P, C, T, H, W)

            if self.patch_forward_chunk_size > 0 and x_flat.shape[0] > self.patch_forward_chunk_size:
                feat_list = []
                for s in range(0, x_flat.shape[0], self.patch_forward_chunk_size):
                    chunk = x_flat[s:s + self.patch_forward_chunk_size]
                    feat_list.append(self._encode_and_pool_clip(chunk, resolution_scale=resolution_scale, res_hw=res_hw))
                feat = torch.cat(feat_list, dim=0)
            else:
                feat = self._encode_and_pool_clip(x_flat, resolution_scale=resolution_scale, res_hw=res_hw)

            # Apply FiLM resolution modulation
            feat = self._apply_resolution_modulation(feat, resolution_scale_bias)

            # FR interaction at backbone feature level (per-patch)
            if ref_frames is not None and ref_frames.dim() == 6:
                r_flat = ref_frames.reshape(B * P, C, T, H, W)
                if self.patch_forward_chunk_size > 0 and r_flat.shape[0] > self.patch_forward_chunk_size:
                    ref_feat_list = []
                    for s in range(0, r_flat.shape[0], self.patch_forward_chunk_size):
                        chunk = r_flat[s:s + self.patch_forward_chunk_size]
                        ref_feat_list.append(self._encode_and_pool_clip(chunk, resolution_scale=resolution_scale, res_hw=res_hw))
                    ref_feat = torch.cat(ref_feat_list, dim=0)
                else:
                    ref_feat = self._encode_and_pool_clip(r_flat, resolution_scale=resolution_scale, res_hw=res_hw)
                ref_feat = self._apply_resolution_modulation(ref_feat, resolution_scale_bias)
                feat = self._apply_fr_interaction(feat, ref_feat)

            score_patch = self.head(feat).view(B, P)
            score = self._reduce_patch_scores(score_patch)

            feat_patch = feat.view(B, P, -1)
            feat_video = self._reduce_patches(feat_patch)

            # Cross-clip temporal interaction (multi_clip_4x8 mode)
            if self.cross_clip_temporal_module is not None and num_clips is not None and num_clips > 1:
                B_real = feat_video.shape[0] // num_clips
                feat_video = feat_video.view(B_real, num_clips, -1)  # [B_real, K, D]
                feat_video = self.cross_clip_temporal_module(feat_video)  # [B_real, D]
                # Recompute score using the fused feature
                score = self.head(feat_video).squeeze(-1)  # [B_real]

            return {
                'feat': feat_video,
                'score': score,
                'score_per_patch': score_patch,
            }
        else:
            # Standard 4D input
            dis_feat = self._encode_frames(dis_frames, resolution_scale=resolution_scale, res_hw=res_hw)
            dis_feat = self._apply_resolution_modulation(dis_feat, resolution_scale_bias)
            BT = dis_feat.shape[0]
            B = BT // max(1, num_frames)
            T = max(1, num_frames)
            pooled = self._temporal_aggregate(dis_feat, B, T)  # [B, D]

            # FR interaction at backbone feature level
            if ref_frames is not None:
                ref_feat = self._encode_frames(ref_frames, resolution_scale=resolution_scale, res_hw=res_hw)
                ref_feat = self._apply_resolution_modulation(ref_feat, resolution_scale_bias)
                ref_pooled = self._temporal_aggregate(ref_feat, B, T)  # [B, D]
                pooled = self._apply_fr_interaction(pooled, ref_pooled)

            score = self.head(pooled).squeeze(-1)  # [B]
            return {
                'feat': pooled,
                'score': score,
            }
