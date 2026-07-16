"""
HMFVQA: Hybrid Multi-Feature Video Quality Assessment.

Top-level model orchestrating:
  - ColorSpaceAdapter (YUV->RGB+normalize, non-learnable)
  - Semantic branch (PE encoder + topiq_deep FR interaction)
  - FusionHead + Aggregator
  - RAPE/ScaleToken/ResolutionConditioner for multi-resolution awareness


"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import logging

from .adapters.colorspace_adapter import ColorSpaceAdapter
from .branches.semantic_branch import SemanticBranch
from .fusion import Aggregator, FusionHead, RAPE, ScaleToken, ResolutionConditioner

logger = logging.getLogger('hmf_vqa.model')


def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


class HMFVQA(nn.Module):
    """
    Hybrid Multi-Feature VQA Model.

    Input contract:
        dis_yuv: [B, 3, T, H, W]  float32 [0,1] YUV
        ref_yuv: [B, 3, T, H, W]  float32 [0,1] YUV (None for NR)
        spatial_info: dict with GMS patches / FuPiC tiles (pre-sampled)
        meta: dict with 'height', 'width' for RAPE/ScaleToken
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.task = getattr(cfg, 'task', 'FR')
        self.is_fr = self.task.upper() == 'FR'

        # ---- Non-learnable adapter ----
        self.colorspace_adapter = ColorSpaceAdapter(
            mode=getattr(cfg, 'colorspace', 'bt709_imagenet'),
        )

        # ---- Branches ----
        enabled = getattr(cfg, 'branches', ['vif', 'detail', 'semantic'])
        if isinstance(enabled, str):
            enabled = [b.strip() for b in enabled.split(',')]
        self.enabled_branches = [b.lower() for b in enabled]

        self.vif_branch = None
        self.detail_branch = None
        self.semantic_branch = None

        vif_dim = 0
        detail_dim = 0
        semantic_dim = 0

        # VIF/Detail branches were removed in this release; the attributes
        # are pinned to their no-op defaults so any downstream conditional
        # (``self.vif_branch is not None``, ``self.vif_use_in_fusion``,
        # ``self.vif_score_fusion != 'none'``) short-circuits to skip.
        self.vif_align_with_other_branches = True
        self.vif_mode_default = 'aligned'
        self.vif_use_in_fusion = False
        self.vif_score_fusion = 'none'
        self.vif_score_fusion_alpha = 1.0

        if 'semantic' in self.enabled_branches:
            sem_backbone = getattr(cfg, 'semantic_backbone', 'convnextv2_tiny')
            temporal = getattr(cfg, 'temporal', 'tsm')
            pretrained = getattr(cfg, 'pretrained', True)
            pe_weights = getattr(cfg, 'pe_weights', None)
            sem_fr_interaction = getattr(cfg, 'semantic_fr_interaction', 'diff_prod')
            # Semantic branch nested config (ablation switches for repro alignment)
            sem_cfg_raw = _cfg_get(cfg, 'semantic_branch', {})
            if not isinstance(sem_cfg_raw, dict):
                sem_cfg_raw = dict(sem_cfg_raw) if sem_cfg_raw else {}
            semantic_cfg = dict(sem_cfg_raw)  # copy
            # Pass colorspace to SemanticBranch for correct ReNormalize src mean/std
            semantic_cfg.setdefault('_colorspace', getattr(cfg, 'colorspace', 'bt709_imagenet'))
            # Pass target size for SwinV2 backbone img_size initialization
            semantic_cfg.setdefault('_target_size', getattr(cfg, 'semantic_target_size', 224))
            # Pass Scheme E rope_resolution_scale (top-level config → semantic_branch nested)
            semantic_cfg.setdefault('rope_resolution_scale', getattr(cfg, 'rope_resolution_scale', False))
            # Pass Scheme F patch_embed_cond (top-level config → semantic_branch nested)
            semantic_cfg.setdefault('patch_embed_cond', getattr(cfg, 'patch_embed_cond', False))
            # Pass Scheme H resolution_token (top-level config → semantic_branch nested)
            semantic_cfg.setdefault('resolution_token', getattr(cfg, 'resolution_token', False))
            # Pass StochasticDepth rate (top-level config → semantic_branch nested)
            semantic_cfg.setdefault('stochastic_depth_rate', getattr(cfg, 'stochastic_depth_rate', 0.0))
            # Pass Feature Whitening (top-level config → semantic_branch nested)
            semantic_cfg.setdefault('feature_whitening', getattr(cfg, 'feature_whitening', False))
            self.semantic_head_mode = str(semantic_cfg.get('head_mode', 'projection')).lower()
            self.semantic_multipatch = bool(semantic_cfg.get('multipatch_input', False))
            # Cross-clip temporal: pass config to semantic_branch
            cross_clip_temporal = str(getattr(cfg, 'cross_clip_temporal', 'none')).lower().strip()
            semantic_cfg.setdefault('cross_clip_temporal', cross_clip_temporal)
            self.cross_clip_temporal = cross_clip_temporal
            self.semantic_branch = SemanticBranch(
                backbone_name=sem_backbone,
                temporal=temporal,
                pretrained=pretrained,
                pe_weights=pe_weights,
                fr_interaction=sem_fr_interaction,
                semantic_cfg=semantic_cfg,
            )
            semantic_dim = self.semantic_branch.out_dim

        # ---- Multi-resolution ----
        rape_str = getattr(cfg, 'rape', 'on')
        scale_str = getattr(cfg, 'scale_token', 'on')
        self.use_rape = (rape_str == 'on') if isinstance(rape_str, str) else bool(rape_str)
        self.use_scale_token = (scale_str == 'on') if isinstance(scale_str, str) else bool(scale_str)
        rape_dim = 0
        scale_dim = 0

        if self.use_rape:
            self.rape = RAPE(embed_dim=32)
            rape_dim = 32
        if self.use_scale_token:
            self.scale_token = ScaleToken(embed_dim=16)
            scale_dim = 16

        # ---- FiLM Resolution Conditioner ----
        # Directly modulates backbone features (bb_dim) instead of small side branch.
        # Enabled by config: res_conditioner='on'
        rc_str = getattr(cfg, 'res_conditioner', 'off')
        self.use_res_conditioner = (rc_str == 'on') if isinstance(rc_str, str) else bool(rc_str)
        if self.use_res_conditioner:
            rc_feat_dim = semantic_dim if semantic_dim > 0 else 1024
            rc_hidden = int(getattr(cfg, 'res_conditioner_hidden', 256))
            self.res_conditioner = ResolutionConditioner(
                feat_dim=rc_feat_dim,
                hidden_dim=rc_hidden,
            )
            logger.info(
                "ResolutionConditioner (FiLM) enabled: feat_dim=%d, hidden=%d",
                rc_feat_dim, rc_hidden,
            )

        # ---- Fusion ----
        fusion_type = getattr(cfg, 'fusion_type', 'late_concat_mlp')
        fusion_hidden = getattr(cfg, 'fusion_hidden', 256)
        branch_dims = {}
        if vif_dim > 0 and self.vif_use_in_fusion:
            branch_dims['vif'] = vif_dim
        if detail_dim > 0:
            branch_dims['detail'] = detail_dim
        if semantic_dim > 0:
            branch_dims['semantic'] = semantic_dim
        if rape_dim + scale_dim > 0:
            branch_dims['extra'] = rape_dim + scale_dim
        self.fusion_head = FusionHead(
            fusion_type=fusion_type,
            branch_dims=branch_dims,
            hidden_dim=fusion_hidden,
        )

        # ---- Aggregator ----
        agg_mode = getattr(cfg, 'aggregator_mode', 'mean')
        self.aggregator = Aggregator(mode=agg_mode, feature_dim=fusion_hidden)

    def forward(
        self,
        dis_yuv: Optional[torch.Tensor],
        ref_yuv: Optional[torch.Tensor] = None,
        spatial_info: Optional[Dict] = None,
        meta: Optional[Dict] = None,
        use_only_vif_branch: bool = False,
        vif_mode_override: Optional[str] = None,
        return_vif_per_frame: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Determine batch size / device from available inputs.
        if dis_yuv is not None:
            B = dis_yuv.shape[0]
            device = dis_yuv.device
        elif ref_yuv is not None:
            B = ref_yuv.shape[0]
            device = ref_yuv.device
        elif spatial_info:
            first_t = next(iter(spatial_info.values()))
            B = first_t.shape[0]
            device = first_t.device
        elif meta and 'height' in meta:
            B = meta['height'].shape[0]
            device = meta['height'].device
        else:
            raise ValueError("Cannot determine batch size: no inputs provided")

        outputs: Dict[str, torch.Tensor] = {}

        # VIF / Detail branches removed in this release. The placeholders
        # below keep the downstream fusion code path identical to the
        # original implementation (which also skipped both branches under
        # the SD3 / ZG2 / LSAM configs because ``branches=semantic``).
        vif_feat = None
        detail_feat = None
        # NOTE: ``use_only_vif_branch`` / ``vif_mode_override`` /
        # ``return_vif_per_frame`` are kept in the signature for API
        # compatibility with the trainer/evaluator's kwargs; they are no-ops
        # without a VIF branch.

        # ---- Semantic branch ----
        semantic_feat = None
        # Compute FiLM resolution conditioning if enabled
        resolution_scale_bias = None
        if self.use_res_conditioner and meta is not None:
            rc_h = meta.get('height', torch.tensor([1080] * B, device=device))
            rc_w = meta.get('width', torch.tensor([1920] * B, device=device))
            if not isinstance(rc_h, torch.Tensor):
                rc_h = torch.tensor(rc_h, dtype=torch.float32, device=device)
                rc_w = torch.tensor(rc_w, dtype=torch.float32, device=device)
            rc_out = self.res_conditioner(rc_h.float(), rc_w.float())
            resolution_scale_bias = {'scale': rc_out['scale'], 'bias': rc_out['bias']}

        # ---- Scheme E: Resolution-Scaled RoPE ----
        # Compute per-batch resolution scale factor for RoPE2D position scaling.
        # Uses max(H, W) / 1920 so 1080p → 1.0, 4K → ~2.0.
        # When the semantic branch has rope_resolution_scale enabled, this is
        # passed through to the backbone to scale RoPE2D grid positions.
        resolution_scale = 1.0
        if meta is not None and hasattr(self, 'semantic_branch') and self.semantic_branch is not None:
            if getattr(self.semantic_branch, 'rope_resolution_scale', False):
                rc_h = meta.get('height', torch.tensor([1080] * B, device=device))
                rc_w = meta.get('width', torch.tensor([1920] * B, device=device))
                if isinstance(rc_h, torch.Tensor):
                    max_dim = float(torch.max(rc_h.float().max(), rc_w.float().max()).item())
                else:
                    max_dim = float(max(rc_h, rc_w))
                resolution_scale = max_dim / 1920.0

        # ---- Scheme F/H: resolution tuple for PatchEmbedCond / ResolutionToken ----
        # Compute (height, width) tuple when semantic branch needs it.
        res_hw = None
        if meta is not None and hasattr(self, 'semantic_branch') and self.semantic_branch is not None:
            if getattr(self.semantic_branch, '_needs_res_hw', lambda: False)():
                rc_h = meta.get('height', torch.tensor([1080] * B, device=device))
                rc_w = meta.get('width', torch.tensor([1920] * B, device=device))
                res_hw = (rc_h, rc_w)

        if self.semantic_branch is not None and spatial_info is not None:
            resize_dis = spatial_info.get('resize_dis')  # [B, 3, T, Rh, Rw] or [B, P, 3, T, Rh, Rw]
            resize_ref = spatial_info.get('resize_ref') if self.is_fr else None
            mss_gms_dis = spatial_info.get('mss_gms_dis')  # [B, P, 3, T, H, W] (MSS local patches)
            mss_gms_ref = spatial_info.get('mss_gms_ref') if self.is_fr else None

            if mss_gms_dis is not None and resize_dis is not None and getattr(self.semantic_branch, 'use_mss', False):
                # MSS path: global resize + local GMS patches
                rd_rgb = self._adapt_temporal(resize_dis)  # [B, 3, T, Rh, Rw]
                rr_rgb = self._adapt_temporal(resize_ref) if resize_ref is not None else None
                B_s, C_s, T_s, H_s, W_s = rd_rgb.shape

                # Convert global resize to [B*T, 3, H, W]
                d_global_flat = rd_rgb.permute(0, 2, 1, 3, 4).reshape(B_s * T_s, C_s, H_s, W_s)
                r_global_flat = None
                if rr_rgb is not None:
                    r_global_flat = rr_rgb.permute(0, 2, 1, 3, 4).reshape(B_s * T_s, C_s, H_s, W_s)

                # Convert local patches [B, P, 3, T, H, W] to RGB
                B_m, P_m, C_m, T_m, H_m, W_m = mss_gms_dis.shape
                md_flat = mss_gms_dis.reshape(B_m * P_m, C_m, T_m, H_m, W_m)
                md_rgb_flat = self._adapt_temporal(md_flat)
                md_rgb = md_rgb_flat.reshape(B_m, P_m, C_m, T_m, H_m, W_m)
                mr_rgb = None
                if mss_gms_ref is not None:
                    mr_flat = mss_gms_ref.reshape(B_m * P_m, C_m, T_m, H_m, W_m)
                    mr_rgb_flat = self._adapt_temporal(mr_flat)
                    mr_rgb = mr_rgb_flat.reshape(B_m, P_m, C_m, T_m, H_m, W_m)

                semantic_feat = self.semantic_branch.forward_mss(
                    dis_global=d_global_flat,
                    dis_patches=md_rgb,
                    ref_global=r_global_flat,
                    ref_patches=mr_rgb,
                    num_frames=T_s,
                )
                outputs['semantic_feat'] = semantic_feat

            elif resize_dis is not None:
                # Detect multi_clip_4x8 mode: 7-dim input [B, K, P, 3, T, H, W]
                is_multi_clip = (resize_dis.dim() == 7)
                num_clips_for_temporal = None
                if is_multi_clip:
                    B_mc, K_mc, P_mc, C_mc, T_mc, H_mc, W_mc = resize_dis.shape
                    num_clips_for_temporal = K_mc
                    # Flatten K dimension: [B*K, P, 3, T, H, W]
                    resize_dis = resize_dis.reshape(B_mc * K_mc, P_mc, C_mc, T_mc, H_mc, W_mc)
                    if resize_ref is not None and resize_ref.dim() == 7:
                        resize_ref = resize_ref.reshape(B_mc * K_mc, P_mc, C_mc, T_mc, H_mc, W_mc)

                if self.semantic_multipatch and resize_dis.dim() == 6:
                    # Multi-patch path: [B, P, 3, T, H, W]
                    # ColorSpace conversion per-patch-per-frame
                    B_s, P_s, C_s, T_s, H_s, W_s = resize_dis.shape
                    rd_flat = resize_dis.reshape(B_s * P_s, C_s, T_s, H_s, W_s)
                    rd_rgb_flat = self._adapt_temporal(rd_flat)  # [B*P, 3, T, H, W]
                    rd_rgb = rd_rgb_flat.reshape(B_s, P_s, C_s, T_s, H_s, W_s)

                    # FR: convert ref patches too
                    rr_rgb = None
                    if resize_ref is not None and resize_ref.dim() == 6:
                        rr_flat = resize_ref.reshape(B_s * P_s, C_s, T_s, H_s, W_s)
                        rr_rgb_flat = self._adapt_temporal(rr_flat)
                        rr_rgb = rr_rgb_flat.reshape(B_s, P_s, C_s, T_s, H_s, W_s)

                    if self.semantic_head_mode == 'repro_mlp':
                        sem_out = self.semantic_branch.forward_with_score(
                            rd_rgb, rr_rgb,
                            resolution_scale_bias=resolution_scale_bias,
                            resolution_scale=resolution_scale,
                            res_hw=res_hw,
                            num_clips=num_clips_for_temporal,
                        )
                        semantic_feat = sem_out['feat']  # [B, D]
                        if sem_out.get('score') is not None:
                            outputs['semantic_score'] = sem_out['score']
                        if sem_out.get('score_per_patch') is not None:
                            outputs['semantic_score_per_patch'] = sem_out['score_per_patch']
                    else:
                        semantic_feat = self.semantic_branch(
                            rd_rgb, rr_rgb,
                            resolution_scale_bias=resolution_scale_bias,
                            resolution_scale=resolution_scale,
                            res_hw=res_hw,
                            num_clips=num_clips_for_temporal,
                        )  # [B, D]
                    outputs['semantic_feat'] = semantic_feat
                elif self.semantic_head_mode == 'repro_mlp' and resize_dis.dim() == 5:
                    # 5D input with repro_mlp head
                    rd_rgb = self._adapt_temporal(resize_dis)
                    B_s, C_s, T_s, H_s, W_s = rd_rgb.shape

                    # FR: convert ref too
                    rr_flat = None
                    if resize_ref is not None:
                        rr_rgb = self._adapt_temporal(resize_ref)
                        rr_flat = rr_rgb.permute(0, 2, 1, 3, 4).reshape(B_s * T_s, C_s, H_s, W_s)

                    sem_out = self.semantic_branch.forward_with_score(
                        rd_rgb.permute(0, 2, 1, 3, 4).reshape(B_s * T_s, C_s, H_s, W_s),
                        ref_frames=rr_flat,
                        num_frames=T_s,
                        resolution_scale_bias=resolution_scale_bias,
                        resolution_scale=resolution_scale,
                        res_hw=res_hw,
                    )
                    semantic_feat = sem_out['feat']  # [B, D]
                    if sem_out.get('score') is not None:
                        outputs['semantic_score'] = sem_out['score']
                    outputs['semantic_feat'] = semantic_feat
                else:
                    # Standard 5D path: [B, 3, T, Rh, Rw]
                    rd_rgb = self._adapt_temporal(resize_dis)
                    rr_rgb = self._adapt_temporal(resize_ref) if resize_ref is not None else None
                    semantic_feat = self._run_semantic_temporal(rd_rgb, rr_rgb, res_hw=res_hw)  # [B, D]
                    outputs['semantic_feat'] = semantic_feat

        # ---- Resolution features ----
        extra_feats = []
        if self.use_rape and meta is not None:
            h = meta.get('height', torch.tensor([1080] * B, device=device))
            w = meta.get('width', torch.tensor([1920] * B, device=device))
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32, device=device)
                w = torch.tensor(w, dtype=torch.float32, device=device)
            rape_feat = self.rape(h.float(), w.float())
            extra_feats.append(rape_feat)

        if self.use_scale_token and meta is not None:
            h = meta.get('height', torch.tensor([1080] * B, device=device))
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h, dtype=torch.float32, device=device)
            scale_feat = self.scale_token(h.float())
            extra_feats.append(scale_feat)

        extra = torch.cat(extra_feats, dim=1) if extra_feats else None

        # ---- Fusion ----
        # vif_feat / detail_feat are always None in this release (branches
        # were removed) — kept in the dict-build to make the diff vs the
        # original semantics obvious.
        branch_outputs = {}
        if vif_feat is not None and self.vif_use_in_fusion:
            branch_outputs['vif'] = vif_feat
        if detail_feat is not None:
            branch_outputs['detail'] = detail_feat
        if semantic_feat is not None:
            branch_outputs['semantic'] = semantic_feat
        if extra is not None:
            branch_outputs['extra'] = extra

        if branch_outputs:
            # When the semantic branch is in repro_mlp mode and is the only
            # scoring branch, use its direct score instead of the fusion head.
            # If extra features (RAPE/ScaleToken) are present they must go
            # through FusionHead to actually influence the score.
            use_semantic_direct_score = (
                hasattr(self, 'semantic_head_mode') and
                self.semantic_head_mode == 'repro_mlp' and
                'semantic_score' in outputs and
                'extra' not in branch_outputs and
                len(branch_outputs) <= 1
            )
            if use_semantic_direct_score:
                score = outputs['semantic_score']
            else:
                score = self.fusion_head(branch_outputs)  # [B]
        else:
            score = torch.zeros(B, device=device)

        outputs['score'] = score
        return outputs

    def _adapt_temporal(self, yuv: torch.Tensor) -> torch.Tensor:
        """Convert YUV [B, 3, T, H, W] -> RGB normalized [B, 3, T, H, W]."""
        B, C, T, H, W = yuv.shape
        yuv_flat = yuv.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        rgb_flat = self.colorspace_adapter(yuv_flat)
        rgb = rgb_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return rgb

    def _run_semantic_temporal(
        self, dis_rgb: torch.Tensor, ref_rgb: Optional[torch.Tensor],
        res_hw: tuple = None,
    ) -> torch.Tensor:
        """Run semantic branch over temporal dimension in batched mode."""
        B, C, T, H, W = dis_rgb.shape
        d_flat = dis_rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        r_flat = None
        if ref_rgb is not None:
            r_flat = ref_rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feat_flat = self.semantic_branch(d_flat, r_flat, num_frames=T, res_hw=res_hw)

        # Temporal aggregation: use branch's temporal_aggregate if attention mode
        if (hasattr(self, 'semantic_head_mode') and
            hasattr(self.semantic_branch, 'temporal_pool_mode') and
            self.semantic_branch.temporal_pool_mode == 'attention'):
            feat = self.semantic_branch._temporal_aggregate(feat_flat, B, T)
        else:
            feat = feat_flat.reshape(B, T, -1).mean(dim=1)
        return feat

    def aggregate_clip_scores(self, clip_scores: torch.Tensor) -> torch.Tensor:
        """
        Aggregate [B, K] clip scores.

        Note:
          clip-level aggregation does not receive supervision in training;
          non-mean modes here can introduce unstable/untrained behavior.
          Use deterministic mean fallback for non-mean modes.
        """
        mode = str(getattr(self.aggregator, 'mode', 'mean')).lower()
        if mode == 'mean':
            return self.aggregator(clip_scores)
        if not getattr(self, '_clip_agg_warned', False):
            logger.warning(
                "Clip aggregation mode '%s' is untrained in current pipeline; "
                "falling back to mean for deterministic evaluation.",
                mode,
            )
            self._clip_agg_warned = True
        return clip_scores.mean(dim=1)
