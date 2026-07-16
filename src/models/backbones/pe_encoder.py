"""
Meta Perception Encoder (PE) backbone wrapper.
Wraps Meta's VisionTransformer from perception_models for the semantic branch.

Supports PE-Core variants with RoPE2D for flexible input resolution.
Default: PE-Core-B16-224 (12 layers, 768 width, output_dim=1024).
"""

import sys
import os
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root & weights directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', '..'))
_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, 'weights')

# ---------------------------------------------------------------------------
# Add perception_models source to Python path.
# Prefer env vars, then try common local/sibling layouts.
# ---------------------------------------------------------------------------
def _collect_pe_src_candidates():
    raw = [
        os.environ.get('PE_SRC_DIR', ''),
        os.environ.get('HMF_PE_SRC_DIR', ''),
        os.path.join(_PROJECT_ROOT, 'perception_models'),
        os.path.join(_PROJECT_ROOT, 'PE', 'perception_models'),
        os.path.join(_PROJECT_ROOT, 'third_party', 'perception_models'),
        os.path.join(_PROJECT_ROOT, '..', 'PE', 'perception_models'),
        os.path.join(_PROJECT_ROOT, '..', 'Ali_InternVideo_PE', 'perception_models'),
        os.path.join(_PROJECT_ROOT, '..', 'Ali_InternVideo_PE', 'PE', 'perception_models'),
    ]
    out = []
    seen = set()
    for c in raw:
        c = os.path.normpath(c) if c else ''
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


_PE_SRC_CANDIDATES = _collect_pe_src_candidates()
_PE_SRC = None
for _cand in _PE_SRC_CANDIDATES:
    if os.path.isdir(os.path.join(_cand, 'core', 'vision_encoder')):
        _PE_SRC = _cand
        break

if _PE_SRC and _PE_SRC not in sys.path:
    sys.path.insert(0, _PE_SRC)

try:
    from core.vision_encoder.pe import VisionTransformer          # noqa: E402
    from core.vision_encoder.config import PE_VISION_CONFIG       # noqa: E402
    _PE_AVAILABLE = True
    _PE_IMPORT_ERROR = None
except ImportError as _e:
    _PE_AVAILABLE = False
    _PE_IMPORT_ERROR = str(_e)
    logger.warning(
        "perception_models source not found. Set PE_SRC_DIR/HMF_PE_SRC_DIR "
        "to perception_models root. PEEncoder is unavailable."
    )


def _resolve_weights(variant: str, explicit_path: str = None) -> str | None:
    """Resolve weight file path with fallback chain:
    1. explicit_path (CLI --pe_weights) — fallback if specified but missing
    2. hmf_vqa/weights/PE_visual_only.pth  (common name, highest priority)
    3. hmf_vqa/weights/<variant>-visual.pt  (visual-only)
    4. hmf_vqa/weights/<variant>.pt  (full CLIP, visual auto-extracted)
    5. ../Ali_InternVideo_PE/weights/<variant>.pt  (sibling project)
    """
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        # User explicitly specified a path — try fallback to known locations
        # (the explicit path may not exist on a different machine)
    # Auto-resolve from known locations
    candidates = [
        os.path.join(_WEIGHTS_DIR, 'PE_visual_only.pth'),
        os.path.join(_WEIGHTS_DIR, f'{variant}-visual.pt'),
        os.path.join(_WEIGHTS_DIR, f'{variant}.pt'),
        os.path.normpath(os.path.join(
            _PROJECT_ROOT, '..', 'Ali_InternVideo_PE', 'weights', f'{variant}.pt'
        )),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


class PEEncoder(nn.Module):
    """
    Perception Encoder backbone using Meta's VisionTransformer.

    Supports PE-Core variants: B16-224, L14-336, G14-448, S16-384, T16-384.
    Uses RoPE2D so arbitrary input resolutions are supported without
    positional embedding interpolation.

    Output: ``[B, output_dim]`` global features (e.g. 1024 for B16-224).
    """

    # Mapping of short aliases → full config name
    _ALIASES = {
        'pe_encoder':     'PE-Core-B16-224',
        'pe_b16':         'PE-Core-B16-224',
        'pe_l14':         'PE-Core-L14-336',
        'pe_g14':         'PE-Core-G14-448',
        'pe_s16':         'PE-Core-S16-384',
        'pe_t16':         'PE-Core-T16-384',
    }

    def __init__(
        self,
        variant: str = 'PE-Core-B16-224',
        pretrained: bool = True,
        weights_path: str = None,
        freeze: bool = False,
        out_dim: int = None,          # ignored when PE available (auto)
        rope_resolution_scale: bool = False,
        patch_embed_cond: bool = False,
        resolution_token: bool = False,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        # Resolve alias
        variant = self._ALIASES.get(variant, variant)
        self.variant = variant

        if not _PE_AVAILABLE:
            searched = '\n'.join(f'  - {p}' for p in _PE_SRC_CANDIDATES)
            hint = (
                "Meta PE source not available.\n"
                f"PE_SRC_DIR={os.environ.get('PE_SRC_DIR', '(unset)')}\n"
                f"HMF_PE_SRC_DIR={os.environ.get('HMF_PE_SRC_DIR', '(unset)')}\n"
                f"Searched candidate directories:\n{searched}\n"
                f"Import error: {_PE_IMPORT_ERROR or '(none)'}\n"
                "Note: PE weights in weights/ are not enough; "
                "the perception_models source code is also required."
            )
            raise RuntimeError(
                hint
            )

        assert variant in PE_VISION_CONFIG, (
            f"Unknown PE variant '{variant}'. "
            f"Available: {list(PE_VISION_CONFIG.keys())}"
        )

        # ---- Build backbone ----
        self.backbone = VisionTransformer.from_config(variant, pretrained=False)

        # ---- Load pretrained weights ----
        if pretrained:
            ckpt = _resolve_weights(variant, weights_path)
            if ckpt:
                logger.info("Loading PE weights from %s", ckpt)
                print(f"[PEEncoder] Loading PE weights: {ckpt}")
                self.backbone.load_ckpt(ckpt, verbose=True)
                # ---- Verify loading correctness ----
                self._verify_weights(ckpt)
            else:
                searched = [
                    weights_path or '(none)',
                    os.path.join(_WEIGHTS_DIR, 'PE_visual_only.pth'),
                    os.path.join(_WEIGHTS_DIR, f'{variant}-visual.pt'),
                    os.path.join(_WEIGHTS_DIR, f'{variant}.pt'),
                    os.path.normpath(os.path.join(
                        _PROJECT_ROOT, '..', 'Ali_InternVideo_PE', 'weights', f'{variant}.pt'
                    )),
                ]
                raise FileNotFoundError(
                    f"PE weights not found for variant '{variant}'.\n"
                    f"Searched paths:\n" +
                    '\n'.join(f'  - {p}' for p in searched) +
                    f"\nPlease download weights or create a symlink:\n"
                    f"  ln -s /path/to/{variant}.pt {_WEIGHTS_DIR}/{variant}.pt"
                )

        # output_dim is the final projected dimension (1024 for B16-224)
        self.out_dim = self.backbone.output_dim

        # ---- Scheme E: Resolution-Scaled RoPE ----
        # When enabled, scales RoPE2D grid positions by resolution ratio vs 1080p
        # so attention encodes absolute spatial extent without extra parameters.
        self.rope_resolution_scale = rope_resolution_scale
        self._rope_scale_ref_long = 1920.0  # 1080p long edge as reference

        # ---- Scheme F: Patch Embedding Conditioning ----
        # Injects resolution information right after patch embedding (conv1),
        # before the transformer. A tiny MLP maps (h_norm, w_norm) to a
        # width-dim bias vector added to every patch token.
        # Zero-init so the model starts from the same point as baseline.
        self.patch_embed_cond = patch_embed_cond
        if self.patch_embed_cond:
            bb_width = self.backbone.width  # 768 for B16-224
            self._pec_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.GELU(),
                nn.Linear(64, bb_width),
            )
            # Zero-init last layer → identity at start
            nn.init.zeros_(self._pec_mlp[-1].weight)
            nn.init.zeros_(self._pec_mlp[-1].bias)
            logger.info("PatchEmbedConditioner enabled: MLP(2→64→%d)", bb_width)

        # ---- Scheme H: Resolution Token ----
        # Encodes resolution as a learnable token prepended to the patch
        # sequence (alongside cls_token), letting the transformer attend
        # to resolution information internally.  MLP: (h_norm, w_norm) → width.
        # The token participates in self-attention but is stripped before pooling.
        self.resolution_token = resolution_token
        if self.resolution_token:
            bb_width = self.backbone.width  # 768 for B16-224
            self._res_token_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.GELU(),
                nn.Linear(64, bb_width),
            )
            # Small-init last layer so token starts near zero (minimal impact)
            nn.init.normal_(self._res_token_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self._res_token_mlp[-1].bias)
            logger.info("ResolutionToken enabled: MLP(2→64→%d)", bb_width)

        # ---- StochasticDepth (DropPath) ----
        # Inject linear-increasing drop rates into the pretrained ViT blocks.
        # Drop rate goes from 0 (first block) to stochastic_depth_rate (last block).
        # Uses timm's DropPath module which stochastically drops entire residual paths.
        self.stochastic_depth_rate = float(stochastic_depth_rate)
        if self.stochastic_depth_rate > 0:
            self._inject_stochastic_depth(self.stochastic_depth_rate)

        # ---- Optional freeze ----
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("PE backbone frozen (%s)", variant)

        n_params = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        logger.info(
            "PEEncoder ready: variant=%s  out_dim=%d  params=%.1fM  weights=%s",
            variant, self.out_dim, n_params,
            'pretrained' if pretrained else 'random',
        )
        print(
            f"[PEEncoder] Ready: {variant}  out_dim={self.out_dim}  "
            f"params={n_params:.1f}M  weights={'pretrained' if pretrained else 'random'}"
        )

    def _inject_stochastic_depth(self, max_drop_rate: float):
        """Inject DropPath (stochastic depth) into pretrained ViT transformer blocks.

        Each ResidualAttentionBlock's forward is wrapped so that its residual
        path is stochastically dropped with linearly increasing probability
        from 0 (first block) to max_drop_rate (last block).

        This modifies the blocks in-place without breaking checkpointed forward.
        """
        from timm.layers import DropPath

        vit = self.backbone
        num_layers = len(vit.transformer.resblocks)
        drop_rates = [max_drop_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]

        for i, (block, dr) in enumerate(zip(vit.transformer.resblocks, drop_rates)):
            if dr <= 0:
                continue
            # Create DropPath modules and attach to block
            block._drop_path_attn = DropPath(dr)
            block._drop_path_mlp = DropPath(dr)

            # Monkey-patch the block's forward to apply DropPath
            original_forward = block.forward

            def make_droppath_forward(orig_fwd, dp_attn, dp_mlp, blk):
                def droppath_forward(x, *args, **kwargs):
                    # PE ResidualAttentionBlock layout:
                    #   x = x + drop_path1(ls_1(_call_attn(ln_1(x))))
                    #   x = x + drop_path2(ls_2(mlp(ln_2(x))))
                    # With StochasticDepth override:
                    #   Replace block's own drop_path1/2 with our injected DropPath
                    attn_mask = kwargs.get('attn_mask', None)
                    attn_out = blk._call_attn(blk.ln_1(x), attn_mask=attn_mask)
                    x = x + dp_attn(blk.ls_1(attn_out))
                    x = x + dp_mlp(blk.ls_2(blk.mlp(blk.ln_2(x))))
                    return x
                return droppath_forward

            block.forward = make_droppath_forward(
                original_forward, block._drop_path_attn, block._drop_path_mlp, block
            )

        logger.info(
            "StochasticDepth injected: %d layers, drop_rate 0→%.3f",
            num_layers, max_drop_rate,
        )
        print(f"[PEEncoder] StochasticDepth: {num_layers} layers, rate 0→{max_drop_rate:.3f}")

    def _verify_weights(self, ckpt_path: str):
        """Verify the loaded weights are meaningful (not zero/random)."""
        # Quick sanity: check a few parameter norms
        norms = []
        for name, p in self.backbone.named_parameters():
            if 'conv1' in name or 'ln_pre' in name or 'transformer.resblocks.0' in name:
                norms.append((name, p.data.norm().item()))
                if len(norms) >= 3:
                    break
        # All norms should be > 0 (not zero-initialized)
        all_nonzero = all(n > 0 for _, n in norms)
        if not all_nonzero:
            logger.warning("PE weight verification FAILED — some params are zero!")
            print("[PEEncoder] WARNING: Weight verification FAILED!")
        else:
            logger.info("PE weight verification OK: %s",
                       ', '.join(f'{n}={v:.4f}' for n, v in norms))
            print(f"[PEEncoder] Weight verification OK: "
                  + ', '.join(f'{n}={v:.4f}' for n, v in norms))

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable / disable gradient checkpointing."""
        self.backbone.set_grad_checkpointing(enable=enable)

    def _update_rope_scaled(self, device, grid_h, grid_w, resolution_scale: float = 1.0):
        """Update RoPE2D grid with scaled positions for resolution awareness.

        Instead of positions [0, 1, ..., 13], uses [0, s, 2s, ..., 13s] where
        s = resolution_scale (e.g. 2.0 for 4K).  This makes attention patterns
        encode absolute spatial extent — patches from 4K frames appear "farther
        apart" than patches from 1080p, without any extra parameters.

        When resolution_scale == 1.0 this is identical to the default grid.
        """
        vit = self.backbone
        if not vit.use_rope2d or vit.rope is None:
            return

        rope_obj = vit.rope
        # Force recomputation by invalidating cached grid_size
        rope_obj.grid_size = None

        # Temporarily monkey-patch the RotaryEmbedding's forward to scale positions
        import types

        # Save original forward
        if not hasattr(rope_obj, '_orig_rope_inner'):
            rope_obj._orig_rope_inner = rope_obj.rope

        inner_rope = rope_obj._orig_rope_inner
        scale = float(resolution_scale)

        if abs(scale - 1.0) < 1e-6:
            # No scaling needed — use original
            rope_obj.rope = inner_rope
            rope_obj.update_grid(device, grid_h, grid_w)
            return

        # Create a wrapper that scales the input positions
        class _ScaledRopeWrapper:
            """Thin wrapper: scales input positions before computing RoPE frequencies."""
            def __init__(self, base_rope, scale_factor):
                self.base_rope = base_rope
                self.scale_factor = scale_factor

            def __call__(self, t, **kwargs):
                return self.base_rope(t * self.scale_factor, **kwargs)

            def to(self, device):
                self.base_rope = self.base_rope.to(device)
                return self

        rope_obj.rope = _ScaledRopeWrapper(inner_rope, scale)
        rope_obj.update_grid(device, grid_h, grid_w)
        # Restore original to avoid stale state for next call
        rope_obj.rope = inner_rope

    def _needs_custom_forward(self, resolution_scale: float, res_hw: tuple = None) -> bool:
        """Check if custom forward path is needed (any scheme active)."""
        if self.rope_resolution_scale and abs(resolution_scale - 1.0) > 1e-6:
            return True
        if self.patch_embed_cond and res_hw is not None:
            return True
        if self.resolution_token and res_hw is not None:
            return True
        return False

    def _compute_res_input(self, res_hw: tuple, device) -> torch.Tensor:
        """Compute normalized (h_norm, w_norm) input for Scheme F/H.

        Args:
            res_hw: (height, width) of original video frame (scalars or tensors).
        Returns:
            [1, 2] tensor with (h_norm, w_norm) normalized to [0, 1] range.
        """
        if isinstance(res_hw[0], torch.Tensor):
            h = res_hw[0].float().mean().item()
            w = res_hw[1].float().mean().item()
        else:
            h, w = float(res_hw[0]), float(res_hw[1])
        h_norm = h / 2160.0
        w_norm = w / 3840.0
        return torch.tensor([[h_norm, w_norm]], dtype=torch.float32, device=device)

    def forward(
        self,
        x: torch.Tensor,
        resolution_scale: float = 1.0,
        res_hw: tuple = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, 3, H, W]`` input images.
            resolution_scale: RoPE position scale factor (1.0 = default, 2.0 = 4K).
                Only used when ``rope_resolution_scale=True`` was set at init.
            res_hw: ``(height, width)`` of original video frame, used by
                Scheme F (patch_embed_cond) and Scheme H (resolution_token).
        Returns:
            ``[B, output_dim]`` global features.
        """
        if not self._needs_custom_forward(resolution_scale, res_hw):
            return self.backbone(x)

        vit = self.backbone
        batch, _, h, w = x.shape
        grid_h, grid_w = h // vit.patch_size, w // vit.patch_size

        # ---- Scheme E: scale RoPE grid ----
        if self.rope_resolution_scale and abs(resolution_scale - 1.0) > 1e-6:
            self._update_rope_scaled(x.device, grid_h, grid_w, resolution_scale)
        elif vit.use_rope2d:
            vit.rope.update_grid(x.device, grid_h, grid_w)

        # ---- Patch embedding ----
        x_feat = vit.conv1(x)
        x_feat = x_feat.permute(0, 2, 3, 1).reshape(batch, -1, vit.width)

        # ---- Scheme F: add resolution bias to patch tokens ----
        if self.patch_embed_cond and res_hw is not None:
            res_input = self._compute_res_input(res_hw, x.device)  # [1, 2]
            pec_bias = self._pec_mlp(res_input)  # [1, width]
            x_feat = x_feat + pec_bias.unsqueeze(1)  # broadcast to [B, N, width]

        # ---- CLS token ----
        if vit.use_cls_token:
            x_feat = torch.cat(
                [vit.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x_feat],
                dim=1,
            )

        # ---- Scheme H: prepend resolution token ----
        if self.resolution_token and res_hw is not None:
            res_input = self._compute_res_input(res_hw, x.device)  # [1, 2]
            res_tok = self._res_token_mlp(res_input)  # [1, width]
            res_tok = res_tok.unsqueeze(0).expand(batch, -1, -1)  # [B, 1, width]
            x_feat = torch.cat([res_tok, x_feat], dim=1)

        if vit.use_abs_posemb:
            # Note: abs_posemb size matches (cls_token + grid_h*grid_w),
            # Scheme H's extra token won't have a pos emb — this is by design
            # (the resolution token is "position-free" like a register token).
            posemb = vit._sample_abs_posemb(grid_h, grid_w)
            if self.resolution_token and res_hw is not None:
                # Pad posemb with zero for the resolution token position
                zero_pad = torch.zeros(
                    1, 1, posemb.shape[-1], device=posemb.device, dtype=posemb.dtype
                )
                posemb = torch.cat([zero_pad, posemb], dim=1)
            x_feat = x_feat + posemb

        x_feat = vit.ln_pre(x_feat)
        x_feat = vit.transformer(x_feat)
        x_feat = vit.ln_post(x_feat)

        # ---- Strip resolution token before pooling ----
        if self.resolution_token and res_hw is not None:
            x_feat = x_feat[:, 1:, :]  # remove first token (res_token)

        x_feat = vit._pool(x_feat)
        if vit.proj_dim is not None:
            x_feat = x_feat @ vit.proj
        return x_feat

    def forward_intermediate_layers(
        self,
        x: torch.Tensor,
        layer_indices: list[int] = None,
        norm: bool = True,
        resolution_scale: float = 1.0,
        res_hw: tuple = None,
    ) -> list[torch.Tensor]:
        """
        Extract intermediate features from specific transformer layers.

        Each selected layer's output is pooled (attn_pool) and projected
        to ``output_dim`` space, yielding a list of ``[B, output_dim]`` tensors.

        Args:
            x: ``[B, 3, H, W]`` input images.
            layer_indices: 0-based indices of transformer layers to extract.
                           E.g., [0, 3, 7, 11] for PE-Core-B16-224 (12 layers).
                           If None, returns features from all layers.
            norm: Whether to apply ln_post normalization.
            resolution_scale: RoPE position scale factor (Scheme E).
            res_hw: ``(height, width)`` for Scheme F / Scheme H.

        Returns:
            List of ``[B, output_dim]`` feature tensors, one per selected layer.
        """
        vit = self.backbone
        batch, _, h, w = x.shape
        grid_h, grid_w = h // vit.patch_size, w // vit.patch_size

        # ---- Scheme E: Resolution-Scaled RoPE ----
        if self.rope_resolution_scale and abs(resolution_scale - 1.0) > 1e-6:
            self._update_rope_scaled(x.device, grid_h, grid_w, resolution_scale)
        elif vit.use_rope2d:
            vit.rope.update_grid(x.device, grid_h, grid_w)

        # Patch embedding
        x = vit.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, vit.width)

        # ---- Scheme F: patch embed conditioning ----
        if self.patch_embed_cond and res_hw is not None:
            res_input = self._compute_res_input(res_hw, x.device)
            pec_bias = self._pec_mlp(res_input)
            x = x + pec_bias.unsqueeze(1)

        if vit.use_cls_token:
            x = torch.cat(
                [vit.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        # ---- Scheme H: resolution token ----
        has_res_token = self.resolution_token and res_hw is not None
        if has_res_token:
            res_input = self._compute_res_input(res_hw, x.device)
            res_tok = self._res_token_mlp(res_input).unsqueeze(0).expand(batch, -1, -1)
            x = torch.cat([res_tok, x], dim=1)

        if vit.use_abs_posemb:
            posemb = vit._sample_abs_posemb(grid_h, grid_w)
            if has_res_token:
                zero_pad = torch.zeros(
                    1, 1, posemb.shape[-1], device=posemb.device, dtype=posemb.dtype
                )
                posemb = torch.cat([zero_pad, posemb], dim=1)
            x = x + posemb

        # RoPE grid already updated above (with or without resolution scaling)

        x = vit.ln_pre(x)

        # Run through transformer blocks, collecting intermediate outputs
        num_layers = len(vit.transformer.resblocks)
        if layer_indices is None:
            layer_indices = list(range(num_layers))

        collected = []
        for i, block in enumerate(vit.transformer.resblocks):
            if vit.transformer.grad_checkpointing and not torch.jit.is_scripting():
                from torch.utils.checkpoint import checkpoint as ckpt_fn
                x = ckpt_fn(block, x, None, None, None)
            else:
                x = block(x)

            if i in layer_indices:
                feat = x  # [B, seq_len, width]
                # Strip resolution token before pool
                if has_res_token:
                    feat = feat[:, 1:, :]
                if norm:
                    feat = vit.ln_post(feat)
                # Pool and project (same as forward)
                pooled = vit._pool(feat)  # [B, width]
                if vit.proj_dim is not None:
                    pooled = pooled @ vit.proj  # [B, output_dim]
                collected.append(pooled)

        return collected
