"""
Spatial Sampler: GMS / Resize / FuPiC strategies.

All samplers operate on YUV float32 [3, T, H, W] tensors.
FR mode: ref and dis are sampled at the same spatial positions.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class GMSSampler:
    """
    Grid Mini-patch Sampling (GMS).
    Samples P patches from a spatial grid on the original resolution.
    Similar to FastVQA's fragment sampling.

    Args:
        patch_size: spatial size of each patch (e.g. 32, 64)
        patches_per_frame: P, total patches per frame
        grid_size: size of the virtual grid (e.g. 7x7)
        is_train: if True, random patch position within each grid cell
    """

    def __init__(
        self,
        patch_size: int = 64,
        patches_per_frame: int = 12,
        grid_size: int = 7,
        is_train: bool = True,
    ):
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.grid_size = grid_size
        self.is_train = is_train

    def sample(
        self,
        dis_yuv: torch.Tensor,
        ref_yuv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample GMS patches from YUV tensors.

        Args:
            dis_yuv: [3, T, H, W]
            ref_yuv: [3, T, H, W] or None

        Returns:
            dict with:
                'dis_patches': [N, 3, ph, pw] where N = T * P
                'ref_patches': [N, 3, ph, pw] if ref provided
                'positions': [N, 2] normalized (y, x) center positions
        """
        _, T, H, W = dis_yuv.shape
        ph, pw = self.patch_size, self.patch_size

        # Compute grid cell positions
        positions = self._get_patch_positions(H, W)  # List of (y, x) top-left coords
        P = len(positions)

        # Build normalized center positions (same semantics as legacy path)
        pos_list = [
            [(y + ph / 2) / H, (x + pw / 2) / W] for (y, x) in positions
        ]
        positions_tensor = torch.tensor(pos_list, dtype=torch.float32)  # [P, 2]
        # Tile across T frames to match the legacy [N=T*P, 2] order
        positions_tensor = positions_tensor.unsqueeze(0).expand(T, P, 2).reshape(T * P, 2).contiguous()

        # Fast path: no boundary overflow (the common case; _get_patch_positions
        # already clamps to [0, H-ph] × [0, W-pw] when H>=ph and W>=pw).
        # Fall back to the legacy per-patch loop (with F.pad) only for tiny
        # frames where a patch can exceed the frame bounds — preserves bit-exact
        # equivalence with the original implementation.
        fast_path = (H >= ph) and (W >= pw)

        if fast_path:
            # dis_yuv: [3, T, H, W]
            # Build row/col index tensors
            ys = torch.as_tensor([p[0] for p in positions], dtype=torch.long)  # [P]
            xs = torch.as_tensor([p[1] for p in positions], dtype=torch.long)  # [P]
            # Row indices: [P, ph]
            y_idx = ys.unsqueeze(1) + torch.arange(ph, dtype=torch.long).unsqueeze(0)
            # Col indices: [P, pw]
            x_idx = xs.unsqueeze(1) + torch.arange(pw, dtype=torch.long).unsqueeze(0)

            # Step 1: gather rows per patch → [3, T, P, ph, W]
            rows = dis_yuv[:, :, y_idx, :]
            # Step 2: gather cols per patch using torch.gather along the last dim.
            #   cols_idx: [1, 1, P, ph, pw]  (broadcast over C=3 and T)
            cols_idx = x_idx.unsqueeze(1).expand(P, ph, pw).unsqueeze(0).unsqueeze(0)  # [1,1,P,ph,pw]
            cols_idx = cols_idx.expand(3, T, P, ph, pw)
            patches_4d = torch.gather(rows, dim=-1, index=cols_idx)  # [3, T, P, ph, pw]
            # Reorder to legacy layout: iterate t then p → [N=T*P, 3, ph, pw]
            dis_patches_stacked = patches_4d.permute(1, 2, 0, 3, 4).reshape(T * P, 3, ph, pw).contiguous()
        else:
            # Legacy fallback for edge cases where patch can overflow.
            dis_patches = []
            for t in range(T):
                frame_dis = dis_yuv[:, t]  # [3, H, W]
                for (y, x) in positions:
                    patch = frame_dis[:, y:y + ph, x:x + pw]
                    if patch.shape[1] != ph or patch.shape[2] != pw:
                        patch = F.pad(patch, (0, pw - patch.shape[2], 0, ph - patch.shape[1]))
                    dis_patches.append(patch)
            dis_patches_stacked = torch.stack(dis_patches)

        result = {
            'dis_patches': dis_patches_stacked,  # [N, 3, ph, pw]
            'positions': positions_tensor,        # [N, 2]
        }

        if ref_yuv is not None:
            if fast_path:
                rows_ref = ref_yuv[:, :, y_idx, :]                                    # [3, T, P, ph, W]
                ref_patches_4d = torch.gather(rows_ref, dim=-1, index=cols_idx)       # [3, T, P, ph, pw]
                result['ref_patches'] = ref_patches_4d.permute(1, 2, 0, 3, 4).reshape(T * P, 3, ph, pw).contiguous()
            else:
                ref_patches = []
                for t in range(T):
                    frame_ref = ref_yuv[:, t]
                    for (y, x) in positions:
                        patch = frame_ref[:, y:y + ph, x:x + pw]
                        if patch.shape[1] != ph or patch.shape[2] != pw:
                            patch = F.pad(patch, (0, pw - patch.shape[2], 0, ph - patch.shape[1]))
                        ref_patches.append(patch)
                result['ref_patches'] = torch.stack(ref_patches)

        return result

    def _get_patch_positions(self, H: int, W: int) -> List[Tuple[int, int]]:
        """Generate patch top-left positions from the grid."""
        ph, pw = self.patch_size, self.patch_size
        gs = self.grid_size

        # Grid cell dimensions
        cell_h = max(1, (H - ph) // gs)
        cell_w = max(1, (W - pw) // gs)

        # Generate all grid cell centers
        all_cells = []
        for gy in range(gs):
            for gx in range(gs):
                y0 = gy * cell_h
                x0 = gx * cell_w
                all_cells.append((y0, x0, cell_h, cell_w))

        # Select P cells
        if len(all_cells) <= self.patches_per_frame:
            selected = all_cells
        elif self.is_train:
            selected = random.sample(all_cells, self.patches_per_frame)
        else:
            # Deterministic: evenly spaced selection
            step = len(all_cells) / self.patches_per_frame
            selected = [all_cells[int(i * step)] for i in range(self.patches_per_frame)]

        # Get actual positions
        positions = []
        for (y0, x0, ch, cw) in selected:
            if self.is_train:
                y = y0 + random.randint(0, max(0, ch - 1))
                x = x0 + random.randint(0, max(0, cw - 1))
            else:
                y = y0 + ch // 2
                x = x0 + cw // 2
            # Clamp
            y = min(max(y, 0), H - ph)
            x = min(max(x, 0), W - pw)
            positions.append((y, x))

        return positions


class ResizeSampler:
    """
    Resize entire frame to a fixed size (e.g. 224x224 or 384x384).
    Used for semantic branch.
    """

    def __init__(
        self,
        target_size: int = 224,
        mode: str = 'bilinear',
        center_crop: bool = False,
    ):
        self.target_size = target_size
        self.mode = mode
        self.center_crop = center_crop

    def sample(
        self,
        dis_yuv: torch.Tensor,
        ref_yuv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Resize frames.

        Args:
            dis_yuv: [3, T, H, W]

        Returns:
            dict with 'dis_frames': [T, 3, target, target]
        """
        _, T, H, W = dis_yuv.shape
        ts = self.target_size

        # [3, T, H, W] -> [T, 3, H, W]
        dis_frames = dis_yuv.permute(1, 0, 2, 3)

        if self.center_crop:
            # Center crop to square first
            short_side = min(H, W)
            cy, cx = H // 2, W // 2
            half = short_side // 2
            dis_frames = dis_frames[:, :, cy - half:cy + half, cx - half:cx + half]

        align = self.mode != 'nearest'
        dis_resized = F.interpolate(
            dis_frames.float(), size=(ts, ts), mode=self.mode,
            align_corners=align if align else None,
        )

        result = {'dis_frames': dis_resized}  # [T, 3, ts, ts]

        if ref_yuv is not None:
            ref_frames = ref_yuv.permute(1, 0, 2, 3)
            if self.center_crop:
                ref_frames = ref_frames[:, :, cy - half:cy + half, cx - half:cx + half]
            ref_resized = F.interpolate(
                ref_frames.float(), size=(ts, ts), mode=self.mode,
                align_corners=align if align else None,
            )
            result['ref_frames'] = ref_resized

        return result


class FragmentSampler:
    """
    Fragment Sampling (FastVQA / DOVER style).

    Divides the frame into a grid_h × grid_w grid. From each grid cell, crops a
    small frag_size × frag_size fragment, then stitches all fragments into ONE composite
    image of size (grid_h * frag_size) × (grid_w * frag_size).

    Example: grid 7×7, frag_size 32 → composite 224×224 covering the entire frame.

    Key advantages over GMS:
      - Full spatial coverage (every grid cell contributes)
      - Memory efficient: P=1 composite image per frame (vs P separate patches in GMS)
      - Matched to FastVQA/DOVER proven fragment-based quality assessment approach

    The composite image preserves relative spatial layout: top-left fragment comes from
    the top-left region of the original frame, etc.

    Output format is compatible with the GMS pipeline:
      gms_dis: [1, 3, T, composite_h, composite_w]  (P=1)
    """

    def __init__(
        self,
        grid_h: int = 7,
        grid_w: int = 7,
        frag_size: int = 32,
        is_train: bool = True,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.frag_size = frag_size
        self.is_train = is_train
        # Expose as properties for compatibility with GMS interface
        self.patches_per_frame = 1  # single composite
        self.patch_size = grid_h * frag_size  # composite size (e.g. 224)

    def _get_fragment_positions(self, H: int, W: int) -> List[Tuple[int, int]]:
        """
        Compute top-left crop position for each grid cell.

        Returns:
            List of (y, x) positions, one per grid cell, in row-major order.
        """
        fs = self.frag_size
        positions = []

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                # Grid cell boundaries
                y_start = int(gy * H / self.grid_h)
                y_end = int((gy + 1) * H / self.grid_h)
                x_start = int(gx * W / self.grid_w)
                x_end = int((gx + 1) * W / self.grid_w)

                # Ensure fragment fits within cell
                max_y = max(y_start, y_end - fs)
                max_x = max(x_start, x_end - fs)
                min_y = y_start
                min_x = x_start

                if self.is_train:
                    y = random.randint(min_y, max(min_y, max_y))
                    x = random.randint(min_x, max(min_x, max_x))
                else:
                    # Deterministic: center of cell
                    y = (min_y + max_y) // 2
                    x = (min_x + max_x) // 2

                # Clamp to frame
                y = min(max(y, 0), max(0, H - fs))
                x = min(max(x, 0), max(0, W - fs))
                positions.append((y, x))

        return positions

    def sample_composite(
        self,
        dis_yuv: torch.Tensor,
        ref_yuv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build composite fragment image from each frame.

        Args:
            dis_yuv: [3, T, H, W]
            ref_yuv: [3, T, H, W] or None

        Returns:
            dict with:
                'dis_composite': [3, T, composite_h, composite_w]
                'ref_composite': same if ref provided
        """
        _, T, H, W = dis_yuv.shape
        fs = self.frag_size
        gh, gw = self.grid_h, self.grid_w
        ch, cw = gh * fs, gw * fs  # composite size
        P = gh * gw

        # Get fragment positions (same for all frames in this sample)
        positions = self._get_fragment_positions(H, W)

        # Fast path requires all fragments fit within frame bounds.
        # _get_fragment_positions clamps to [0, max(0, H-fs)] × [0, max(0, W-fs)],
        # which is safe when H>=fs and W>=fs. Otherwise fall back to the legacy
        # per-frame / per-fragment loop with F.pad for bit-exact equivalence.
        fast_path = (H >= fs) and (W >= fs)

        if fast_path:
            # Build index tensors once (row-major over grid cells).
            ys = torch.as_tensor([p[0] for p in positions], dtype=torch.long)  # [P]
            xs = torch.as_tensor([p[1] for p in positions], dtype=torch.long)  # [P]
            y_idx = ys.unsqueeze(1) + torch.arange(fs, dtype=torch.long).unsqueeze(0)  # [P, fs]
            x_idx = xs.unsqueeze(1) + torch.arange(fs, dtype=torch.long).unsqueeze(0)  # [P, fs]

            # dis_yuv: [3, T, H, W]
            rows_dis = dis_yuv[:, :, y_idx, :]                                  # [3, T, P, fs, W]
            cols_idx = x_idx.unsqueeze(1).expand(P, fs, fs).unsqueeze(0).unsqueeze(0)  # [1,1,P,fs,fs]
            cols_idx_dis = cols_idx.expand(3, T, P, fs, fs)
            frags_dis = torch.gather(rows_dis, dim=-1, index=cols_idx_dis)      # [3, T, P, fs, fs]
            # Reshape P → (gh, gw), then interleave to composite HxW layout.
            frags_dis = frags_dis.view(3, T, gh, gw, fs, fs)
            # [3, T, gh, gw, fs, fs] → [3, T, gh, fs, gw, fs] → [3, T, gh*fs, gw*fs]
            dis_composite = frags_dis.permute(0, 1, 2, 4, 3, 5).reshape(3, T, ch, cw).contiguous()
        else:
            # Legacy fallback (tiny frames): preserves bit-exact output incl. F.pad.
            dis_composites = []
            for t in range(T):
                frame = dis_yuv[:, t]  # [3, H, W]
                composite = torch.zeros(3, ch, cw, dtype=frame.dtype, device=frame.device)
                for idx, (y, x) in enumerate(positions):
                    gy_idx = idx // gw
                    gx_idx = idx % gw
                    frag = frame[:, y:y + fs, x:x + fs]
                    if frag.shape[1] != fs or frag.shape[2] != fs:
                        frag = F.pad(frag, (0, fs - frag.shape[2], 0, fs - frag.shape[1]))
                    composite[:, gy_idx * fs:(gy_idx + 1) * fs,
                               gx_idx * fs:(gx_idx + 1) * fs] = frag
                dis_composites.append(composite)
            dis_composite = torch.stack(dis_composites, dim=1)  # [3, T, ch, cw]

        result = {
            'dis_composite': dis_composite,  # [3, T, ch, cw]
        }

        if ref_yuv is not None:
            if fast_path:
                rows_ref = ref_yuv[:, :, y_idx, :]                              # [3, T, P, fs, W]
                cols_idx_ref = cols_idx.expand(3, T, P, fs, fs)
                frags_ref = torch.gather(rows_ref, dim=-1, index=cols_idx_ref)  # [3, T, P, fs, fs]
                frags_ref = frags_ref.view(3, T, gh, gw, fs, fs)
                result['ref_composite'] = frags_ref.permute(0, 1, 2, 4, 3, 5).reshape(3, T, ch, cw).contiguous()
            else:
                ref_composites = []
                for t in range(T):
                    frame = ref_yuv[:, t]
                    composite = torch.zeros(3, ch, cw, dtype=frame.dtype, device=frame.device)
                    for idx, (y, x) in enumerate(positions):
                        gy_idx = idx // gw
                        gx_idx = idx % gw
                        frag = frame[:, y:y + fs, x:x + fs]
                        if frag.shape[1] != fs or frag.shape[2] != fs:
                            frag = F.pad(frag, (0, fs - frag.shape[2], 0, fs - frag.shape[1]))
                        composite[:, gy_idx * fs:(gy_idx + 1) * fs,
                                   gx_idx * fs:(gx_idx + 1) * fs] = frag
                    ref_composites.append(composite)
                result['ref_composite'] = torch.stack(ref_composites, dim=1)

        return result


class FuPiCSampler:
    """
    Full-Picture Coverage (FuPiC) tiling sampler for inference.
    Tiles the full frame with overlap, used for high-resolution evaluation.

    Args:
        tile_size: size of each tile
        stride: stride between tiles (< tile_size for overlap)
    """

    def __init__(self, tile_size: int = 224, stride: int = 192):
        self.tile_size = tile_size
        self.stride = stride

    def sample(
        self,
        dis_yuv: torch.Tensor,
        ref_yuv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tile the frames.

        Args:
            dis_yuv: [3, T, H, W]

        Returns:
            dict with:
                'dis_tiles': [Ntiles, 3, tile, tile] (per-frame tiles flattened across T)
                'ref_tiles': same if ref provided
                'tile_positions': [Ntiles, 4] (y, x, h, w) normalized
                'tiles_per_frame': int
        """
        _, T, H, W = dis_yuv.shape
        ts = self.tile_size
        st = self.stride

        # Compute tile grid
        ys = list(range(0, max(1, H - ts + 1), st))
        xs = list(range(0, max(1, W - ts + 1), st))
        # Ensure last tile covers the edge
        if ys and ys[-1] + ts < H:
            ys.append(H - ts)
        if xs and xs[-1] + ts < W:
            xs.append(W - ts)
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]

        tiles_per_frame = len(ys) * len(xs)

        dis_tiles = []
        tile_positions = []

        for t in range(T):
            frame = dis_yuv[:, t]  # [3, H, W]
            for y in ys:
                for x in xs:
                    tile = frame[:, y:y + ts, x:x + ts]
                    if tile.shape[1] != ts or tile.shape[2] != ts:
                        tile = F.pad(tile, (0, ts - tile.shape[2], 0, ts - tile.shape[1]))
                    dis_tiles.append(tile)
                    tile_positions.append([y / H, x / W, ts / H, ts / W])

        result = {
            'dis_tiles': torch.stack(dis_tiles),
            'tile_positions': torch.tensor(tile_positions, dtype=torch.float32),
            'tiles_per_frame': tiles_per_frame,
        }

        if ref_yuv is not None:
            ref_tiles = []
            for t in range(T):
                frame = ref_yuv[:, t]
                for y in ys:
                    for x in xs:
                        tile = frame[:, y:y + ts, x:x + ts]
                        if tile.shape[1] != ts or tile.shape[2] != ts:
                            tile = F.pad(tile, (0, ts - tile.shape[2], 0, ts - tile.shape[1]))
                        ref_tiles.append(tile)
            result['ref_tiles'] = torch.stack(ref_tiles)

        return result


def build_spatial_sampler(name: str, cfg: dict, is_train: bool = True):
    """Factory function to build a spatial sampler from config."""
    if name == 'gms':
        return GMSSampler(
            patch_size=cfg.get('gms_patch', 256),
            patches_per_frame=cfg.get('gms_patches_per_frame', 8),
            grid_size=cfg.get('gms_grid_size', 7),
            is_train=is_train,
        )
    elif name == 'fragment':
        return FragmentSampler(
            grid_h=cfg.get('fragment_grid_h', 7),
            grid_w=cfg.get('fragment_grid_w', 7),
            frag_size=cfg.get('fragment_size', 32),
            is_train=is_train,
        )
    elif name.startswith('resize'):
        # Support arbitrary resize: resize224, resize384, resize512, etc.
        size_str = name[len('resize'):]
        try:
            target = int(size_str)
        except ValueError:
            raise ValueError(f"Invalid resize sampler name '{name}', expected 'resizeNNN' (e.g. resize224, resize512)")
        return ResizeSampler(target_size=target)
    elif name == 'center_crop':
        return ResizeSampler(target_size=cfg.get('semantic_target_size', 224), center_crop=True)
    elif name == 'fullres':
        # No spatial sampling - return as-is
        return None
    elif name == 'fupic':
        return FuPiCSampler(
            tile_size=cfg.get('fupic_tile', 224),
            stride=cfg.get('fupic_stride', 192),
        )
    else:
        raise ValueError(f"Unknown spatial sampler: {name}")
