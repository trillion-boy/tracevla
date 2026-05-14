"""
latent_foveation.py

Spatially-varying scale modulation for projected ViT patch embeddings.

Patch layout after PrismaticVisionBackbone for 2-image TraceVLA input
(num_steps=2, image_size=224, patch_size=14 → grid 16×16=256):

  [0   : 256]  original observation patches
  [256]        separator token  (untouched)
  [257 : 513]  trace-overlaid image patches

Weight zones (all configurable):
  fovea      – tokens whose patch centre falls inside `fovea_bbox`      → fovea_scale   (default 1.2)
  secondary  – tokens inside `secondary_bbox` (e.g. hand / src object)  → secondary_scale (default 0.7)
  background – all remaining tokens                                      → bg_scale      (default 0.4)

Usage:
    lf = LatentFoveation(image_size=224, patch_size=14)
    weighted = lf(projected_patch_embeddings, fovea_bbox)   # fovea_bbox: [B, 4] pixel xyxy
"""

from typing import Optional

import torch
import torch.nn as nn


class LatentFoveation(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        fovea_scale: float = 1.2,
        secondary_scale: float = 0.7,
        bg_scale: float = 0.4,
        apply_to_trace_image: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size   # 16
        self.num_patches = self.grid_size ** 2       # 256
        self.fovea_scale = fovea_scale
        self.secondary_scale = secondary_scale
        self.bg_scale = bg_scale
        self.apply_to_trace_image = apply_to_trace_image

    # ------------------------------------------------------------------
    def _make_weight_map(
        self,
        fovea_bbox: torch.Tensor,                    # [B, 4] pixel-space xyxy
        secondary_bbox: Optional[torch.Tensor],       # [B, 4] or None
    ) -> torch.Tensor:                               # [B, num_patches]
        B = fovea_bbox.shape[0]
        G = self.grid_size
        dev = fovea_bbox.device

        # Normalise bbox to [0, 1]
        scale = float(self.image_size)
        bbox = fovea_bbox.float() / scale             # [B, 4]

        # Patch centre coordinates on the [0,1] grid – [G*G] each
        t = torch.arange(G, device=dev, dtype=torch.float32)
        cx = (t + 0.5) / G                            # column centres [G]
        cy = (t + 0.5) / G                            # row centres    [G]
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")  # [G, G]
        gx = grid_x.reshape(-1)                        # [G*G]
        gy = grid_y.reshape(-1)                        # [G*G]

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]  # [B]

        in_fovea = (
            (gx[None] >= x1[:, None]) & (gx[None] <= x2[:, None]) &
            (gy[None] >= y1[:, None]) & (gy[None] <= y2[:, None])
        )  # [B, G*G]  bool

        weights = torch.full(
            (B, G * G), self.bg_scale,
            device=dev, dtype=fovea_bbox.dtype,
        )

        if secondary_bbox is not None:
            sec = secondary_bbox.float() / scale
            sx1, sy1, sx2, sy2 = sec[:, 0], sec[:, 1], sec[:, 2], sec[:, 3]
            in_sec = (
                (gx[None] >= sx1[:, None]) & (gx[None] <= sx2[:, None]) &
                (gy[None] >= sy1[:, None]) & (gy[None] <= sy2[:, None])
            )
            weights[in_sec] = self.secondary_scale

        # Fovea takes priority over secondary
        weights[in_fovea] = self.fovea_scale

        return weights  # [B, G*G]

    # ------------------------------------------------------------------
    def forward(
        self,
        projected_patch_embeddings: torch.Tensor,    # [B, N_seq, D]
        fovea_bbox: torch.Tensor,                    # [B, 4] pixel-space xyxy
        secondary_bbox: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return modulated embeddings.  Text / action tokens and the separator are untouched.
        Only the original-image patch block (and optionally the trace-image block) is scaled.
        """
        wmap = self._make_weight_map(fovea_bbox, secondary_bbox)  # [B, num_patches]
        w = wmap.unsqueeze(-1)                                     # [B, num_patches, 1]

        out = projected_patch_embeddings.clone()

        # Original observation patches: positions 0 … num_patches-1
        N = self.num_patches
        out[:, :N, :] = projected_patch_embeddings[:, :N, :] * w

        # Optionally scale trace-image patches too: positions num_patches+1 … 2*num_patches
        if self.apply_to_trace_image:
            sep = N + 1  # skip separator at position N
            out[:, sep : sep + N, :] = projected_patch_embeddings[:, sep : sep + N, :] * w

        return out
