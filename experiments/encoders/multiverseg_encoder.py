from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import sys

import einops
import torch
from torch import nn

# Ensure local deps are on sys.path for notebook/script usage without prior setup.
REPO_ROOT = Path(__file__).resolve().parents[2]
for dep in ("MultiverSeg", "UniverSeg"):
    dep_path = REPO_ROOT / dep
    if dep_path.exists() and str(dep_path) not in sys.path:
        sys.path.append(str(dep_path))

from multiverseg.models.network import CrossBlock
from universeg.validation import Kwargs, as_2tuple, size2t, validate_arguments_init
from universeg.nn.vmap import vmap

from .base import BaseEncoder


@validate_arguments_init
@dataclass(eq=False, repr=False)
class MultiverSegEncoder(BaseEncoder):
    """Encoder replica of MultiverSegNet (no decoder or output head)."""

    encoder_blocks: List[size2t]
    cross_relu: bool = True
    block_kws: Optional[Kwargs] = None
    in_channels: Tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.enc_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))

        block_kws = self.block_kws or {}
        if "cross_kws" not in block_kws:
            block_kws["cross_kws"] = {"nonlinearity": "LeakyReLU"}
        if not self.cross_relu:
            block_kws["cross_kws"]["nonlinearity"] = None

        in_ch = self.in_channels
        for cross_ch, conv_ch in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)

    def forward(
        self,
        image: torch.Tensor,
        support_images: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return a single embedding per image using global pooling over the final
        MultiverSeg encoder feature map.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        target, _support, _skips = self._encode(
            image=image,
            support_images=support_images,
            support_labels=support_labels,
        )
        # target: (B, 1, C, H, W) -> (B, C, H, W)
        feat = target.squeeze(1)
        # Global avg and max pooling to capture coarse appearance statistics
        gap = feat.mean(dim=(2, 3))
        gmp = feat.amax(dim=(2, 3))
        emb = torch.cat([gap, gmp], dim=-1)
        # L2 normalize for downstream clustering stability
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb

    def _encode(
        self,
        image: torch.Tensor,
        support_images: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Match MultiverSegNet input expectations (B 1 C H W).
        if len(image.shape) == 4:
            target = einops.rearrange(image, "B C H W -> B 1 C H W")
        else:
            target = image

        if support_images is None or support_images.shape[1] == 0:
            bs, _, _, h, w = target.shape
            support_images = 0.5 * torch.ones((bs, 1, 1, h, w), device=target.device)
            support_labels = 0.5 * torch.ones((bs, 1, 1, h, w), device=target.device)

        support = torch.cat([support_images, support_labels], dim=2)

        skip_connections: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            skip_connections.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        return target, support, skip_connections
