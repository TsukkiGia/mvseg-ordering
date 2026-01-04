from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseEncoder

def _load_open_clip(model_name: str, pretrained: Optional[str]):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    return model, preprocess

def _extract_clip_norm_and_size(preprocess):
    """
    Pull resize size + normalize mean/std out of the torchvision transform pipeline
    so we don't hardcode anything.
    """
    size = None
    mean = None
    std = None
    for t in getattr(preprocess, "transforms", []):
        name = t.__class__.__name__.lower()
        if "resize" in name and hasattr(t, "size"):
            # size can be int or (h,w)
            size = t.size
        if "normalize" in name and hasattr(t, "mean") and hasattr(t, "std"):
            mean = t.mean
            std = t.std
    # normalize to (H,W)
    if isinstance(size, int):
        size = (size, size)
    return size, mean, std

@dataclass(eq=False, repr=False)
class CLIPEncoder(BaseEncoder):
    model_name: str = "ViT-B-32"
    pretrained: Optional[str] = "openai"
    in_channels: int = 1              
    normalize_output: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        self.model, self.preprocess = _load_open_clip(self.model_name, self.pretrained)
        self.model.eval()

        # derive expected size + mean/std from preprocess (no magic numbers)
        size, mean, std = _extract_clip_norm_and_size(self.preprocess)
        if size is None or mean is None or std is None:
            raise RuntimeError("Could not infer CLIP preprocess (size/mean/std).")

        self.target_hw = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))

        # deterministic grayscale -> 3ch
        if self.in_channels not in (1, 3):
            raise ValueError("For CLIP, use in_channels=1 or 3 (or define a deterministic mapping).")

    @torch.no_grad()
    def forward(self, image: torch.Tensor, support_images=None, support_labels=None) -> torch.Tensor:
        x = image
        if x.dtype != torch.float32 and x.dtype != torch.float16 and x.dtype != torch.bfloat16:
            x = x.float()

        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)  # (B,3,H,W)

        # resize to CLIP expected resolution
        x = F.interpolate(x, size=self.target_hw, mode="bicubic", align_corners=False)

        # CLIP normalize
        x = (x - self.mean) / self.std

        feats = self.model.encode_image(x)

        if self.normalize_output:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats
