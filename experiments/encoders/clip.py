from __future__ import annotations
from dataclasses import dataclass
import open_clip 
import torch
from typing import Optional

from .base import BaseEncoder
from .encoder_util import build_tensor_preprocess


@dataclass(eq=False, repr=False)
class CLIPEncoder(BaseEncoder):
    model_name: str = "ViT-B-32"
    pretrained: Optional[str] = "openai"

    def __post_init__(self) -> None:
        super().__init__()
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
            force_quick_gelu=True,
        )
        self.preprocess = build_tensor_preprocess(preprocess.transforms)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, support_images=None, support_labels=None) -> torch.Tensor:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        processed_image = self.preprocess(image)
        feats = self.model.encode_image(processed_image)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
