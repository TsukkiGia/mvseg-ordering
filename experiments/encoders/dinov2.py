from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, Dinov2Model

from .base import BaseEncoder
from .encoder_util import convert_to_rgb_tensor


@dataclass(eq=False, repr=False)
class DinoV2Encoder(BaseEncoder):
    """DINOv2 image encoder (returns L2-normalized CLS embedding)."""

    model_name: str = "facebook/dinov2-base"
    local_path: Optional[str] = None

    def __post_init__(self) -> None:
        super().__init__()
        model_id = self.local_path or self.model_name
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = Dinov2Model.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, support_images=None, support_labels=None) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (B,C,H,W)

        image = convert_to_rgb_tensor(image)

        # WARNING: to_pil_image assumes either [0,1] float or uint8 [0,255]
        pil_images = [to_pil_image(img.cpu()) for img in image]

        inputs = self.processor(images=pil_images, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state[:, 0, :]  # CLS token

        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
