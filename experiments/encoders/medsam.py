from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import BaseEncoder
from .encoder_util import convert_to_rgb_tensor


@dataclass(eq=False, repr=False)
class MedSAMEncoder(BaseEncoder):
    """MedSAM encoder wrapper (SAM image encoder with medical checkpoint)."""

    model_type: str = "vit_b"
    checkpoint_path = "/data/ddmg/mvseg-ordering/weights/medsam_vit_b.pth"
    pooling: str = "gap_gmp"

    def __post_init__(self) -> None:
        super().__init__()
        try:
            from segment_anything import sam_model_registry  # type: ignore
        except Exception as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "MedSAMEncoder requires the 'segment_anything' package. "
                "Install it (or ensure it is on PYTHONPATH) and provide a MedSAM checkpoint."
            ) from exc

        if not self.checkpoint_path:
            raise ValueError("MedSAMEncoder requires checkpoint_path to a MedSAM/SAM .pth file.")

        if self.model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model_type: {self.model_type}")

        # Ensure checkpoints saved on CUDA can load on CPU-only machines.
        if torch.cuda.is_available():
            self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        else:
            # Build without checkpoint then load with map_location.
            self.model = sam_model_registry[self.model_type](checkpoint=None)
            state = torch.load(self.checkpoint_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, support_images=None, support_labels=None) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = convert_to_rgb_tensor(image).to(torch.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        if not hasattr(self.model, "preprocess"):
            raise AttributeError("SAM model is missing preprocess().")
        processed = self.model.preprocess(image)
        feats = self.model.image_encoder(processed)

        mode = self.pooling.lower()
        if mode == "gap":
            emb = feats.mean(dim=(2, 3))
        elif mode == "gmp":
            emb = feats.amax(dim=(2, 3))
        elif mode == "gap_gmp":
            gap = feats.mean(dim=(2, 3))
            gmp = feats.amax(dim=(2, 3))
            emb = torch.cat([gap, gmp], dim=-1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
