from __future__ import annotations

from typing import Any, Optional

import torch

from .clip import CLIPEncoder
from .dinov2 import DinoV2Encoder
from .medsam import MedSAMEncoder
from .multiverseg_encoder import MultiverSegEncoder
from .vit import ViTEncoder


def build_encoder_from_cfg(
    encoder_cfg: dict[str, Any],
    *,
    device: Optional[str | torch.device] = None,
) -> torch.nn.Module:
    enc_type = str(encoder_cfg.get("type", "multiverseg")).lower()
    if enc_type == "multiverseg":
        pooling = encoder_cfg.get("pooling", "gap_gmp")
        encoder = MultiverSegEncoder(pooling=pooling)
    elif enc_type == "clip":
        encoder = CLIPEncoder(
            model_name=encoder_cfg.get("model_name", "ViT-B-32"),
            pretrained=encoder_cfg.get("pretrained", "openai"),
        )
    elif enc_type == "vit":
        encoder = ViTEncoder(
            model_name=encoder_cfg.get("model_name", "vit_b_16"),
            pretrained=bool(encoder_cfg.get("pretrained", True)),
        )
    elif enc_type == "dinov2":
        encoder = DinoV2Encoder(
            model_name=encoder_cfg.get("model_name", "facebook/dinov2-base"),
            local_path=encoder_cfg.get("local_path"),
        )
    elif enc_type == "medsam":
        encoder = MedSAMEncoder(
            model_type=encoder_cfg.get("model_type", "vit_b"),
            pooling=encoder_cfg.get("pooling", "gap_gmp"),
        )
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")

    if device is not None:
        encoder = encoder.to(torch.device(device))
    encoder.eval()
    return encoder
