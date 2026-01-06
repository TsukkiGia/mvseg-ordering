import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from .base import BaseEncoder
from .encoder_util import convert_to_rgb_tensor

class ViTEncoder(BaseEncoder):
    def __init__(self, model_name="vit_b_16", pretrained=True):
        super().__init__()
        self.weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        if self.weights is None:
            raise ValueError("Use pretrained=True for a stable representation baseline.")
        self.model = getattr(models, model_name)(weights=self.weights)
        self.model.heads = nn.Identity()
        self.model.eval()
        # Pull expected preprocessing from the weights 
        self.preprocess = self.weights.transforms()

    @torch.no_grad()
    def forward(self, image, support_images=None, support_labels=None):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = convert_to_rgb_tensor(image)
        processed_image = self.preprocess(image)
        feats = self.model(processed_image)          # heads=Identity -> (1,768)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats
