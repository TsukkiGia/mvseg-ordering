import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

class ViTEncoder(BaseEncoder):
    def __init__(self, model_name="vit_b_16", pretrained=True):
        super().__init__()
        self.weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = getattr(models, model_name)(weights=self.weights)
        self.model.heads = nn.Identity()
        self.model.eval()

        if self.weights is None:
            raise ValueError("Use pretrained=True for a stable representation baseline.")

        # Pull expected preprocessing from the weights 
        pre = self.weights.transforms()
        # torchvision ImageClassification has these attrs
        target_hw = tuple(pre.crop_size)  # typically (224, 224)
        mean = pre.mean
        std = pre.std

        self.target_hw = target_hw
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image, support_images=None, support_labels=None):
        x = image.float()
        x = x.repeat(1, 3, 1, 1)  # grayscale -> 3ch
        x = F.interpolate(x, size=self.target_hw, mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std

        feats = self.model.forward_features(x)  # (B, D) in torchvision ViT
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # good for clustering
        return feats
