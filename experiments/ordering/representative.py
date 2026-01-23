from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans

from .base import NonAdaptiveOrderingConfig
from experiments.encoders.clip import CLIPEncoder
from experiments.encoders.dinov2 import DinoV2Encoder
from experiments.encoders.medsam import MedSAMEncoder
from experiments.encoders.multiverseg_encoder import MultiverSegEncoder
from experiments.encoders.vit import ViTEncoder

def _l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

class RepresentativeConfig(NonAdaptiveOrderingConfig):
    """
    Cluster-based representative ordering using image embeddings.

    - Compute embeddings z_i = encoder(x_i) for all candidates.
    - Run k-means (k=num_clusters) on embeddings.
    - Order clusters by distance of cluster centroid to dataset centroid.
    - Within each cluster, append nearest-to-centroid then farthest-from-centroid.
    - Append any remaining images by increasing distance to dataset centroid.
    """

    def __init__(
        self,
        seed: int,
        encoder_cfg: Optional[dict[str, Any]] = None,
        num_clusters: int = 3,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(seed=seed, name=name)
        if not encoder_cfg:
            raise ValueError("RepresentativeConfig requires either an encoder_cfg.")
        self.encoder = None
        self.encoder_cfg = encoder_cfg or {}
        self.num_clusters = max(1, int(num_clusters))
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.permutation_indices = [0]  # single ordering

    def _ensure_encoder(self) -> Any:
        """Instantiate encoder lazily so the parent process doesn't touch CUDA before sharding."""
        if self.encoder is not None:
            return self.encoder

        enc_type = str(self.encoder_cfg.get("type", "multiverseg")).lower()
        if enc_type == "multiverseg":
            pooling = self.encoder_cfg.get("pooling", "gap_gmp")
            encoder = MultiverSegEncoder(pooling=pooling)
        elif enc_type == "clip":
            encoder = CLIPEncoder(
                model_name=self.encoder_cfg.get("model_name", "ViT-B-32"),
                pretrained=self.encoder_cfg.get("pretrained", "openai"),
            )
        elif enc_type == "vit":
            encoder = ViTEncoder(
                model_name=self.encoder_cfg.get("model_name", "vit_b_16"),
                pretrained=bool(self.encoder_cfg.get("pretrained", True)),
            )
        elif enc_type == "dinov2":
            encoder = DinoV2Encoder(
                model_name=self.encoder_cfg.get("model_name", "facebook/dinov2-base"),
                local_path=self.encoder_cfg.get("local_path"),
            )
        elif enc_type == "medsam":
            encoder = MedSAMEncoder(
                model_type=self.encoder_cfg.get("model_type", "vit_b"),
                checkpoint_path=self.encoder_cfg.get("checkpoint_path"),
                image_size=int(self.encoder_cfg.get("image_size", 1024)),
                pooling=self.encoder_cfg.get("pooling", "gap_gmp"),
            )
        else:
            raise ValueError(f"Unknown encoder type: {enc_type}")

        encoder = encoder.to(self.device)
        encoder.eval()
        self.encoder = encoder
        return encoder

    def _kmeans(self, data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """K-means via scikit-learn (k-means++ init, multiple inits)."""
        n, _ = data.shape
        k = min(int(k), n)
        if k <= 0:
            raise ValueError("k must be >= 1.")

        # Use an explicit integer for n_init for broad sklearn compatibility.
        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def _order_from_clusters(
        self,
        support_indices: list[int],
        embeddings: dict[int, np.ndarray],
        labels: np.ndarray,
        centroids: np.ndarray,
        dataset_centroid: np.ndarray,
    ) -> list[int]:
        """
        Given cluster assignments and centroids, return an ordering.

        Default strategy:
          - Order clusters by distance of centroid to dataset centroid.
          - Within each cluster, append nearest-to-centroid then farthest-from-centroid.
          - Append any remaining images by distance to dataset centroid.
        """
        k = int(centroids.shape[0])
        # Group indices by cluster
        cluster_to_indices: dict[int, list[int]] = {i: [] for i in range(k)}
        for pos, ci in enumerate(support_indices):
            cluster_to_indices[labels[pos]].append(ci)

        # Order clusters by distance of centroid to dataset centroid
        cluster_order = sorted(
            range(k),
            key=lambda j: float(np.linalg.norm(centroids[j] - dataset_centroid)),
        )

        ordering: list[int] = []
        used: set[int] = set()
        for j in cluster_order:
            members = cluster_to_indices.get(j, [])
            if not members:
                continue
            # Distances to cluster centroid
            member_dists = [(idx, float(np.linalg.norm(embeddings[idx] - centroids[j]))) for idx in members]
            near = min(member_dists, key=lambda x: x[1])[0]
            far = max(member_dists, key=lambda x: x[1])[0]
            for candidate in (near, far):
                if candidate not in used:
                    ordering.append(candidate)
                    used.add(candidate)

        # Append any remaining images by distance to dataset centroid
        remaining = [idx for idx in support_indices if idx not in used]
        if remaining:
            remaining.sort(key=lambda idx: float(np.linalg.norm(embeddings[idx] - dataset_centroid)))
            ordering.extend(remaining)
        return ordering

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        support_indices = list(candidate_indices)
        if not support_indices:
            return []

        # Compute embeddings
        encoder = self._ensure_encoder()
        embeddings: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for idx in support_indices:
                img, _ = support_dataset.get_item_by_data_index(idx)
                img = img.to(self.device)
                emb = encoder(img).detach().cpu().numpy()
                embeddings[idx] = _l2(np.asarray(emb.squeeze(0), dtype=emb.dtype))

        emb_matrix = np.stack([embeddings[i] for i in support_indices], axis=0)
        k = self.num_clusters

        labels, centroids = self._kmeans(emb_matrix, k)
        dataset_centroid = emb_matrix.mean(axis=0, keepdims=True)[0]

        ordering = self._order_from_clusters(
            support_indices=support_indices,
            embeddings=embeddings,
            labels=labels,
            centroids=centroids,
            dataset_centroid=dataset_centroid,
        )
        return [ordering]

    def get_ordering_labels(self):
        return self.permutation_indices

    def get_ordering_seeds(self):
        return [self.seed]
