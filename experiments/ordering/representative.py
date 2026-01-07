from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from .base import NonAdaptiveOrderingConfig
from experiments.encoders.clip import CLIPEncoder
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
        else:
            raise ValueError(f"Unknown encoder type: {enc_type}")

        encoder = encoder.to(self.device)
        encoder.eval()
        self.encoder = encoder
        return encoder

    def _kmeans(self, data: np.ndarray, k: int, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Lightweight k-means on CPU."""
        n, _ = data.shape
        k = min(int(k), n)
        rng = np.random.default_rng(self.seed)

        # Pick k distinct points as initial centroids
        init_idx = rng.choice(n, size=k, replace=False)
        centroids = data[init_idx]
        labels = np.zeros(n, dtype=int)

        for _ in range(iters):
            # Assign each point to the nearest centroid
            dists = np.empty((n, k), dtype=data.dtype)
            for i in range(n):
                for j in range(k):
                    diff = data[i] - centroids[j]      # (D,)
                    dists[i, j] = (diff * diff).sum()  

            labels = dists.argmin(axis=1)
            # Recompute centroids as the mean of assigned points
            new_centroids = centroids.copy()
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = data[mask].mean(axis=0)
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
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
                embeddings[idx] = np.asarray(emb.squeeze(0), dtype=emb.dtype)

        emb_matrix = np.stack([embeddings[i] for i in support_indices], axis=0)
        k = self.num_clusters

        labels, centroids = self._kmeans(emb_matrix, k)
        dataset_centroid = _l2(emb_matrix.mean(axis=0, keepdims=True))[0]
        centroids = _l2(centroids)

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
