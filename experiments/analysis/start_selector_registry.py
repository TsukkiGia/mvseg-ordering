#!/usr/bin/env python3
"""Registry of start-image nomination criteria for embedding-based simulations."""

from __future__ import annotations

from typing import Callable

import numpy as np

SelectorFn = Callable[[np.ndarray, np.ndarray], int]

def _validate_inputs(image_ids: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_ids_arr = np.asarray(image_ids, dtype=int).reshape(-1)
    emb = np.asarray(embeddings, dtype=float)
    if image_ids_arr.size == 0:
        raise ValueError("image_ids must be non-empty.")
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D), got shape={emb.shape}.")
    if emb.shape[0] != image_ids_arr.size:
        raise ValueError(
            "image_ids and embeddings length mismatch: "
            f"len(image_ids)={image_ids_arr.size}, embeddings.shape[0]={emb.shape[0]}"
        )
    if np.unique(image_ids_arr).size != image_ids_arr.size:
        raise ValueError("image_ids must be unique for deterministic selection.")
    if not np.all(np.isfinite(emb)):
        raise ValueError("embeddings contain non-finite values.")
    return image_ids_arr, emb


def _arg_pick_with_tie_break(
    scores: np.ndarray,
    image_ids: np.ndarray,
    *,
    maximize: bool,
) -> int:
    vals = np.asarray(scores, dtype=float).reshape(-1)
    if vals.size != image_ids.size:
        raise ValueError("scores and image_ids length mismatch.")
    if maximize:
        best_val = float(np.max(vals))
        tied = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)
    else:
        best_val = float(np.min(vals))
        tied = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)
    tied_ids = np.asarray(image_ids, dtype=int)[tied]
    if tied_ids.size == 0:
        raise RuntimeError("Internal tie-break failure: no tied candidates found.")
    return int(np.min(tied_ids))


def closest_centroid(image_ids: np.ndarray, embeddings: np.ndarray) -> int:
    ids, emb = _validate_inputs(image_ids, embeddings)
    centroid = emb.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(emb - centroid, axis=1)
    return _arg_pick_with_tie_break(dists, ids, maximize=False)


def medoid(image_ids: np.ndarray, embeddings: np.ndarray) -> int:
    ids, emb = _validate_inputs(image_ids, embeddings)
    pairwise = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    mean_dist = pairwise.mean(axis=1)
    return _arg_pick_with_tie_break(mean_dist, ids, maximize=False)



SELECTOR_REGISTRY: dict[str, SelectorFn] = {
    "closest_centroid": closest_centroid,
    "medoid": medoid,
}


def list_selectors() -> list[str]:
    return sorted(SELECTOR_REGISTRY.keys())


def get_selector(name: str) -> SelectorFn:
    key = str(name).strip()
    if key not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector '{name}'. Available selectors: {list_selectors()}"
        )
    return SELECTOR_REGISTRY[key]
