import numpy as np
import pytest

from experiments.analysis.start_selector_registry import (
    closest_centroid,
    medoid,
)


def test_core_selectors_on_simple_line_embeddings():
    image_ids = np.array([10, 11, 12], dtype=int)
    embeddings = np.array([[0.0], [1.0], [3.0]], dtype=float)

    assert closest_centroid(image_ids, embeddings) == 11
    assert medoid(image_ids, embeddings) == 11


def test_tie_breaking_uses_smallest_image_id():
    image_ids = np.array([20, 10, 30], dtype=int)
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)

    assert closest_centroid(image_ids, embeddings) == 10
    assert medoid(image_ids, embeddings) == 10


def test_selector_rejects_duplicate_image_ids():
    image_ids = np.array([1, 1, 2], dtype=int)
    embeddings = np.array([[0.0], [1.0], [2.0]], dtype=float)
    with pytest.raises(ValueError, match="image_ids must be unique"):
        closest_centroid(image_ids, embeddings)
