from __future__ import annotations

import torch


def pairwise_dice_disagreement(
    samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute pairwise Dice disagreement between K Monte Carlo samples.

    Args:
        samples:
            Tensor of shape (K, ...) containing K segmentation predictions
            (probabilities or binary masks). All non-leading dimensions are
            treated as spatial / channel dimensions.
        eps:
            Small constant to avoid division by zero.

    Returns:
        Scalar tensor: 1 - mean pairwise Dice across all unordered pairs.
    """
    if samples.ndim < 1:
        raise ValueError(f"samples must have at least 1 dimension, got shape {tuple(samples.shape)}")
    if samples.size(0) < 2:
        raise ValueError("pairwise_dice_disagreement requires at least K=2 samples.")

    k = samples.size(0)
    flat = samples.float().reshape(k, -1)

    # Intuitive nested-loop implementation over unordered pairs (i < j).
    dice_values = []
    for i in range(k):
        a = flat[i]
        for j in range(i + 1, k):
            b = flat[j]
            intersection = torch.sum(a * b)
            union = torch.sum(a) + torch.sum(b)
            d = (2.0 * intersection + eps) / (union + eps)
            dice_values.append(d)

    mean_dice = torch.stack(dice_values).mean()
    return 1.0 - mean_dice


def binary_entropy_from_mc_probs(
    probs: torch.Tensor,
    eps: float = 1e-8,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Compute binary entropy from K Monte Carlo probabilities for a binary label.

    This implements the standard MC-predictive entropy:

        H_hat[y | x, D_train] = - sum_c p_hat(c) log p_hat(c),
        with c in {0, 1},
        p_hat(c) = (1 / K) sum_t p_t(c | x, w_t).

    Here we assume `probs` contains sigmoid outputs giving p_t(y=1 | x, w_t)
    for each Monte Carlo sample t (typically the segmentation / foreground
    probability). The complementary class probability is then 1 - p.
    """
    if probs.ndim < 1:
        raise ValueError(f"probs must have at least 1 dimension, got shape {tuple(probs.shape)}")
    if probs.size(0) < 1:
        raise ValueError("probs must contain at least one MC sample (K >= 1).")

    # Average over MC samples along the leading dimension.
    p = probs.float().mean(dim=0)
    # Ensure strictly inside (0, 1) for stable log.
    p = p.clamp(min=eps, max=1.0 - eps)
    q = 1.0 - p

    entropy = -(p * torch.log(p) + q * torch.log(q))

    if reduce:
        return entropy.mean()
    return entropy
