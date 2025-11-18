import numpy as np
import torch

from experiments.dataset.tyche_augs import (
    AugType,
    TycheAugs,
    _ensure_4d,
    _restore_shape,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_sharpness,
    apply_brightness_contrast,
    apply_tyche_aug,
    apply_tyche_augs,
)


def test_ensure_restore_shape_roundtrip():
    # 2D
    img_2d = torch.randn(16, 16)
    img4d, orig_shape = _ensure_4d(img_2d)
    restored = _restore_shape(img4d, orig_shape)
    assert restored.shape == img_2d.shape
    assert torch.allclose(restored, img_2d)

    # 3D (C, H, W)
    img_3d = torch.randn(3, 16, 16)
    img4d, orig_shape = _ensure_4d(img_3d)
    restored = _restore_shape(img4d, orig_shape)
    assert restored.shape == img_3d.shape
    assert torch.allclose(restored, img_3d)

    # 4D (N, C, H, W)
    img_4d = torch.randn(2, 3, 16, 16)
    img4d, orig_shape = _ensure_4d(img_4d)
    assert img4d is img_4d  # unchanged
    restored = _restore_shape(img4d, orig_shape)
    assert restored.shape == img_4d.shape
    assert torch.allclose(restored, img_4d)


def test_tycheaugs_sampling_reproducible_with_seed():
    t1 = TycheAugs(seed=123)
    t2 = TycheAugs(seed=123)

    augs1 = t1.sample_augs_with_params(N=5)
    augs2 = t2.sample_augs_with_params(N=5)

    # Same seed -> same aug types and parameters
    assert [a for a, _ in augs1] == [a for a, _ in augs2]
    for (_, p1), (_, p2) in zip(augs1, augs2):
        assert set(p1.keys()) == set(p2.keys())
        for k in p1:
            assert np.isclose(p1[k], p2[k])

    # Different seeds should very likely differ in at least one type or param
    t3 = TycheAugs(seed=124)
    augs3 = t3.sample_augs_with_params(N=5)
    different = False
    for (a1, p1), (a3, p3) in zip(augs1, augs3):
        if a1 != a3:
            different = True
            break
        # Same aug type: check any overlapping parameter differs
        shared_keys = set(p1.keys()) & set(p3.keys())
        if any(not np.isclose(p1[k], p3[k]) for k in shared_keys):
            different = True
            break
    assert different


def test_gaussian_blur_preserves_shape_and_changes_values():
    img = torch.rand(1, 32, 32)
    blurred = apply_gaussian_blur(img, sigma=0.5, kernel_size=5)
    assert blurred.shape == img.shape
    # With non-zero sigma and random input, blurred should differ
    assert not torch.allclose(blurred, img)


def test_gaussian_noise_preserves_shape_and_uses_torch_rng():
    img = torch.zeros(1, 8, 8)
    torch.manual_seed(0)
    noisy1 = apply_gaussian_noise(img, mu=0.0, sigma=0.1)
    torch.manual_seed(0)
    noisy2 = apply_gaussian_noise(img, mu=0.0, sigma=0.1)
    assert noisy1.shape == img.shape
    assert torch.allclose(noisy1, noisy2)


def test_brightness_contrast_and_sharpness_behaviour():
    img = torch.full((1, 4, 4), 0.5)

    # Brightness/contrast with known params
    bc = apply_brightness_contrast(img, brightness=0.1, contrast=2.0)
    expected_bc = torch.clamp(img * 2.0 + 0.1, 0.0, 1.0)
    assert torch.allclose(bc, expected_bc)

    # Sharpness with factor 0 -> identity
    sharp0 = apply_sharpness(img, factor=0.0)
    assert torch.allclose(sharp0, img)


def test_apply_tyche_aug_dispatch():
    img = torch.full((1, 8, 8), 0.5)

    out_bc = apply_tyche_aug(
        img,
        AugType.BRIGHTNESS_CONTRAST,
        {"brightness": 0.1, "contrast": 1.0},
    )
    expected_bc = apply_brightness_contrast(img, brightness=0.1, contrast=1.0)
    assert torch.allclose(out_bc, expected_bc)


def test_apply_tyche_augs_returns_independent_perturbations():
    img = torch.rand(1, 16, 16)
    tyche = TycheAugs(seed=123)
    augs_with_params = tyche.sample_augs_with_params(N=3)

    outputs = apply_tyche_augs(img, augs_with_params)

    # Should return one output per sampled augmentation, each same shape as input.
    assert isinstance(outputs, list)
    assert len(outputs) == len(augs_with_params)
    for out in outputs:
        assert isinstance(out, torch.Tensor)
        assert out.shape == img.shape

    # At least one output should differ from the original image for typical configs.
    assert any(not torch.allclose(out, img) for out in outputs)
