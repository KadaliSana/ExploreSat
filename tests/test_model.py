"""Tests for segmentation model factory and SimpleUNet."""

from __future__ import annotations

import pytest
import torch

from exploresat.models.segmentation import SimpleUNet, build_simple_unet


def test_simple_unet_output_shape():
    model = SimpleUNet(in_channels=3, num_classes=6)
    model.eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 256, 256), f"Unexpected shape: {out.shape}"


def test_simple_unet_factory():
    model = build_simple_unet(in_channels=3, num_classes=6)
    assert isinstance(model, SimpleUNet)


def test_simple_unet_4channel_input():
    """Model should accept 4-channel (RGBI) input."""
    model = SimpleUNet(in_channels=4, num_classes=6)
    model.eval()
    x = torch.zeros(1, 4, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 128, 128)


def test_build_model_smp():
    """build_model should return an smp model when smp is available."""
    pytest.importorskip("segmentation_models_pytorch",
                        reason="segmentation-models-pytorch not installed")
    from exploresat.models.segmentation import build_model
    model = build_model(architecture="unet", encoder="resnet18",
                        num_classes=6, encoder_weights=None)
    model.eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 256, 256)


def test_build_model_unknown_arch():
    pytest.importorskip("segmentation_models_pytorch")
    from exploresat.models.segmentation import build_model
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(architecture="nonexistent")
