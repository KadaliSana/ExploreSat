"""
Segmentation model factory.

Wraps ``segmentation-models-pytorch`` to provide a simple interface
for creating U-Net / U-Net++ / DeepLabV3+ models suitable for
topographical feature extraction.

RTX 3060 (12 GB VRAM) guidance
--------------------------------
- ``encoder = "resnet34"``   → ~190 MB  (recommended default)
- ``encoder = "resnet50"``   → ~280 MB
- ``encoder = "resnet101"``  → ~450 MB  (High VRAM usage)
- ``encoder = "efficientnet-b3"`` → ~170 MB
- ``encoder = "xception"``   → ~210 MB
- ``encoder = "mit_b2"``     → ~250 MB  (Transformer-based)

All encoders are pre-trained on ImageNet (free weights).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except ImportError:
    _SMP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "unet": "Unet",
    "unet++": "UnetPlusPlus",
    "deeplabv3+": "DeepLabV3Plus",
    "segformer": "Segformer",
    "pspnet": "PSPNet",
    "linknet": "Linknet",
    "manet": "MAnet",
}

ENCODERS = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "xception": "xception",
    "mit_b0": "mit_b0",
    "mit_b2": "mit_b2",
    "mit_b5": "mit_b5",
    "efficientnet-b3": "efficientnet-b3",
    "efficientnet-b4": "efficientnet-b4",
    "mobilenet-v2": "mobilenet_v2",
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(
    architecture: str = "unet",
    encoder: str = "resnet34",
    in_channels: int = 3,
    num_classes: int = 6,
    encoder_weights: Optional[str] = "imagenet",
    activation: Optional[str] = None,
) -> nn.Module:
    """Create a segmentation model.

    Parameters
    ----------
    architecture:
        One of ``"unet"``, ``"unet++"``, ``"deeplabv3+"``, ``"segformer"``,
        ``"pspnet"``, ``"linknet"``, ``"manet"``.
    encoder:
        Backbone name, e.g. ``"resnet34"`` or ``"efficientnet-b3"``.
    in_channels:
        Number of input channels (3 for RGB, 4 for RGBI).
    num_classes:
        Number of output segmentation classes.
    encoder_weights:
        Pre-trained weight set (``"imagenet"`` or ``None``).
    activation:
        Final activation function.  ``None`` returns raw logits (use
        with ``CrossEntropyLoss``).  Use ``"softmax2d"`` for inference
        probability maps.

    Returns
    -------
    nn.Module
        The segmentation model.

    Raises
    ------
    ImportError
        If ``segmentation-models-pytorch`` is not installed.
    ValueError
        If an unsupported architecture is requested.
    """
    if not _SMP_AVAILABLE:
        raise ImportError(
            "segmentation-models-pytorch is required.\n"
            "Install with: pip install segmentation-models-pytorch"
        )

    arch_key = architecture.lower()
    if arch_key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'.  "
            f"Choose from: {list(ARCHITECTURES)}"
        )

    arch_class_name = ARCHITECTURES[arch_key]
    model_class = getattr(smp, arch_class_name)

    model = model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
    )
    return model


# ---------------------------------------------------------------------------
# Lightweight fallback U-Net (no smp dependency)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleUNet(nn.Module):
    """Minimal U-Net implementation with no external dependencies.

    Useful as a fallback when ``segmentation-models-pytorch`` is not
    available, or for very resource-constrained environments.

    Parameters
    ----------
    in_channels:
        Input channels (e.g. 3 for RGB).
    num_classes:
        Number of output segmentation classes.
    features:
        Number of feature maps at each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        features: tuple = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        prev = in_channels
        for feat in features:
            self.encoders.append(_DoubleConv(prev, feat))
            prev = feat

        self.bottleneck = _DoubleConv(prev, prev * 2)
        prev = prev * 2

        for feat in reversed(features):
            self.upsamples.append(
                nn.ConvTranspose2d(prev, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(_DoubleConv(feat * 2, feat))
            prev = feat

        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(
            self.upsamples, self.decoders, reversed(skip_connections)
        ):
            x = up(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


def build_simple_unet(in_channels: int = 3, num_classes: int = 6) -> SimpleUNet:
    """Convenience factory for the built-in lightweight U-Net."""
    return SimpleUNet(in_channels=in_channels, num_classes=num_classes)
