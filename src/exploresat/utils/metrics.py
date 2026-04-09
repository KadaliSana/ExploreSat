"""
Segmentation metrics: IoU (Jaccard), Dice, pixel accuracy.

All functions accept PyTorch tensors (logits or class predictions)
and work on CPU or GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class-index tensor (B x H x W) to one-hot (B x C x H x W)."""
    b, h, w = labels.shape
    one_hot = torch.zeros(b, num_classes, h, w,
                          dtype=torch.float32, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-class Intersection-over-Union (Jaccard index).

    Parameters
    ----------
    preds:
        Predicted class indices (B x H x W) or logits (B x C x H x W).
    targets:
        Ground-truth class indices (B x H x W).
    num_classes:
        Total number of classes.
    ignore_index:
        Class index to exclude from the computation.
    eps:
        Small value for numerical stability.

    Returns
    -------
    torch.Tensor
        Per-class IoU scores, shape (num_classes,).
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]

    iou_per_class = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        pred_cls = preds == cls
        tgt_cls = targets == cls
        intersection = (pred_cls & tgt_cls).sum().float()
        union = (pred_cls | tgt_cls).sum().float()
        iou_per_class[cls] = (intersection + eps) / (union + eps)
    return iou_per_class


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """Return mean IoU across all classes (scalar float)."""
    scores = iou_score(preds, targets, num_classes, ignore_index)
    return scores.mean().item()


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-class Dice coefficient.

    Returns
    -------
    torch.Tensor
        Per-class Dice scores, shape (num_classes,).
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]

    dice_per_class = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()
        intersection = (pred_cls * tgt_cls).sum()
        dice_per_class[cls] = (2.0 * intersection + eps) / (
            pred_cls.sum() + tgt_cls.sum() + eps
        )
    return dice_per_class


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor,
                   ignore_index: int = -1) -> float:
    """Overall pixel accuracy (scalar float)."""
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    mask = targets != ignore_index
    correct = (preds[mask] == targets[mask]).sum().float()
    total = mask.sum().float()
    return (correct / total.clamp(min=1)).item()
