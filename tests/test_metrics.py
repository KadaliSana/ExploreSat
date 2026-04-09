"""Tests for segmentation metrics."""

from __future__ import annotations

import torch

from utils.metrics import (
    dice_score,
    iou_score,
    mean_iou,
    pixel_accuracy,
)

NUM_CLASSES = 6


def _make_perfect(n: int = 2, h: int = 4, w: int = 4):
    """Return identical pred / target tensors (perfect predictions)."""
    targets = torch.randint(0, NUM_CLASSES, (n, h, w))
    preds = torch.zeros(n, NUM_CLASSES, h, w)
    preds.scatter_(1, targets.unsqueeze(1), 1.0)  # one-hot logits
    return preds, targets


def test_perfect_iou():
    preds, targets = _make_perfect()
    scores = iou_score(preds, targets, NUM_CLASSES)
    # Every class present should have IoU == 1.0
    for cls in range(NUM_CLASSES):
        if (targets == cls).any():
            assert abs(scores[cls].item() - 1.0) < 1e-4, (
                f"Class {cls} IoU should be 1.0 for perfect predictions"
            )


def test_perfect_pixel_accuracy():
    preds, targets = _make_perfect()
    acc = pixel_accuracy(preds, targets)
    assert abs(acc - 1.0) < 1e-4


def test_mean_iou_range():
    preds, targets = _make_perfect()
    miou = mean_iou(preds, targets, NUM_CLASSES)
    assert 0.0 <= miou <= 1.0


def test_dice_perfect():
    preds, targets = _make_perfect()
    scores = dice_score(preds, targets, NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        if (targets == cls).any():
            assert abs(scores[cls].item() - 1.0) < 1e-4


def test_iou_all_wrong():
    n, h, w = 2, 4, 4
    targets = torch.zeros(n, h, w, dtype=torch.long)      # all class 0
    wrong = torch.ones(n, NUM_CLASSES, h, w) * -1e9
    wrong[:, 1, :, :] = 1.0                                # predict class 1
    scores = iou_score(wrong, targets, NUM_CLASSES)
    # Class 0 should have IoU == 0 (predicted nowhere)
    assert scores[0].item() < 0.01


def test_ignore_index():
    preds, targets = _make_perfect()
    targets[:, :, 0] = -1           # mark first column as ignored
    acc = pixel_accuracy(preds, targets, ignore_index=-1)
    assert 0.0 <= acc <= 1.0
