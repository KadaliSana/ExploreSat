"""Tests for dataset utilities."""

from __future__ import annotations

import numpy as np
import pytest

from exploresat.data.dataset import (
    CLASS_NAMES,
    ISPRS_PALETTE,
    NUM_CLASSES,
    class_to_rgb_mask,
    rgb_mask_to_class,
)


def test_palette_length():
    assert len(ISPRS_PALETTE) == NUM_CLASSES
    assert len(CLASS_NAMES) == NUM_CLASSES


def test_rgb_mask_roundtrip():
    """Converting RGB → class → RGB should reproduce the original palette."""
    h, w = 64, 64
    # Build a synthetic label image with one class per row
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rows_per_class = h // NUM_CLASSES
    for cls_idx, colour in enumerate(ISPRS_PALETTE):
        y_start = cls_idx * rows_per_class
        y_end = y_start + rows_per_class
        rgb[y_start:y_end, :] = colour

    class_mask = rgb_mask_to_class(rgb)
    recovered_rgb = class_to_rgb_mask(class_mask)

    # The recovered image should match the original in the clean rows
    for cls_idx, colour in enumerate(ISPRS_PALETTE):
        y_start = cls_idx * rows_per_class
        y_end = y_start + rows_per_class
        assert np.all(recovered_rgb[y_start:y_end, :] == np.array(colour)), (
            f"Class {cls_idx} colour mismatch after roundtrip"
        )


def test_unknown_colour_maps_to_background():
    """Pixels with unknown colours should map to class 5 (background)."""
    rgb = np.array([[[123, 45, 67]]], dtype=np.uint8)  # Not in palette
    class_mask = rgb_mask_to_class(rgb)
    assert class_mask[0, 0] == 5


def test_class_to_rgb_shape():
    class_mask = np.zeros((32, 32), dtype=np.int64)
    rgb = class_to_rgb_mask(class_mask)
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
