"""
Dataset loaders for topographical feature extraction.

Supported datasets (all free / non-commercial):
  - ISPRS Potsdam 2D Labeling Contest  (aerial, 5 cm/px)
  - ISPRS Vaihingen 2D Labeling Contest (aerial, 9 cm/px)
  - Generic tile dataset (any GeoTIFF image + label pairs)

Potsdam / Vaihingen label colours → class indices
-------------------------------------------------
  0  Impervious surfaces  (white  #FFFFFF)
  1  Buildings            (blue   #0000FF)
  2  Low vegetation       (cyan   #00FFFF)
  3  Trees                (green  #00FF00)
  4  Cars                 (yellow #FFFF00)
  5  Background / clutter (red    #FF0000)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Colour palette for ISPRS Potsdam / Vaihingen
# ---------------------------------------------------------------------------
ISPRS_PALETTE: List[Tuple[int, int, int]] = [
    (255, 255, 255),  # 0 – Impervious surfaces
    (0,   0,   255),  # 1 – Buildings
    (0,   255, 255),  # 2 – Low vegetation
    (0,   255,   0),  # 3 – Trees
    (255, 255,   0),  # 4 – Cars
    (255,   0,   0),  # 5 – Background / clutter
]

CLASS_NAMES = [
    "Impervious surfaces",
    "Buildings",
    "Low vegetation",
    "Trees",
    "Cars",
    "Background / clutter",
]

NUM_CLASSES = len(CLASS_NAMES)


def rgb_mask_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB label mask (H×W×3) to a class-index mask (H×W).

    Each pixel colour is mapped to its class index using the ISPRS
    palette.  Unknown colours are assigned to class 5 (background).
    """
    h, w = mask_rgb.shape[:2]
    class_mask = np.full((h, w), fill_value=5, dtype=np.int64)
    for idx, colour in enumerate(ISPRS_PALETTE):
        match = np.all(mask_rgb == colour, axis=-1)
        class_mask[match] = idx
    return class_mask


def class_to_rgb_mask(class_mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask (H×W) back to an RGB image (H×W×3)."""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, colour in enumerate(ISPRS_PALETTE):
        rgb[class_mask == idx] = colour
    return rgb


# ---------------------------------------------------------------------------
# Base tile dataset
# ---------------------------------------------------------------------------

class TopographyDataset(Dataset):
    """PyTorch dataset for paired image / label tiles.

    Expected directory layout::

        root/
          images/   *.tif  (or *.png / *.jpg)
          labels/   *.tif  (RGB colour-coded, same stem)

    Parameters
    ----------
    root:
        Path to the dataset root directory.
    split:
        ``"train"``, ``"val"``, or ``"test"``.  Used to locate an
        optional ``split_<split>.txt`` file that lists image stems.
        If no split file is found every image in ``images/`` is used.
    image_size:
        Height and width to which every tile is resized (square).
    transform:
        Optional *albumentations* transform applied to both the image
        and the label.
    mean / std:
        Per-channel normalisation statistics (default: ImageNet).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found in {root}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"labels/ directory not found in {root}")

        split_file = self.root / f"split_{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                stems = [line.strip() for line in f if line.strip()]
        else:
            # Use all images when no split file is provided
            stems = [
                p.stem
                for p in sorted(images_dir.iterdir())
                if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            ]

        self.samples: List[Tuple[Path, Path]] = []
        for stem in stems:
            img_candidates = list(images_dir.glob(f"{stem}.*"))
            lbl_candidates = list(labels_dir.glob(f"{stem}.*"))
            if img_candidates and lbl_candidates:
                self.samples.append((img_candidates[0], lbl_candidates[0]))

        if not self.samples:
            raise RuntimeError(
                f"No valid image/label pairs found in {root} for split='{split}'."
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        image = self._load_image(img_path)
        label = self._load_label(lbl_path)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        # Normalise image
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).long()

        return image, label

    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Load an image tile as uint8 RGB (H×W×3)."""
        # Try rasterio first so we honour geospatial metadata
        try:
            import rasterio
            with rasterio.open(path) as src:
                bands = min(src.count, 3)
                data = src.read(list(range(1, bands + 1)))  # (C, H, W)
                data = np.transpose(data, (1, 2, 0))        # (H, W, C)
                if data.dtype != np.uint8:
                    # Rescale to 0-255 per band
                    lo, hi = data.min(), data.max()
                    data = ((data - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
                if bands == 1:
                    data = np.repeat(data, 3, axis=-1)
                return data
        except Exception:
            pass
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot open image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_label(path: Path) -> np.ndarray:
        """Load a label tile as a class-index array (H×W)."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                if src.count >= 3:
                    rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                    return rgb_mask_to_class(rgb.astype(np.uint8))
                else:
                    # Single-channel class index label
                    return src.read(1).astype(np.int64)
        except Exception:
            pass
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot open label: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_mask_to_class(rgb)
