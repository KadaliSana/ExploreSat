"""
Dataset download helpers.

Free / non-commercial datasets supported
-----------------------------------------
1. **ISPRS Potsdam** (2D Semantic Labeling):
   https://www.isprs.org/education/benchmarks/UrbanClassification/
   Register (free) to get the download links, then use
   ``download_isprs_potsdam(dest_dir)``.

2. **ISPRS Vaihingen** (2D Semantic Labeling):
   Same portal as Potsdam.

3. **Sentinel-2** (multi-spectral satellite, ~10 m GSD):
   Freely available via the Copernicus Open Access Hub
   (https://scihub.copernicus.eu/).  The ``sentinelsat`` library is
   used.  You need a free account.

Note
----
ISPRS datasets require manual registration; this helper prints the
instructions and, if the user has already downloaded the archives,
will unzip and tile them automatically.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ISPRS helper
# ---------------------------------------------------------------------------

ISPRS_INSTRUCTIONS = """
=======================================================================
ISPRS Potsdam / Vaihingen – Manual Download Steps
=======================================================================
1. Go to: https://www.isprs.org/education/benchmarks/UrbanClassification/
2. Click on "Potsdam" or "Vaihingen" benchmark.
3. Fill in the free registration form.
4. You will receive download links for:
     - Top_Potsdam_<tile>_IRRG.tif   (Infrared-Red-Green image)
     - Top_Potsdam_<tile>_RGB.tif    (optional RGB image)
     - Top_Potsdam_<tile>_label.tif  (RGB colour-coded label)
5. Download all tiles into:  {dest_dir}
6. Re-run this script – it will tile and split the data automatically.
=======================================================================
"""


def prepare_isprs(archive_dir: str | Path, dest_dir: str | Path,
                  tile_size: int = 512, overlap: int = 64) -> None:
    """Tile ISPRS GeoTIFFs that the user has already downloaded.

    Parameters
    ----------
    archive_dir:
        Directory containing the raw ISPRS GeoTIFF / zip files.
    dest_dir:
        Destination directory.  ``images/`` and ``labels/`` sub-
        directories will be created.
    tile_size:
        Tile size in pixels.
    overlap:
        Overlap between adjacent tiles (stride = tile_size - overlap).
    """
    import numpy as np
    import cv2

    archive_dir = Path(archive_dir)
    dest_dir = Path(dest_dir)
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Unzip any archives first
    for zf in archive_dir.glob("*.zip"):
        print(f"Unzipping {zf.name} …")
        with zipfile.ZipFile(zf) as z:
            z.extractall(archive_dir)

    # Match images to labels by tile stem
    image_files = sorted(archive_dir.rglob("*RGB*.tif")) + \
                  sorted(archive_dir.rglob("*IRRG*.tif"))
    label_files = sorted(archive_dir.rglob("*label*.tif"))

    if not image_files:
        print(ISPRS_INSTRUCTIONS.format(dest_dir=archive_dir))
        return

    paired = _match_pairs(image_files, label_files)
    stride = tile_size - overlap
    tile_count = 0

    for img_path, lbl_path in paired.items():
        print(f"Tiling {img_path.name} …")
        try:
            image = _read_geotiff_as_rgb(img_path)
            label = _read_geotiff_rgb(lbl_path)
        except Exception as exc:
            print(f"  Skipping {img_path.name}: {exc}")
            continue

        h, w = image.shape[:2]
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                img_tile = image[y:y + tile_size, x:x + tile_size]
                lbl_tile = label[y:y + tile_size, x:x + tile_size]

                stem = f"{img_path.stem}_y{y:05d}_x{x:05d}"
                cv2.imwrite(
                    str(dest_dir / "images" / f"{stem}.png"),
                    cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(dest_dir / "labels" / f"{stem}.png"),
                    cv2.cvtColor(lbl_tile, cv2.COLOR_RGB2BGR),
                )
                tile_count += 1

    print(f"Done – {tile_count} tiles written to {dest_dir}")
    _write_splits(dest_dir, val_frac=0.15, test_frac=0.05)


# ---------------------------------------------------------------------------
# Sentinel-2 download helper
# ---------------------------------------------------------------------------

def download_sentinel2(
    username: str,
    password: str,
    dest_dir: str | Path,
    footprint_wkt: str,
    date_range: tuple = ("20230101", "20231231"),
    cloud_cover: int = 20,
    max_scenes: int = 5,
) -> None:
    """Download Sentinel-2 L2A scenes via ``sentinelsat``.

    Parameters
    ----------
    username / password:
        Copernicus Open Access Hub credentials (free registration at
        https://scihub.copernicus.eu/).
    dest_dir:
        Where to save downloaded zip files.
    footprint_wkt:
        Area of interest in WKT (WGS 84), e.g.
        ``"POLYGON((73 18, 78 18, 78 22, 73 22, 73 18))"``.
    date_range:
        ``(start, end)`` in YYYYMMDD format.
    cloud_cover:
        Maximum cloud-cover percentage (0–100).
    max_scenes:
        Maximum number of scenes to download.
    """
    try:
        from sentinelsat import SentinelAPI, make_wkt_footprint
    except ImportError as exc:
        raise ImportError(
            "sentinelsat is required.  Install with: pip install sentinelsat"
        ) from exc

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = SentinelAPI(username, password, "https://scihub.copernicus.eu/dhus")

    products = api.query(
        footprint_wkt,
        date=date_range,
        platformname="Sentinel-2",
        producttype="S2MSI2A",
        cloudcoverpercentage=(0, cloud_cover),
    )

    if not products:
        print("No Sentinel-2 scenes found for the given parameters.")
        return

    product_ids = list(products.keys())[:max_scenes]
    print(f"Found {len(products)} scenes – downloading {len(product_ids)} …")
    api.download_all(product_ids, directory_path=str(dest_dir))
    print(f"Download complete.  Files in: {dest_dir}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _match_pairs(
    images: list,
    labels: list,
) -> dict:
    """Match image files to label files by the tile number in their name."""
    import re

    label_map: dict = {}
    for lbl in labels:
        m = re.search(r"(\d+_\d+)", lbl.stem)
        key = m.group(1) if m else lbl.stem
        label_map[key] = lbl

    pairs: dict = {}
    for img in images:
        m = re.search(r"(\d+_\d+)", img.stem)
        key = m.group(1) if m else img.stem
        if key in label_map:
            pairs[img] = label_map[key]
    return pairs


def _read_geotiff_as_rgb(path: Path) -> "np.ndarray":
    """Read a GeoTIFF and return it as uint8 RGB (H×W×3)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        bands = min(src.count, 3)
        data = src.read(list(range(1, bands + 1))).astype(np.float32)
        # Percentile stretch per band
        stretched = np.zeros_like(data)
        for i in range(bands):
            lo, hi = np.percentile(data[i], [2, 98])
            stretched[i] = np.clip((data[i] - lo) / max(hi - lo, 1) * 255, 0, 255)
        rgb = np.transpose(stretched, (1, 2, 0)).astype(np.uint8)
        if bands == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        return rgb


def _read_geotiff_rgb(path: Path) -> "np.ndarray":
    """Read an RGB label GeoTIFF as uint8 (H×W×3)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
    return rgb


def _write_splits(dest_dir: Path, val_frac: float = 0.15,
                  test_frac: float = 0.05) -> None:
    """Write train/val/test split text files."""
    import random

    stems = [p.stem for p in sorted((dest_dir / "images").iterdir())]
    random.shuffle(stems)
    n = len(stems)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_stems = stems[:n_test]
    val_stems = stems[n_test:n_test + n_val]
    train_stems = stems[n_test + n_val:]

    for split, items in [("train", train_stems), ("val", val_stems),
                         ("test", test_stems)]:
        with open(dest_dir / f"split_{split}.txt", "w") as f:
            f.write("\n".join(items))
    print(f"Splits: train={len(train_stems)}, val={len(val_stems)}, "
          f"test={len(test_stems)}")
