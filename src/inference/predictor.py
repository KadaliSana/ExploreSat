"""
Sliding-window inference on large GeoTIFF images with GeoTIFF / shapefile output.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import class_to_rgb_mask, CLASS_NAMES, NUM_CLASSES


class Predictor:
    """Run sliding-window segmentation on arbitrarily large images.

    Parameters
    ----------
    model:
        Trained segmentation model.
    tile_size:
        Inference tile size (should match training tile size).
    overlap:
        Pixel overlap between tiles (soft-blending at edges).
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    mean / std:
        Per-channel normalisation statistics used during training.
    """

    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 512,
        overlap: int = 64,
        device: str = "auto",
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """Run inference on an RGB image array."""
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap

        # Dynamically determine expected input channels from model's first layer
        try:
            expected_c = next(self.model.parameters()).shape[1]
            if image.shape[-1] < expected_c:
                pad_width = expected_c - image.shape[-1]
                image = np.pad(image, ((0,0), (0,0), (0, pad_width)), mode="constant", constant_values=0)
        except Exception:
            pass

        prob_map = None
        count_map = np.zeros((1, h, w), dtype=np.float32)

        # Pad image so that it is at least tile_size, and h,w are completely covered
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)
        
        if (h + pad_h - self.tile_size) % stride != 0:
            pad_h += stride - ((h + pad_h - self.tile_size) % stride)
        if (w + pad_w - self.tile_size) % stride != 0:
            pad_w += stride - ((w + pad_w - self.tile_size) % stride)

        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        ph, pw = padded.shape[:2]

        for y in range(0, ph - self.tile_size + 1, stride):
            for x in range(0, pw - self.tile_size + 1, stride):
                tile = padded[y:y + self.tile_size, x:x + self.tile_size]
                logits = self._infer_tile(tile)   # (C, tile_size, tile_size)
                
                if prob_map is None:
                    num_classes = logits.shape[0]
                    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
                
                probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()

                y2 = min(y + self.tile_size, h)
                x2 = min(x + self.tile_size, w)

                if y < h and x < w:
                    ty = y2 - y
                    tx = x2 - x
                    prob_map[:, y:y2, x:x2] += probs[:, :ty, :tx]
                    count_map[:, y:y2, x:x2] += 1.0

        count_map = np.clip(count_map, 1e-6, None)
        prob_map /= count_map
        return prob_map.argmax(axis=0).astype(np.uint8)

    # ------------------------------------------------------------------

    def predict_geotiff(
        self,
        src_path: str | Path,
        out_path: str | Path,
        export_rgb: bool = True,
        export_vector: bool = False,
    ) -> Path:
        """Predict over a GeoTIFF and save the result.

        Parameters
        ----------
        src_path:
            Path to the input GeoTIFF (RGB or multi-band).
        out_path:
            Path for the output class-index GeoTIFF.
        export_rgb:
            Also save a colour-coded RGB visualisation
            (``<out_path>_rgb.tif``).
        export_vector:
            Vectorise the prediction and save a GeoPackage
            (``<out_path>_polygons.gpkg``).

        Returns
        -------
        Path
            Path to the saved class-index GeoTIFF.
        """
        import rasterio
        from rasterio.transform import from_bounds

        src_path = Path(src_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(src_path) as src:
            meta = src.meta.copy()
            bands = src.count
            data = src.read(list(range(1, bands + 1)))
            data = np.transpose(data, (1, 2, 0))
            if data.dtype != np.uint8:
                lo, hi = data.min(), data.max()
                data = ((data - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
            if bands == 1:
                data = np.repeat(data, 3, axis=-1)

        print(f"Running inference on {src_path.name}  "
              f"({data.shape[1]} x {data.shape[0]} px) …")
        pred = self.predict_image(data)

        # Save class-index raster
        meta.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": 0})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(pred[np.newaxis, ...])
        print(f"  Class map saved: {out_path}")

        if export_rgb:
            rgb_path = out_path.with_name(out_path.stem + "_rgb.tif")
            rgb = class_to_rgb_mask(pred)
            rgb_meta = meta.copy()
            rgb_meta.update({"count": 3, "dtype": "uint8", "nodata": 0})
            with rasterio.open(rgb_path, "w", **rgb_meta) as dst:
                dst.write(np.transpose(rgb, (2, 0, 1)))
            print(f"  RGB visualisation saved: {rgb_path}")

        if export_vector:
            self._vectorise(pred, meta, out_path)

        return out_path

    # ------------------------------------------------------------------

    def _infer_tile(self, tile: np.ndarray) -> np.ndarray:
        """Normalise and run the model on a single tile."""
        img = tile.astype(np.float32) / 255.0
        
        # Dynamically pad mean/std if array has more channels (e.g., 4-channel RGBI)
        c = img.shape[-1]
        mean = np.pad(self.mean, (0, max(0, c - len(self.mean))), constant_values=0.0) if c > len(self.mean) else self.mean[:c]
        std = np.pad(self.std, (0, max(0, c - len(self.std))), constant_values=1.0) if c > len(self.std) else self.std[:c]
        
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.cuda.amp.autocast(
            enabled=self.device.type == "cuda"
        ):
            logits = self.model(tensor)
        return logits.squeeze(0).cpu().numpy()

    @staticmethod
    def _vectorise(pred: np.ndarray, meta: dict, base_path: Path) -> None:
        """Polygonise the raster prediction and save as GeoPackage."""
        try:
            import geopandas as gpd
            import rasterio.features
            from shapely.geometry import shape

            shapes = list(rasterio.features.shapes(
                pred.astype(np.int16),
                transform=meta["transform"],
            ))
            records = [
                {"geometry": shape(geom), "class_id": int(val),
                 "class_name": CLASS_NAMES[int(val)] if int(val) < len(CLASS_NAMES) else "unknown"}
                for geom, val in shapes
            ]
            gdf = gpd.GeoDataFrame(records, crs=meta.get("crs", "EPSG:4326"))
            vec_path = base_path.with_name(base_path.stem + "_polygons.gpkg")
            gdf.to_file(str(vec_path), driver="GPKG")
            print(f"  Vector polygons saved: {vec_path}")
        except ImportError:
            print("  Skipping vectorisation – geopandas not installed.")
