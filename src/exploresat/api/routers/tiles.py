"""
XYZ map-tile router – serve raster GeoTIFFs as standard Web Map Tiles.

Tiles are served at:
    GET /tiles/{layer}/{z}/{x}/{y}.png

where ``layer`` is the stem of a GeoTIFF file inside ``data/predictions/``
or ``data/raw/``.

QGIS connection
---------------
In QGIS → Layer → Add Layer → Add XYZ Tile Layer:

    URL:  http://localhost:8000/tiles/{layer}/{z}/{x}/{y}.png

This lets QGIS display both the raw satellite imagery **and** the
colour-coded segmentation overlays side-by-side in the same project.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/tiles", tags=["Map Tiles (XYZ)"])

# Directories that contain serveable rasters
_SEARCH_DIRS = [
    Path("data/predictions"),
    Path("data/raw/sentinel2"),
    Path("data/raw/landsat"),
    Path("data/raw/dem"),
    Path("data/raw/naip"),
    Path("data/raw"),
]

# Colour palette for class-index rasters (ISPRS Potsdam / Vaihingen)
_CLASS_COLOURS = np.array([
    [255, 255, 255],   # 0 – Impervious surfaces
    [0,   0,   255],   # 1 – Buildings
    [0,   255, 255],   # 2 – Low vegetation
    [0,   255,   0],   # 3 – Trees
    [255, 255,   0],   # 4 – Cars
    [255,   0,   0],   # 5 – Background / clutter
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/layers", summary="List serveable raster layers")
async def list_layers() -> dict:
    """Return the names of all GeoTIFFs that can be served as XYZ tiles."""
    layers = []
    for d in _SEARCH_DIRS:
        if d.exists():
            layers.extend(p.stem for p in d.glob("*.tif"))
    return {"layers": sorted(set(layers))}


@router.get("/{layer}/{z}/{x}/{y}.png",
            response_class=Response,
            summary="Fetch a 256x256 PNG map tile")
async def get_tile(layer: str, z: int, x: int, y: int) -> Response:
    """Serve a standard 256x256 XYZ PNG tile for the requested layer.

    Parameters
    ----------
    layer:
        Stem of the GeoTIFF file (without ``.tif``).
    z / x / y:
        Standard Web Mercator tile coordinates.
    """
    tif_path = _find_layer(layer)
    if tif_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Layer '{layer}' not found. "
                   f"Call GET /tiles/layers to see available layers.",
        )

    try:
        png_bytes = _render_tile(tif_path, z, x, y)
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Tile render error: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_layer(layer: str) -> Optional[Path]:
    """Locate the GeoTIFF file for the given layer stem."""
    for d in _SEARCH_DIRS:
        candidate = d / f"{layer}.tif"
        if candidate.exists():
            return candidate
    return None


def _render_tile(tif_path: Path, z: int, x: int, y: int) -> bytes:
    """Render a 256x256 PNG tile using rio-tiler.

    Falls back to a manual crop/warp when rio-tiler is unavailable.
    """
    try:
        return _render_tile_riotiler(tif_path, z, x, y)
    except ImportError:
        return _render_tile_rasterio(tif_path, z, x, y)


def _render_tile_riotiler(tif_path: Path, z: int, x: int, y: int) -> bytes:
    """Render via rio-tiler (preferred – fast COG-aware tile server)."""
    from PIL import Image
    from rio_tiler.io import COGReader

    with COGReader(str(tif_path)) as cog:
        img = cog.tile(x, y, z, tilesize=256)

    # img.data shape: (bands, 256, 256)
    data = img.data
    mask = img.mask  # 0 = nodata, 255 = valid

    if data.shape[0] == 1:
        # Single-band: treat as class-index map and colourise
        rgb = _colourise_class_map(data[0])
    elif data.shape[0] >= 3:
        # Multi-band: use first three bands as RGB, stretch to uint8
        rgb = _stretch_to_uint8(data[:3])
    else:
        rgb = np.stack([data[0]] * 3, axis=-1)
        rgb = _stretch_to_uint8(rgb.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Apply nodata mask as alpha
    alpha = mask
    rgba = np.dstack([rgb, alpha])

    pil_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _render_tile_rasterio(tif_path: Path, z: int, x: int, y: int) -> bytes:
    """Minimal fallback tile renderer using rasterio + mercantile."""
    import rasterio
    import rasterio.warp
    from PIL import Image

    try:
        import mercantile
        bounds = mercantile.bounds(x, y, z)
        west, south, east, north = bounds
    except ImportError:
        raise ImportError(
            "Install either 'rio-tiler' or 'mercantile' for tile serving."
        )

    tile_size = 256
    with rasterio.open(tif_path) as src:
        dst_transform = rasterio.transform.from_bounds(
            west, south, east, north, tile_size, tile_size
        )
        bands = min(src.count, 3)
        out_shape = (bands, tile_size, tile_size)
        data = np.zeros(out_shape, dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, list(range(1, bands + 1))),
            destination=data,
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            resampling=rasterio.enums.Resampling.nearest,
        )

    if bands == 1:
        rgb = _colourise_class_map(data[0].astype(np.uint8))
    else:
        rgb = _stretch_to_uint8(data)

    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _colourise_class_map(class_map: np.ndarray) -> np.ndarray:
    """Map a (H, W) class-index array to (H, W, 3) uint8 RGB."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(_CLASS_COLOURS)
    for cls in range(n):
        mask = class_map == cls
        rgb[mask] = _CLASS_COLOURS[cls]
    return rgb


def _stretch_to_uint8(data: np.ndarray) -> np.ndarray:
    """Percentile-stretch a (C, H, W) float array to uint8 (H, W, 3)."""
    out = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    for i in range(min(data.shape[0], 3)):
        band = data[i].astype(np.float32)
        lo, hi = np.percentile(band[band > 0], [2, 98]) if band.any() else (0, 1)
        stretched = np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        out[:, :, i] = stretched.astype(np.uint8)
    return out
