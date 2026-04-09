"""
Dataset acquisition from free cloud platforms.

Two backends are supported – use whichever is most convenient:

Google Earth Engine (GEE)
--------------------------
* Free tier available (sign up at https://earthengine.google.com/).
* Requires a Google account and a Cloud project (non-commercial use is free).
* Install: ``pip install earthengine-api``
* Authenticate once with: ``earthengine authenticate``
* Datasets available: Sentinel-2, Landsat 8/9, SRTM DEM, and thousands more.
* Images are exported to Google Drive or Cloud Storage, then downloaded.

Microsoft Planetary Computer (MPC)
------------------------------------
* Completely free, no account required for public datasets.
* Install: ``pip install planetary-computer pystac-client stackstac``
* Datasets available: Sentinel-2 L2A, Landsat Collection 2, NAIP,
  Copernicus DEM, ASTER, and many more.
* Data is streamed as cloud-optimised GeoTIFFs (no login needed).

Quick-start
-----------
>>> # Option A – Planetary Computer (no sign-in needed)
>>> from exploresat.data.download import PlanetaryComputerDownloader
>>> dl = PlanetaryComputerDownloader(dest_dir="data/raw")
>>> dl.download_sentinel2(
...     bbox=(77.5, 28.5, 77.8, 28.8),   # lon_min, lat_min, lon_max, lat_max
...     date_range=("2023-01-01", "2023-12-31"),
...     max_cloud_cover=20,
...     max_items=3,
... )

>>> # Option B – Google Earth Engine
>>> from exploresat.data.download import GEEDownloader
>>> dl = GEEDownloader(project="my-gee-project", dest_dir="data/raw")
>>> dl.download_sentinel2(
...     bbox=(77.5, 28.5, 77.8, 28.8),
...     date_range=("2023-01-01", "2023-12-31"),
...     max_cloud_cover=20,
... )
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BBox = Tuple[float, float, float, float]   # (lon_min, lat_min, lon_max, lat_max)
DateRange = Tuple[str, str]                 # ("YYYY-MM-DD", "YYYY-MM-DD")


# ===========================================================================
# Microsoft Planetary Computer – no account required
# ===========================================================================

class PlanetaryComputerDownloader:
    """Download satellite imagery from Microsoft Planetary Computer (free).

    No account or sign-in is required for any of the public datasets.

    Parameters
    ----------
    dest_dir:
        Root directory where downloaded GeoTIFFs will be saved.
        Sub-directories are created automatically per dataset.
    """

    CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, dest_dir: str | Path = "data/raw") -> None:
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._check_deps()

    # ------------------------------------------------------------------
    # Sentinel-2
    # ------------------------------------------------------------------

    def download_sentinel2(
        self,
        bbox: BBox,
        date_range: DateRange = ("2023-01-01", "2023-12-31"),
        max_cloud_cover: float = 20.0,
        max_items: int = 5,
        bands: List[str] = ("B04", "B03", "B02", "B08"),   # R, G, B, NIR
        resolution: int = 10,
        output_subdir: str = "sentinel2",
    ) -> List[Path]:
        """Search and download Sentinel-2 L2A imagery.

        Parameters
        ----------
        bbox:
            Area of interest as ``(lon_min, lat_min, lon_max, lat_max)``
            in WGS-84 (EPSG:4326).
        date_range:
            ``("YYYY-MM-DD", "YYYY-MM-DD")`` inclusive date window.
        max_cloud_cover:
            Maximum scene cloud-cover percentage (0-100).
        max_items:
            Maximum number of scenes to retrieve.
        bands:
            Sentinel-2 band names to stack.  Defaults to R-G-B-NIR
            (10 m resolution bands).
        resolution:
            Output pixel size in metres.
        output_subdir:
            Sub-directory inside ``dest_dir`` for the saved files.

        Returns
        -------
        List[Path]
            Paths to the saved GeoTIFF files (one per scene).
        """
        import planetary_computer
        import pystac_client
        import stackstac

        print(f"[Planetary Computer] Searching Sentinel-2 L2A …")
        catalog = pystac_client.Client.open(
            self.CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="/".join(date_range),
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=max_items,
            sortby="-properties.datetime",
        )

        items = list(search.items())
        if not items:
            print("  No scenes found – try widening bbox, dates, or cloud threshold.")
            return []

        print(f"  Found {len(items)} scene(s). Downloading …")
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for item in items:
            path = self._save_sentinel2_item(
                item, bands, bbox, resolution, outdir
            )
            if path:
                saved.append(path)

        print(f"  Saved {len(saved)} file(s) to {outdir}")
        return saved

    # ------------------------------------------------------------------
    # Landsat 8 / 9 (Collection 2, Level-2)
    # ------------------------------------------------------------------

    def download_landsat(
        self,
        bbox: BBox,
        date_range: DateRange = ("2023-01-01", "2023-12-31"),
        max_cloud_cover: float = 20.0,
        max_items: int = 5,
        bands: List[str] = ("red", "green", "blue", "nir08"),
        resolution: int = 30,
        output_subdir: str = "landsat",
    ) -> List[Path]:
        """Search and download Landsat 8/9 Collection 2 Level-2 imagery.

        Parameters
        ----------
        bbox:
            Area of interest (lon_min, lat_min, lon_max, lat_max).
        date_range:
            Inclusive date window ("YYYY-MM-DD", "YYYY-MM-DD").
        max_cloud_cover:
            Maximum cloud-cover percentage.
        max_items:
            Maximum number of scenes to retrieve.
        bands:
            Band names from the STAC asset keys.
        resolution:
            Output pixel size in metres.
        output_subdir:
            Sub-directory name inside ``dest_dir``.

        Returns
        -------
        List[Path]
            Paths to saved GeoTIFF files.
        """
        import planetary_computer
        import pystac_client
        import stackstac

        print(f"[Planetary Computer] Searching Landsat Collection-2 …")
        catalog = pystac_client.Client.open(
            self.CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="/".join(date_range),
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=max_items,
            sortby="-properties.datetime",
        )

        items = list(search.items())
        if not items:
            print("  No scenes found.")
            return []

        print(f"  Found {len(items)} scene(s). Downloading …")
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for item in items:
            path = self._save_stac_item(
                item, bands, bbox, resolution, outdir,
                prefix=item.id,
            )
            if path:
                saved.append(path)

        print(f"  Saved {len(saved)} file(s) to {outdir}")
        return saved

    # ------------------------------------------------------------------
    # Copernicus DEM (30 m global elevation)
    # ------------------------------------------------------------------

    def download_dem(
        self,
        bbox: BBox,
        resolution: int = 30,
        output_subdir: str = "dem",
    ) -> Optional[Path]:
        """Download a Copernicus DEM tile for the given bounding box.

        Returns
        -------
        Path or None
            Path to the saved GeoTIFF, or ``None`` on failure.
        """
        import planetary_computer
        import pystac_client
        import stackstac

        print("[Planetary Computer] Downloading Copernicus DEM …")
        catalog = pystac_client.Client.open(
            self.CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox,
        )
        items = list(search.items())
        if not items:
            print("  No DEM tiles found for this bbox.")
            return None

        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        stack = stackstac.stack(
            items,
            assets=["data"],
            bounds=bbox,
            epsg=4326,
            resolution=resolution / 111_320,   # degrees per metre (approx)
            dtype="float32",
        )
        dem_np = stack.squeeze().values

        out_path = outdir / "copernicus_dem.tif"
        _save_geotiff(dem_np[np.newaxis, ...], bbox, str(out_path),
                      crs="EPSG:4326")
        print(f"  DEM saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # NAIP (USA only – 1 m aerial, free)
    # ------------------------------------------------------------------

    def download_naip(
        self,
        bbox: BBox,
        year: int = 2020,
        output_subdir: str = "naip",
    ) -> List[Path]:
        """Download NAIP aerial imagery (USA only, ~1 m, RGB+NIR).

        Parameters
        ----------
        bbox:
            Area of interest (lon_min, lat_min, lon_max, lat_max).
        year:
            Survey year (NAIP is acquired every 2–3 years per state).
        output_subdir:
            Sub-directory name inside ``dest_dir``.

        Returns
        -------
        List[Path]
            Paths to saved GeoTIFF files.
        """
        import planetary_computer
        import pystac_client

        print(f"[Planetary Computer] Searching NAIP {year} …")
        catalog = pystac_client.Client.open(
            self.CATALOG_URL,
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["naip"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            max_items=10,
        )
        items = list(search.items())
        if not items:
            print("  No NAIP tiles found.  NAIP only covers the USA.")
            return []

        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for item in items:
            signed = planetary_computer.sign(item)
            href = signed.assets["image"].href
            out_path = outdir / f"{item.id}.tif"
            print(f"  Downloading {item.id} …")
            _stream_cog_to_file(href, str(out_path), bbox)
            saved.append(out_path)

        print(f"  Saved {len(saved)} file(s) to {outdir}")
        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_sentinel2_item(
        self,
        item,
        bands: List[str],
        bbox: BBox,
        resolution: int,
        outdir: Path,
    ) -> Optional[Path]:
        """Crop and stack Sentinel-2 bands, save as multi-band GeoTIFF."""
        try:
            import planetary_computer
            import stackstac

            signed = planetary_computer.sign(item)
            stack = stackstac.stack(
                [signed],
                assets=list(bands),
                bounds=bbox,
                epsg=4326,
                resolution=resolution / 111_320,
                dtype="float32",
                fill_value=0,
            )
            # shape: (time=1, band, H, W) – squeeze time
            arr = stack.squeeze(dim="time").values  # (band, H, W)

            # Scale reflectance: Sentinel-2 L2A values are 0–10000
            arr = np.clip(arr / 10000.0, 0, 1).astype(np.float32)

            out_path = outdir / f"{item.id}.tif"
            _save_geotiff(arr, bbox, str(out_path), crs="EPSG:4326",
                          band_names=list(bands))
            return out_path
        except Exception as exc:
            print(f"  Warning: could not save {item.id}: {exc}")
            return None

    def _save_stac_item(
        self,
        item,
        bands: List[str],
        bbox: BBox,
        resolution: int,
        outdir: Path,
        prefix: str = "scene",
    ) -> Optional[Path]:
        """Generic STAC item saver using stackstac."""
        try:
            import planetary_computer
            import stackstac

            signed = planetary_computer.sign(item)
            stack = stackstac.stack(
                [signed],
                assets=list(bands),
                bounds=bbox,
                epsg=4326,
                resolution=resolution / 111_320,
                dtype="float32",
                fill_value=0,
            )
            arr = stack.squeeze(dim="time").values

            out_path = outdir / f"{prefix}.tif"
            _save_geotiff(arr, bbox, str(out_path), crs="EPSG:4326",
                          band_names=list(bands))
            return out_path
        except Exception as exc:
            print(f"  Warning: could not save {prefix}: {exc}")
            return None

    @staticmethod
    def _check_deps() -> None:
        missing = []
        for pkg in ("planetary_computer", "pystac_client", "stackstac"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg.replace("_", "-"))
        if missing:
            raise ImportError(
                "Missing packages for Planetary Computer downloader.\n"
                f"Install with: pip install {' '.join(missing)}"
            )


# ===========================================================================
# Google Earth Engine
# ===========================================================================

class GEEDownloader:
    """Download satellite imagery via Google Earth Engine (GEE).

    Free non-commercial use is available with a Google account.

    Setup (one-time)
    ----------------
    1. Sign up at https://earthengine.google.com/
    2. Create (or select) a Google Cloud project at
       https://console.cloud.google.com/
    3. Enable the Earth Engine API for your project.
    4. Run ``earthengine authenticate`` in a terminal and follow the
       browser prompts.

    Parameters
    ----------
    project:
        Your Google Cloud project ID (required since 2024).
    dest_dir:
        Root directory where exported GeoTIFFs will be saved.
    """

    def __init__(
        self,
        project: str,
        dest_dir: str | Path = "data/raw",
    ) -> None:
        self.project = project
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._ee = self._init_ee(project)

    # ------------------------------------------------------------------
    # Sentinel-2
    # ------------------------------------------------------------------

    def download_sentinel2(
        self,
        bbox: BBox,
        date_range: DateRange = ("2023-01-01", "2023-12-31"),
        max_cloud_cover: float = 20.0,
        bands: List[str] = ("B4", "B3", "B2", "B8"),   # R, G, B, NIR
        scale: int = 10,
        output_subdir: str = "sentinel2",
    ) -> Path:
        """Export a cloud-free Sentinel-2 composite to a local GeoTIFF.

        GEE mosaics the least-cloudy pixels from all scenes in the
        date window and exports a single composite image.

        Parameters
        ----------
        bbox:
            ``(lon_min, lat_min, lon_max, lat_max)`` in WGS-84.
        date_range:
            ``("YYYY-MM-DD", "YYYY-MM-DD")``.
        max_cloud_cover:
            Maximum per-scene cloud cover for the collection filter.
        bands:
            GEE band names. Defaults to R-G-B-NIR (10 m).
        scale:
            Output pixel size in metres.
        output_subdir:
            Sub-directory name inside ``dest_dir``.

        Returns
        -------
        Path
            Path to the exported GeoTIFF.
        """
        ee = self._ee
        region = self._bbox_to_ee_geometry(bbox)
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        print("[GEE] Building Sentinel-2 cloud-free composite …")
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(*date_range)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
            .select(list(bands))
        )

        count = collection.size().getInfo()
        if count == 0:
            raise RuntimeError(
                "No Sentinel-2 scenes found. "
                "Try wider date range or higher cloud-cover threshold."
            )
        print(f"  {count} scene(s) in collection.")

        composite = collection.median().divide(10000.0)   # scale to 0-1
        out_path = outdir / "sentinel2_composite.tif"

        self._export_image(
            image=composite,
            region=region,
            scale=scale,
            out_path=out_path,
            description="S2_composite",
        )
        return out_path

    # ------------------------------------------------------------------
    # Landsat 8 / 9
    # ------------------------------------------------------------------

    def download_landsat(
        self,
        bbox: BBox,
        date_range: DateRange = ("2023-01-01", "2023-12-31"),
        max_cloud_cover: float = 20.0,
        bands: List[str] = ("SR_B4", "SR_B3", "SR_B2", "SR_B5"),  # R,G,B,NIR
        scale: int = 30,
        output_subdir: str = "landsat",
    ) -> Path:
        """Export a cloud-free Landsat 8/9 composite to a local GeoTIFF.

        Uses the USGS Landsat Collection 2 Level-2 surface reflectance
        product (free via GEE).

        Parameters
        ----------
        bbox / date_range / max_cloud_cover:
            Same as :meth:`download_sentinel2`.
        bands:
            GEE band names (SR_B4=Red, SR_B3=Green, SR_B2=Blue, SR_B5=NIR).
        scale:
            Output pixel size in metres (native 30 m).
        output_subdir:
            Sub-directory name inside ``dest_dir``.

        Returns
        -------
        Path
            Path to the exported GeoTIFF.
        """
        ee = self._ee
        region = self._bbox_to_ee_geometry(bbox)
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        print("[GEE] Building Landsat cloud-free composite …")

        def _apply_scale(img):
            optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
            return img.addBands(optical, overwrite=True)

        collection = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
            .filterBounds(region)
            .filterDate(*date_range)
            .filter(ee.Filter.lte("CLOUD_COVER", max_cloud_cover))
            .map(_apply_scale)
            .select(list(bands))
        )

        count = collection.size().getInfo()
        if count == 0:
            raise RuntimeError("No Landsat scenes found.")
        print(f"  {count} scene(s) in collection.")

        composite = collection.median()
        out_path = outdir / "landsat_composite.tif"

        self._export_image(
            image=composite,
            region=region,
            scale=scale,
            out_path=out_path,
            description="Landsat_composite",
        )
        return out_path

    # ------------------------------------------------------------------
    # SRTM Digital Elevation Model (30 m)
    # ------------------------------------------------------------------

    def download_srtm_dem(
        self,
        bbox: BBox,
        scale: int = 30,
        output_subdir: str = "dem",
    ) -> Path:
        """Download SRTM 30 m DEM for the given bounding box.

        The SRTM data is available globally (between 60N and 56S) and
        is free / non-commercial via GEE.

        Returns
        -------
        Path
            Path to the saved GeoTIFF.
        """
        ee = self._ee
        region = self._bbox_to_ee_geometry(bbox)
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        print("[GEE] Exporting SRTM DEM …")
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
        out_path = outdir / "srtm_dem.tif"

        self._export_image(
            image=dem,
            region=region,
            scale=scale,
            out_path=out_path,
            description="SRTM_DEM",
        )
        return out_path

    # ------------------------------------------------------------------
    # OpenBuildings labels (Africa / South Asia – free)
    # ------------------------------------------------------------------

    def download_open_buildings_labels(
        self,
        bbox: BBox,
        output_subdir: str = "labels",
    ) -> Path:
        """Export a building-footprint raster from Google Open Buildings.

        Covers Africa, South and Southeast Asia, and Latin America.
        Returns a binary raster (1 = building, 0 = other).

        Returns
        -------
        Path
            Path to the saved GeoTIFF.
        """
        ee = self._ee
        region = self._bbox_to_ee_geometry(bbox)
        outdir = self.dest_dir / output_subdir
        outdir.mkdir(parents=True, exist_ok=True)

        print("[GEE] Exporting Open Buildings labels …")
        buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons") \
                      .filterBounds(region)

        mask = buildings.reduceToImage(
            properties=["confidence"],
            reducer=ee.Reducer.max(),
        ).gt(0).rename("buildings").unmask(0)

        out_path = outdir / "building_labels.tif"
        self._export_image(
            image=mask,
            region=region,
            scale=10,
            out_path=out_path,
            description="OpenBuildings",
        )
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export_image(
        self,
        image,
        region,
        scale: int,
        out_path: Path,
        description: str = "exploresat_export",
    ) -> None:
        """Export a GEE Image to a local GeoTIFF via getDownloadURL."""
        ee = self._ee

        url = image.getDownloadURL({
            "region": region,
            "scale": scale,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
            "description": description,
        })

        print(f"  Downloading {out_path.name} …")
        import urllib.request
        urllib.request.urlretrieve(url, str(out_path))
        print(f"  Saved to {out_path}")

    @staticmethod
    def _bbox_to_ee_geometry(bbox: BBox):
        """Convert (lon_min, lat_min, lon_max, lat_max) to ee.Geometry."""
        import ee
        lon_min, lat_min, lon_max, lat_max = bbox
        return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    @staticmethod
    def _init_ee(project: str):
        """Initialise the Earth Engine API."""
        try:
            import ee
        except ImportError as exc:
            raise ImportError(
                "earthengine-api is required.\n"
                "Install with: pip install earthengine-api\n"
                "Authenticate with: earthengine authenticate"
            ) from exc
        try:
            ee.Initialize(project=project)
        except Exception:
            # Fall back to default credential path
            ee.Authenticate(quiet=True)
            ee.Initialize(project=project)
        return ee


# ===========================================================================
# Shared utilities
# ===========================================================================

def _save_geotiff(
    arr: np.ndarray,
    bbox: BBox,
    out_path: str,
    crs: str = "EPSG:4326",
    band_names: Optional[List[str]] = None,
) -> None:
    """Write a NumPy array (C x H x W) to a GeoTIFF.

    Parameters
    ----------
    arr:
        Array of shape ``(C, H, W)`` or ``(H, W)`` (single band).
    bbox:
        Bounding box ``(lon_min, lat_min, lon_max, lat_max)``.
    out_path:
        Output file path.
    crs:
        Coordinate reference system string (EPSG code).
    band_names:
        Optional list of band name tags stored in the TIFF metadata.
    """
    import rasterio
    from rasterio.transform import from_bounds

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    c, h, w = arr.shape
    lon_min, lat_min, lon_max, lat_max = bbox
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

    dtype = arr.dtype
    if dtype == np.float64:
        dtype = np.float32
        arr = arr.astype(np.float32)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr)
        if band_names:
            for i, name in enumerate(band_names[:c], start=1):
                dst.update_tags(i, name=name)


def _stream_cog_to_file(href: str, out_path: str, bbox: BBox) -> None:
    """Crop and save a cloud-optimised GeoTIFF from a remote URL."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box

    with rasterio.open(href) as src:
        geom = [box(*bbox).__geo_interface__]
        try:
            cropped, transform = rio_mask(src, geom, crop=True)
            meta = src.meta.copy()
        except Exception:
            # bbox outside raster extent – read the whole file
            cropped = src.read()
            transform = src.transform
            meta = src.meta.copy()

        meta.update({
            "driver": "GTiff",
            "height": cropped.shape[1],
            "width": cropped.shape[2],
            "transform": transform,
            "compress": "lzw",
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(cropped)


def tile_geotiff(
    src_path: str | Path,
    dest_dir: str | Path,
    tile_size: int = 512,
    overlap: int = 64,
    min_valid_fraction: float = 0.5,
) -> List[Path]:
    """Slice a large GeoTIFF into fixed-size tiles.

    Parameters
    ----------
    src_path:
        Path to the source GeoTIFF.
    dest_dir:
        Directory where tiles will be saved.
    tile_size:
        Square tile size in pixels.
    overlap:
        Pixel overlap between adjacent tiles.
    min_valid_fraction:
        Skip tiles where the fraction of non-zero pixels is below
        this threshold (avoids saving mostly-nodata edge tiles).

    Returns
    -------
    List[Path]
        Paths to all saved tile files.
    """
    import rasterio
    from rasterio.windows import Window

    src_path = Path(src_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stride = tile_size - overlap
    saved: List[Path] = []

    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        meta = src.meta.copy()
        meta.update({"height": tile_size, "width": tile_size,
                      "compress": "lzw"})

        for row in range(0, h - tile_size + 1, stride):
            for col in range(0, w - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                data = src.read(window=window)

                # Skip mostly-nodata tiles
                valid = np.count_nonzero(data) / data.size
                if valid < min_valid_fraction:
                    continue

                tile_transform = src.window_transform(window)
                out_meta = meta.copy()
                out_meta["transform"] = tile_transform

                out_path = dest_dir / f"{src_path.stem}_r{row:05d}_c{col:05d}.tif"
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(data)
                saved.append(out_path)

    print(f"Tiled {src_path.name} → {len(saved)} tiles in {dest_dir}")
    return saved
