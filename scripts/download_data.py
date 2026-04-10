"""
CLI: Download satellite data from Microsoft Planetary Computer or GEE.

Examples
--------
# Sentinel-2 over Delhi using Planetary Computer (no account needed)
python scripts/download_data.py \\
    --backend planetary_computer \\
    --dataset sentinel2 \\
    --bbox 77.0 28.4 77.4 28.8 \\
    --date-start 2025-06-01 --date-end 2025-09-30 \\
    --dest data/raw

# Landsat over Mumbai using GEE (requires --gee-project)
python scripts/download_data.py \\
    --backend gee --gee-project my-gcp-project \\
    --dataset landsat \\
    --bbox 72.7 18.8 73.1 19.2 \\
    --dest data/raw

# Tile the downloaded GeoTIFFs into 512x512 patches
python scripts/download_data.py --tile-only \\
    --src data/raw/sentinel2 --dest data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download satellite data for ExploreSat"
    )
    p.add_argument("--backend", choices=["planetary_computer", "gee"],
                   default="planetary_computer")
    p.add_argument("--dataset",
                   choices=["sentinel2", "landsat", "dem", "naip", "landcovernet_asia"],
                   default="sentinel2")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[77.0, 28.4, 77.4, 28.8],
                   help="Bounding box in WGS-84")
    p.add_argument("--date-start", default="2025-06-01")
    p.add_argument("--date-end",   default="2025-12-31")
    p.add_argument("--max-cloud",  type=float, default=20.0)
    p.add_argument("--max-items",  type=int,   default=5)
    p.add_argument("--max-chips",  type=int,   default=1,
                   help="Max chips per tile for LandCoverNet (use -1 for all)")
    p.add_argument("--dest",       default="data/raw")
    p.add_argument("--gee-project", default="",
                   help="GCP project ID (GEE backend only)")
    # Tiling
    p.add_argument("--tile-only",  action="store_true",
                   help="Skip download; tile existing GeoTIFFs in --src")
    p.add_argument("--src",        default=None,
                   help="Source directory for --tile-only mode")
    p.add_argument("--tile-size",  type=int, default=512)
    p.add_argument("--overlap",    type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bbox = tuple(args.bbox)
    date_range = (args.date_start, args.date_end)

    if args.tile_only:
        from data.download import tile_geotiff
        src_dir = Path(args.src or args.dest)
        tif_files = sorted(src_dir.rglob("*.tif"))
        if not tif_files:
            print(f"No .tif files found in {src_dir}")
            sys.exit(1)
        for tif in tif_files:
            tile_geotiff(
                src_path=tif,
                dest_dir=Path(args.dest) / tif.stem,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )
        return

    if args.dataset == "landcovernet_asia":
        from data.download import LandCoverNetDownloader
        dl = LandCoverNetDownloader(dest_dir=args.dest)
        dl.download_asia_subset(max_chips_per_tile=args.max_chips)

    elif args.backend == "planetary_computer":
        from data.download import PlanetaryComputerDownloader
        dl = PlanetaryComputerDownloader(dest_dir=args.dest)
        getattr(dl, f"download_{args.dataset}")(
            bbox=bbox,
            **_filter_kwargs(args.dataset, args, date_range),
        )

    elif args.backend == "gee":
        if not args.gee_project:
            print("--gee-project is required for the GEE backend.")
            sys.exit(1)
        from data.download import GEEDownloader
        dl = GEEDownloader(project=args.gee_project, dest_dir=args.dest)
        if args.dataset == "dem":
            dl.download_srtm_dem(bbox=bbox)
        else:
            getattr(dl, f"download_{args.dataset}")(
                bbox=bbox,
                date_range=date_range,
                max_cloud_cover=args.max_cloud,
            )


def _filter_kwargs(dataset: str, args: argparse.Namespace,
                   date_range: tuple) -> dict:
    if dataset == "dem":
        return {}
    if dataset == "naip":
        return {}
    return {
        "date_range": date_range,
        "max_cloud_cover": args.max_cloud,
        "max_items": args.max_items,
    }


if __name__ == "__main__":
    main()
