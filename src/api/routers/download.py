"""
Download router – trigger GEE or Planetary Computer downloads via REST.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/download", tags=["Download"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    lon_min: float = Field(..., example=77.5)
    lat_min: float = Field(..., example=28.5)
    lon_max: float = Field(..., example=77.8)
    lat_max: float = Field(..., example=28.8)


class DownloadRequest(BaseModel):
    backend: Literal["planetary_computer", "gee"] = "planetary_computer"
    dataset: Literal["sentinel2", "landsat", "dem", "naip"] = "sentinel2"
    bbox: BBox
    date_start: str = Field("2023-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_end: str = Field("2023-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$")
    max_cloud_cover: float = Field(20.0, ge=0, le=100)
    max_items: int = Field(5, ge=1, le=20)
    dest_dir: str = "data/raw"
    # GEE-only
    gee_project: Optional[str] = None


class DownloadResponse(BaseModel):
    status: str
    message: str
    files: List[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=DownloadResponse,
             summary="Start a dataset download")
async def start_download(
    req: DownloadRequest,
    background_tasks: BackgroundTasks,
) -> DownloadResponse:
    """Enqueue a background download job.

    The download runs asynchronously; check ``data/raw/`` for results.
    """
    bbox_tuple = (req.bbox.lon_min, req.bbox.lat_min,
                  req.bbox.lon_max, req.bbox.lat_max)
    date_range = (req.date_start, req.date_end)

    if req.backend == "planetary_computer":
        background_tasks.add_task(
            _run_planetary_computer,
            dataset=req.dataset,
            bbox=bbox_tuple,
            date_range=date_range,
            max_cloud_cover=req.max_cloud_cover,
            max_items=req.max_items,
            dest_dir=req.dest_dir,
        )
    elif req.backend == "gee":
        if not req.gee_project:
            raise HTTPException(
                status_code=422,
                detail="gee_project is required when backend='gee'",
            )
        background_tasks.add_task(
            _run_gee,
            dataset=req.dataset,
            bbox=bbox_tuple,
            date_range=date_range,
            max_cloud_cover=req.max_cloud_cover,
            dest_dir=req.dest_dir,
            gee_project=req.gee_project,
        )
    else:
        raise HTTPException(status_code=422, detail="Unknown backend.")

    return DownloadResponse(
        status="started",
        message=(
            f"{req.dataset} download queued via {req.backend}. "
            f"Files will appear in {req.dest_dir}/."
        ),
    )


# ---------------------------------------------------------------------------
# Background task implementations
# ---------------------------------------------------------------------------

def _run_planetary_computer(
    dataset: str,
    bbox: tuple,
    date_range: tuple,
    max_cloud_cover: float,
    max_items: int,
    dest_dir: str,
) -> None:
    from data.download import PlanetaryComputerDownloader

    dl = PlanetaryComputerDownloader(dest_dir=dest_dir)
    if dataset == "sentinel2":
        dl.download_sentinel2(bbox, date_range, max_cloud_cover, max_items)
    elif dataset == "landsat":
        dl.download_landsat(bbox, date_range, max_cloud_cover, max_items)
    elif dataset == "dem":
        dl.download_dem(bbox)
    elif dataset == "naip":
        dl.download_naip(bbox)


def _run_gee(
    dataset: str,
    bbox: tuple,
    date_range: tuple,
    max_cloud_cover: float,
    dest_dir: str,
    gee_project: str,
) -> None:
    from data.download import GEEDownloader

    dl = GEEDownloader(project=gee_project, dest_dir=dest_dir)
    if dataset == "sentinel2":
        dl.download_sentinel2(bbox, date_range, max_cloud_cover)
    elif dataset == "landsat":
        dl.download_landsat(bbox, date_range, max_cloud_cover)
    elif dataset == "dem":
        dl.download_srtm_dem(bbox)
