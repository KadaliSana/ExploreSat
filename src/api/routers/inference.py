"""
Inference router – upload an image, receive segmentation results.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/inference", tags=["Inference"])

PREDICTIONS_DIR = Path("data/predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    job_id: str
    class_map_tif: str
    rgb_vis_tif: str
    vector_gpkg: str | None = None
    message: str


class ResultList(BaseModel):
    results: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=PredictionResult,
             summary="Run feature extraction on an uploaded image")
async def predict(
    file: UploadFile = File(..., description="GeoTIFF or PNG/JPEG image"),
    export_vector: bool = False,
) -> PredictionResult:
    """Upload a satellite or drone image and receive a segmentation map.

    Returns paths to:
    * ``class_map_tif`` – class-index GeoTIFF (1 band, uint8)
    * ``rgb_vis_tif``   – colour-coded RGB GeoTIFF for visual inspection
    * ``vector_gpkg``   – GeoPackage polygon features (if requested)
    """
    job_id = uuid.uuid4().hex[:8]
    suffix = Path(file.filename).suffix if file.filename else ".tif"
    upload_path = PREDICTIONS_DIR / f"{job_id}_input{suffix}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        model = _load_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model found at checkpoints/best_model.pth. "
                "Train first with:  python scripts/train.py"
            ),
        )

    from inference.predictor import Predictor

    # Use correct 4-channel (RGBI) normalization stats matching training
    predictor = Predictor(
        model=model,
        mean=(0.485, 0.456, 0.406, 0.5),
        std=(0.229, 0.224, 0.225, 0.25),
    )
    out_path = PREDICTIONS_DIR / f"{job_id}_pred.tif"

    predictor.predict_geotiff(
        src_path=upload_path,
        out_path=out_path,
        export_rgb=True,
        export_vector=export_vector,
    )

    rgb_path = out_path.with_name(out_path.stem + "_rgb.tif")
    vec_path = out_path.with_name(out_path.stem + "_polygons.gpkg")

    return PredictionResult(
        job_id=job_id,
        class_map_tif=str(out_path),
        rgb_vis_tif=str(rgb_path) if rgb_path.exists() else "",
        vector_gpkg=str(vec_path) if (export_vector and vec_path.exists()) else None,
        message="Prediction complete.",
    )


@router.get("/results", response_model=ResultList,
            summary="List all saved prediction files")
async def list_results() -> ResultList:
    """Return names of all GeoTIFFs in ``data/predictions/``."""
    files = sorted(p.name for p in PREDICTIONS_DIR.glob("*.tif"))
    return ResultList(results=files)


@router.get("/results/{filename}",
            summary="Download a prediction file by name")
async def download_result(filename: str) -> FileResponse:
    """Download a specific prediction GeoTIFF or GeoPackage by name."""
    path = PREDICTIONS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), filename=filename)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_model():
    """Load the best checkpoint; raise FileNotFoundError if absent."""
    import torch
    import yaml
    from models.segmentation import build_model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu",
                      weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    # Read config for model architecture
    try:
        with open("configs/default_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)["model"]
        arch = cfg["architecture"]
        encoder = cfg["encoder"]
        in_channels = cfg.get("in_channels", 4)
        num_classes = cfg.get("num_classes", 8)
    except Exception:
        arch, encoder, in_channels, num_classes = "unet", "resnet34", 4, 8

    # Try building with config settings first
    try:
        model = build_model(
            architecture=arch,
            encoder=encoder,
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_weights=None,
        )
        model.load_state_dict(state)
        print(f"Model loaded: {arch} / {encoder} / {in_channels}ch / {num_classes}cls")
    except RuntimeError as e:
        # Config doesn't match checkpoint – fall back to known-good architecture
        print(f"Warning: Config ({arch}/{encoder}) doesn't match checkpoint weights.")
        print(f"  Detail: {e}")
        print(f"  Falling back to UNet/ResNet34 (matching trained checkpoint).")
        model = build_model(
            architecture="unet",
            encoder="resnet34",
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_weights=None,
        )
        model.load_state_dict(state)

    model.eval()
    return model

