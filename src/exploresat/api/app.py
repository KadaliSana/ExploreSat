"""
ExploreSat FastAPI application.

Start the server
----------------
    # Development (auto-reload):
    uvicorn exploresat.api.app:app --reload --host 0.0.0.0 --port 8000

    # Production (4 workers):
    gunicorn exploresat.api.app:app -k uvicorn.workers.UvicornWorker \\
        --workers 4 --bind 0.0.0.0:8000

QGIS integration
----------------
After starting the server, add layers in QGIS via:

  Layer → Add Layer → Add XYZ Tile Layer
  URL:  http://localhost:8000/tiles/{layer}/{z}/{x}/{y}.png

  Layer → Add Layer → Add Vector Layer  (GeoPackage from POST /inference)

Web map (browser)
-----------------
  http://localhost:8000/        → Leaflet.js interactive map
  http://localhost:8000/docs    → Interactive Swagger UI
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

from exploresat.api.routers import download, inference, tiles

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ExploreSat – Topographical Feature Extraction",
    description=(
        "Automatic extraction of topographical features from satellite "
        "and drone imagery using deep-learning segmentation models.\n\n"
        "**Data sources:** Google Earth Engine, Microsoft Planetary Computer\n\n"
        "**Visualization:** QGIS XYZ tiles + Leaflet.js web map"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow QGIS / browser clients on any origin (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

app.include_router(download.router)
app.include_router(inference.router)
app.include_router(tiles.router)

# ---------------------------------------------------------------------------
# Static files & Jinja2 templates (web map UI)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_STATIC_DIR = _HERE / "static"
_TEMPLATES_DIR = _HERE / "templates"

_STATIC_DIR.mkdir(exist_ok=True)
_TEMPLATES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Root – Leaflet.js web map
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request) -> HTMLResponse:
    """Serve the interactive Leaflet.js map."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health() -> dict:
    """Return server status and available CUDA devices."""
    import torch
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    return {
        "status": "ok",
        "cuda": cuda_available,
        "gpu": gpu_name,
    }
