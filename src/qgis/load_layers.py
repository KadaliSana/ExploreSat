"""
QGIS integration – load ExploreSat layers directly from the FastAPI server.

Usage
-----
1. Start the ExploreSat server::

    python scripts/serve.py

2. Open the QGIS Python Console (Plugins → Python Console) and run::

    exec(open('/path/to/exploresat/qgis/load_layers.py').read())

   Or install as a QGIS plugin (see ``qgis_plugin/`` for the full plugin
   boilerplate).

What this script does
---------------------
* Fetches the list of available prediction layers from the FastAPI server.
* Adds each segmentation result as an XYZ tile layer in the current QGIS
  project (colour-coded, semi-transparent overlay on top of a satellite
  base map).
* Adds any GeoPackage (.gpkg) vector files found in ``data/predictions/``
  as vector polygon layers with class-colour styling.
* Adds the Esri World Imagery base map as a reference layer.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

# Default server URL – change if running on a different host/port
SERVER_URL = os.environ.get("EXPLORESAT_SERVER", "http://localhost:8000")

# Where the server stores prediction files (must be accessible from this machine)
PREDICTIONS_DIR = Path(os.environ.get("EXPLORESAT_PREDICTIONS", "data/predictions"))

# ISPRS colour map: class_id → (R, G, B, A) 0-255
CLASS_COLOURS = {
    0: (255, 255, 255, 180),   # Impervious surfaces – white
    1: (0,   0,   255, 200),   # Buildings           – blue
    2: (0,   255, 255, 180),   # Low vegetation      – cyan
    3: (0,   255,   0, 180),   # Trees               – green
    4: (255, 255,   0, 200),   # Cars                – yellow
    5: (255,   0,   0, 160),   # Background/clutter  – red
}
CLASS_NAMES = [
    "Impervious surfaces", "Buildings", "Low vegetation",
    "Trees", "Cars", "Background / clutter",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all(server: str = SERVER_URL) -> None:
    """Load all available layers into the current QGIS project.

    Call from the QGIS Python Console::

        from qgis.load_layers import load_all
        load_all()
    """
    _ensure_qgis()

    add_esri_basemap()
    layers = _fetch_layer_list(server)
    for name in layers:
        add_xyz_tile_layer(name, server=server)

    gpkg_files = sorted(PREDICTIONS_DIR.glob("*_polygons.gpkg"))
    for gpkg in gpkg_files:
        add_vector_layer(gpkg)

    from qgis.utils import iface
    iface.mapCanvas().refresh()
    print(f"[ExploreSat] Loaded {len(layers)} tile layer(s) "
          f"and {len(gpkg_files)} vector layer(s).")


def add_esri_basemap() -> None:
    """Add Esri World Imagery as the background base layer."""
    _ensure_qgis()
    from qgis.core import QgsRasterLayer, QgsProject

    url = (
        "type=xyz"
        "&url=https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        "&zmax=18&zmin=0"
    )
    layer = QgsRasterLayer(url, "Esri World Imagery", "wms")
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        print("[ExploreSat] Added Esri World Imagery base map.")
    else:
        print("[ExploreSat] Warning: Could not add Esri base map.")


def add_xyz_tile_layer(layer_name: str, server: str = SERVER_URL,
                       opacity: float = 0.7) -> None:
    """Add a segmentation result as an XYZ tile layer.

    Parameters
    ----------
    layer_name:
        Layer stem as returned by ``GET /tiles/layers``.
    server:
        FastAPI server base URL.
    opacity:
        Layer opacity in QGIS (0.0 – 1.0).
    """
    _ensure_qgis()
    from qgis.core import QgsRasterLayer, QgsProject

    tile_url = f"{server}/tiles/{layer_name}/{{z}}/{{x}}/{{y}}.png"
    conn_str = (
        f"type=xyz&url={urllib.request.quote(tile_url, safe=':/?={{}}&')}"
        "&zmax=21&zmin=0"
    )
    layer = QgsRasterLayer(conn_str, f"ExploreSat: {layer_name}", "wms")
    if layer.isValid():
        layer.setOpacity(opacity)
        QgsProject.instance().addMapLayer(layer)
        print(f"[ExploreSat] Added tile layer: {layer_name}")
    else:
        print(f"[ExploreSat] Warning: tile layer invalid – {layer_name}")


def add_vector_layer(gpkg_path: str | Path) -> None:
    """Add a GeoPackage prediction as a styled vector layer.

    Parameters
    ----------
    gpkg_path:
        Path to a ``*_polygons.gpkg`` file produced by the predictor.
    """
    _ensure_qgis()
    from qgis.core import (
        QgsVectorLayer, QgsProject, QgsSymbol,
        QgsCategorizedSymbolRenderer, QgsRendererCategory,
        QgsMarkerSymbol,
    )
    from qgis.core import QgsFillSymbol
    from PyQt5.QtGui import QColor

    gpkg_path = str(gpkg_path)
    layer = QgsVectorLayer(gpkg_path, Path(gpkg_path).stem, "ogr")
    if not layer.isValid():
        print(f"[ExploreSat] Warning: vector layer invalid – {gpkg_path}")
        return

    # Build categorised renderer by class_id
    categories = []
    for cls_id, (r, g, b, a) in CLASS_COLOURS.items():
        symbol = QgsFillSymbol.createSimple({
            "color": f"{r},{g},{b},{a}",
            "outline_style": "no",
        })
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        categories.append(QgsRendererCategory(cls_id, symbol, label))

    renderer = QgsCategorizedSymbolRenderer("class_id", categories)
    layer.setRenderer(renderer)
    QgsProject.instance().addMapLayer(layer)
    print(f"[ExploreSat] Added vector layer: {Path(gpkg_path).name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_layer_list(server: str) -> List[str]:
    """Fetch available layer names from the FastAPI server."""
    try:
        with urllib.request.urlopen(f"{server}/tiles/layers", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("layers", [])
    except Exception as exc:
        print(f"[ExploreSat] Could not reach server at {server}: {exc}")
        return []


def _ensure_qgis() -> None:
    """Raise a clear error if QGIS Python bindings are not available."""
    try:
        import qgis.core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "QGIS Python bindings (qgis.core) not found.\n"
            "Run this script inside QGIS (Plugins → Python Console), "
            "or install the 'qgis' package in your environment."
        ) from exc
