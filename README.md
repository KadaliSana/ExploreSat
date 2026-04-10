# ExploreSat

**Automatic Extraction of Topographical Features from Satellite and Drone Images**

Deep-learning segmentation pipeline that detects buildings, roads, water,
vegetation, and other topographical features from satellite or drone imagery.

---

## Feature Overview

| Capability | Details |
|---|---|
| **Segmentation models** | U-Net, U-Net++, DeepLabV3+, FPN (via `segmentation-models-pytorch`) |
| **Pre-trained encoders** | ResNet-34/50, EfficientNet-B3/B4, MobileNet-V2 (ImageNet weights) |
| **Classes detected** | Impervious surfaces, Buildings, Low vegetation, Trees, Cars, Background |
| **Data acquisition** | Microsoft Planetary Computer  · Google Earth Engine |
| **Datasets** | Sentinel-2 (10 m), Landsat 8/9 (30 m), SRTM DEM, NAIP (1 m USA), Copernicus DEM |
| **Visualisation** | FastAPI server + Leaflet.js web map · QGIS XYZ tile integration |
| **Output formats** | Class-index GeoTIFF · colour-coded RGB GeoTIFF · GeoPackage polygons |

---

## Project Structure

```
ExploreSat/
├── configs/
│   └── default_config.yaml          # All tunable parameters
├── scripts/
│   ├── download_data.py             # CLI: download satellite imagery
│   ├── train.py                     # CLI: train segmentation model
│   ├── predict.py                   # CLI: run inference on GeoTIFFs
│   └── serve.py                     # CLI: start FastAPI server
├── src/exploresat/
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset (ISPRS Potsdam/Vaihingen)
│   │   └── download.py              # GEE + Planetary Computer downloaders
│   ├── models/
│   │   └── segmentation.py          # Model factory + fallback SimpleUNet
│   ├── training/
│   │   └── trainer.py               # AMP training loop
│   ├── inference/
│   │   └── predictor.py             # Sliding-window inference + export
│   ├── utils/
│   │   └── metrics.py               # IoU, Dice, pixel accuracy
│   ├── api/
│   │   ├── app.py                   # FastAPI application
│   │   ├── routers/
│   │   │   ├── download.py          # POST /download/
│   │   │   ├── inference.py         # POST /inference/
│   │   │   └── tiles.py             # GET /tiles/{layer}/{z}/{x}/{y}.png
│   │   └── templates/
│   │       └── index.html           # Leaflet.js web map
│   └── qgis/
│       └── load_layers.py           # PyQGIS helper to load layers
├── data/
│   ├── raw/                         # Downloaded GeoTIFFs
│   ├── processed/                   # Tiled image/label pairs
│   └── predictions/                 # Inference outputs
├── checkpoints/                     # Saved model weights
├── notebooks/
│   └── 01_download_and_explore.ipynb
└── tests/
```

---

## Quick-start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download satellite data

**Option A – Microsoft Planetary Computer (no account needed)**

```bash
python scripts/download_data.py \
    --backend planetary_computer \
    --dataset sentinel2 \
    --bbox 77.0 28.4 77.4 28.8 \
    --date-start 2023-06-01 --date-end 2023-09-30 \
    --dest data/raw
```

**Option B – Google Earth Engine**

```bash
# One-time: earthengine authenticate
python scripts/download_data.py \
    --backend gee --gee-project MY_GCP_PROJECT \
    --dataset sentinel2 \
    --bbox 77.0 28.4 77.4 28.8 \
    --dest data/raw
```

Available datasets: `sentinel2` · `landsat` · `dem` · `naip`

### 3. Tile into training patches

```bash
python scripts/download_data.py --tile-only \
    --src data/raw/sentinel2 \
    --dest data/processed \
    --tile-size 512 --overlap 64
```

### 4. Train

```bash
python scripts/train.py \
    --data data/processed \
    --arch unet --encoder resnet34 \
    --epochs 50 --batch-size 8
```

Best checkpoint is saved to `checkpoints/best_model.pth`.

### 5. Run inference

```bash
# Single file
python scripts/predict.py \
    --input data/raw/sentinel2/scene.tif \
    --output data/predictions/scene_pred.tif \
    --vector

# Batch (all .tif in a directory)
python scripts/predict.py \
    --input data/raw/sentinel2/ \
    --output data/predictions/
```

### 6. Start the visualisation server

```bash
python scripts/serve.py
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/` | Leaflet.js interactive map |
| `http://localhost:8000/docs` | Swagger API explorer |
| `http://localhost:8000/tiles/{layer}/{z}/{x}/{y}.png` | XYZ tiles for QGIS |

---

## QGIS Integration

1. Start the server: `python scripts/serve.py`
2. In QGIS: **Layer → Add Layer → Add XYZ Tile Layer**
3. Enter the URL:
   ```
   http://localhost:8000/tiles/{layer}/{z}/{x}/{y}.png
   ```
   Replace `{layer}` with a name from `GET /tiles/layers`
   (e.g. `abc123_pred_rgb`).
4. For vector polygons: **Layer → Add Layer → Add Vector Layer**,
   select the `.gpkg` file from `data/predictions/`.

**Or load all layers automatically from the QGIS Python Console:**

```python
import sys; sys.path.insert(0, '/path/to/ExploreSat/src')
from exploresat.qgis.load_layers import load_all
load_all()   # fetches layer list from server and adds everything
```

---

## REST API Reference

### `POST /download/`
Enqueue a background satellite data download.

```json
{
  "backend": "planetary_computer",
  "dataset": "sentinel2",
  "bbox": {"lon_min": 77.0, "lat_min": 28.4, "lon_max": 77.4, "lat_max": 28.8},
  "date_start": "2023-06-01",
  "date_end": "2023-09-30",
  "max_cloud_cover": 20
}
```

### `POST /inference/`
Upload an image, receive segmentation outputs.

```bash
curl -X POST http://localhost:8000/inference/ \
     -F "file=@scene.tif" -F "export_vector=true"
```

### `GET /tiles/{layer}/{z}/{x}/{y}.png`
XYZ map tile — plug directly into QGIS or Leaflet.

### `GET /tiles/layers`
List all available layers.

### `GET /inference/results`
List all saved prediction files.

---

## License

See [LICENSE](LICENSE). Data from Google Earth Engine and Microsoft Planetary
Computer is subject to their respective free non-commercial use terms.
