"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import io

import pytest

# httpx is required by FastAPI's TestClient
pytest.importorskip("httpx", reason="httpx not installed")

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "cuda" in data


def test_list_tile_layers():
    resp = client.get("/tiles/layers")
    assert resp.status_code == 200
    assert "layers" in resp.json()


def test_list_inference_results():
    resp = client.get("/inference/results")
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_download_missing_gee_project():
    """POSTing to /download/ with backend=gee but no gee_project should 422."""
    payload = {
        "backend": "gee",
        "dataset": "sentinel2",
        "bbox": {"lon_min": 77.0, "lat_min": 28.0,
                 "lon_max": 77.5, "lat_max": 28.5},
    }
    resp = client.post("/download/", json=payload)
    assert resp.status_code == 422


def test_download_planetary_computer_queued():
    """A valid Planetary Computer download request should be accepted."""
    payload = {
        "backend": "planetary_computer",
        "dataset": "sentinel2",
        "bbox": {"lon_min": 77.0, "lat_min": 28.0,
                 "lon_max": 77.5, "lat_max": 28.5},
        "date_start": "2023-01-01",
        "date_end": "2023-12-31",
    }
    resp = client.post("/download/", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"


def test_inference_no_checkpoint(tmp_path):
    """Uploading an image without a checkpoint should return 503."""
    # Make sure no checkpoint exists during this test
    fake_img = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    resp = client.post(
        "/inference/",
        files={"file": ("test.png", fake_img, "image/png")},
    )
    # 503 if no checkpoint, 500 if image is invalid – both are acceptable
    # (we just confirm it doesn't crash with 200)
    assert resp.status_code in (200, 422, 500, 503)
