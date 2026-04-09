"""
CLI: Start the ExploreSat FastAPI server.

The server provides:
  - REST API for downloads, inference, and XYZ tile serving
  - Leaflet.js web map at http://localhost:8000/
  - Swagger UI at http://localhost:8000/docs

QGIS XYZ tile URL (add in Layer > Add XYZ Tile Layer):
  http://localhost:8000/tiles/{layer}/{z}/{x}/{y}.png

Examples
--------
# Default (development, auto-reload)
python scripts/serve.py

# Custom host/port
python scripts/serve.py --host 0.0.0.0 --port 8080

# Production (multi-worker via gunicorn)
python scripts/serve.py --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start the ExploreSat API server")
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8000)
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (>1 uses gunicorn)")
    p.add_argument("--reload",  action="store_true",
                   help="Enable auto-reload (development only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.workers > 1:
        # Multi-worker production mode via gunicorn
        try:
            import gunicorn  # noqa: F401
        except ImportError:
            print("gunicorn is not installed.  "
                  "Install with: pip install gunicorn")
            sys.exit(1)
        import os
        os.execvp("gunicorn", [
            "gunicorn",
            "exploresat.api.app:app",
            "-k", "uvicorn.workers.UvicornWorker",
            "--workers", str(args.workers),
            "--bind", f"{args.host}:{args.port}",
        ])
    else:
        import uvicorn
        print(f"Starting ExploreSat server at http://{args.host}:{args.port}")
        print(f"  Web map:   http://localhost:{args.port}/")
        print(f"  Swagger:   http://localhost:{args.port}/docs")
        print(f"  QGIS tile: http://localhost:{args.port}/tiles/{{layer}}/{{z}}/{{x}}/{{y}}.png")
        uvicorn.run(
            "exploresat.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
