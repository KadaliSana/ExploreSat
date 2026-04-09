"""
CLI: Run inference (feature extraction) on a GeoTIFF image.

Examples
--------
# Single file – outputs class map + RGB overlay + vector polygons
python scripts/predict.py \\
    --input data/raw/sentinel2/scene.tif \\
    --output data/predictions/scene_pred.tif \\
    --vector

# Batch mode – process every .tif in a directory
python scripts/predict.py \\
    --input data/raw/sentinel2/ \\
    --output data/predictions/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run topographical feature extraction on GeoTIFF images"
    )
    p.add_argument("--input",  required=True,
                   help="Input GeoTIFF file or directory of GeoTIFFs")
    p.add_argument("--output", required=True,
                   help="Output file (single) or directory (batch)")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--tile-size",  type=int, default=512)
    p.add_argument("--overlap",    type=int, default=64)
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--vector",     action="store_true",
                   help="Also export vector polygons (.gpkg)")
    p.add_argument("--no-rgb",     action="store_true",
                   help="Skip saving the colour-coded RGB overlay")
    p.add_argument("--arch",       default="unet")
    p.add_argument("--encoder",    default="resnet34")
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from exploresat.inference.predictor import Predictor

    model = _load_model(args)

    predictor = Predictor(
        model=model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        tif_files = sorted(input_path.rglob("*.tif"))
        if not tif_files:
            print(f"No .tif files found in {input_path}")
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)
        for tif in tif_files:
            out = output_path / (tif.stem + "_pred.tif")
            predictor.predict_geotiff(
                src_path=tif, out_path=out,
                export_rgb=not args.no_rgb,
                export_vector=args.vector,
            )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.predict_geotiff(
            src_path=input_path, out_path=output_path,
            export_rgb=not args.no_rgb,
            export_vector=args.vector,
        )

    print("Done.")


def _load_model(args: argparse.Namespace):
    import torch
    from pathlib import Path

    ckpt_path = Path(args.checkpoint)

    try:
        from exploresat.models.segmentation import build_model
        model = build_model(
            architecture=args.arch,
            encoder=args.encoder,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            encoder_weights=None,
        )
    except ImportError:
        from exploresat.models.segmentation import build_simple_unet
        model = build_simple_unet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
        )

    if not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path}. "
              "Using randomly initialised weights.")
        return model

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


if __name__ == "__main__":
    main()
