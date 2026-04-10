"""
CLI: Train the segmentation model.

Examples
--------
# Quick-start with default settings (RTX 3060)
python scripts/train.py --data data/processed --epochs 50

# Use a larger encoder and batch size
python scripts/train.py \\
    --data data/processed \\
    --arch unet++ --encoder efficientnet-b3 \\
    --epochs 100 --batch-size 4

# CPU-only training (slow, for testing)
python scripts/train.py --data data/processed --device cpu --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ExploreSat segmentation model")
    p.add_argument("--data",        default="data/processed",
                   help="Dataset root (must contain images/ and labels/)")
    p.add_argument("--dataset-type", default="isprs",
                   choices=["isprs", "landcovernet"])
    p.add_argument("--arch",        default="unet",
                   choices=["unet", "unet++", "deeplabv3+", "segformer",
                             "pspnet", "linknet", "manet"])
    p.add_argument("--encoder",     default="resnet34")
    p.add_argument("--in-channels", type=int, default=0) # 0 means use dataset default
    p.add_argument("--num-classes", type=int, default=6)
    p.add_argument("--image-size",  type=int, default=512)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--workers",     type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import albumentations as A
    import torch
    from torch.utils.data import DataLoader, random_split

    from data.dataset import TopographyDataset, LandCoverNetDataset
    from models.segmentation import build_model
    from training.trainer import Trainer

    # ---- Augmentations ----
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
    ])

    # ---- Datasets ----
    ds_class = TopographyDataset if args.dataset_type == "isprs" else LandCoverNetDataset
    default_classes = 6 if args.dataset_type == "isprs" else 8
    num_classes = args.num_classes if args.num_classes != 6 else default_classes
    
    # Default in_channels based on dataset
    if args.in_channels == 0:
        in_channels = 3 if args.dataset_type == "isprs" else 4
    else:
        in_channels = args.in_channels

    try:
        train_ds = ds_class(
            root=args.data, split="train",
            image_size=args.image_size, transform=train_transform,
        )
        val_ds = ds_class(
            root=args.data, split="val",
            image_size=args.image_size,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        # Fall back to a random split when no split files exist
        print(f"Note: {exc} – using random 85/15 split.")
        full_ds = ds_class(
            root=args.data, image_size=args.image_size,
            transform=train_transform,
        )
        n_val = max(1, int(len(full_ds) * 0.15))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} tiles | Val: {len(val_ds)} tiles")

    # ---- Model ----
    try:
        model = build_model(
            architecture=args.arch,
            encoder=args.encoder,
            in_channels=in_channels,
            num_classes=num_classes,
        )
    except ImportError:
        from models.segmentation import build_simple_unet
        print("segmentation-models-pytorch not found – using built-in SimpleUNet.")
        model = build_simple_unet(
            in_channels=args.in_channels,
            num_classes=num_classes,
        )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    best_miou = max(history["val_miou"])
    print(f"\nTraining complete. Best validation mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
