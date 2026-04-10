"""
Training loop for segmentation models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils.metrics import mean_iou, pixel_accuracy


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits.float(), dim=1)
        
        # Convert targets to one-hot: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignore_index (-1) if necessary
        mask = (targets >= 0).float().unsqueeze(1)
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth + 1e-7)
        return 1 - dice.mean()
    
    def __repr__(self):
        return f"DiceLoss(smooth={self.smooth})"


class Trainer:
    """Train a segmentation model.

    Parameters
    ----------
    model:
        Segmentation model (``nn.Module``).
    num_classes:
        Number of segmentation classes.
    lr:
        Initial learning rate.
    weight_decay:
        AdamW weight-decay regularisation.
    checkpoint_dir:
        Directory to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 6,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "auto",
        checkpoint_dir: str | Path = "checkpoints",
        **kwargs,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.dice_criterion = DiceLoss()
        
        self.optimizer = AdamW(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        scheduler_epochs: Optional[int] = None,
    ) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        train_loader / val_loader:
            PyTorch DataLoaders for training and validation sets.
        epochs:
            Number of training epochs.
        scheduler_epochs:
            Total epochs for the cosine LR schedule (defaults to
            ``epochs``).

        Returns
        -------
        dict
            Training history: ``{"train_loss", "val_loss", "val_miou",
            "val_acc"}`` lists indexed by epoch.
        """
        t_epochs = scheduler_epochs or epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=t_epochs,
                                      eta_min=1e-6)

        history: dict = {"train_loss": [], "val_loss": [],
                         "val_miou": [], "val_acc": []}
        best_miou = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_miou, val_acc = self._val_epoch(val_loader)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_miou"].append(val_miou)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mIoU={val_miou:.4f} | "
                f"acc={val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_miou > best_miou:
                best_miou = val_miou
                self.save_checkpoint("best_model.pth")
                print(f"  → New best mIoU: {best_miou:.4f} – checkpoint saved.")

        self.save_checkpoint("last_model.pth")
        return history

    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Skip batches where ALL labels are ignored (-1)
            if (labels >= 0).sum() == 0:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss_ce = self.ce_criterion(outputs, labels)
            loss_dice = self.dice_criterion(outputs, labels)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            # Guard against NaN loss – skip batch instead of corrupting weights
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
        return total_loss / max(valid_batches, 1)

    def _val_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        valid_batches = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                loss_ce = self.ce_criterion(logits, labels)
                loss_dice = self.dice_criterion(logits, labels)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    valid_batches += 1
                all_preds.append(logits.cpu())
                all_targets.append(labels.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        val_loss = total_loss / max(valid_batches, 1)
        val_miou = mean_iou(preds, targets, self.num_classes)
        val_acc = pixel_accuracy(preds, targets)
        return val_loss, val_miou, val_acc

    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {path}")
