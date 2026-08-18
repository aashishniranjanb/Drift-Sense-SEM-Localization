"""
Training Script for Drift-Sense++ HCR Siamese Structural Re-Ranker

Trains the MultiScaleSiameseEncoder on hard-negative triplets mined from
the synthetic SEM dataset. Uses triplet loss with online hard-negative emphasis.

Usage:
  python train_siamese.py --data-dir data/hcr_train --epochs 40 --batch-size 64
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from siamese_model import MultiScaleSiameseEncoder, TripletLoss, count_parameters


class HardNegativeTripletDataset(Dataset):
    """
    Loads pre-mined triplets: (ref_64, ref_128, pos_64, pos_128, neg_64, neg_128).
    Applies online augmentation: random horizontal flip, small intensity jitter.
    """
    def __init__(self, manifest_path: str, augment: bool = True):
        self.augment = augment
        self.triplets = []

        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.triplets.append({
                    "ref_64": row["ref_64_path"],
                    "ref_128": row["ref_128_path"],
                    "pos_64": row["pos_64_path"],
                    "pos_128": row["pos_128_path"],
                    "neg_64": row["neg_64_path"],
                    "neg_128": row["neg_128_path"],
                    "pos_ncc": float(row["pos_ncc"]),
                    "neg_ncc": float(row["neg_ncc"]),
                })

        print(f"Loaded {len(self.triplets)} triplets from {manifest_path}")

    def __len__(self):
        return len(self.triplets)

    def _load_and_preprocess(self, path: str, target_size: int) -> torch.Tensor:
        """Load numpy patch, normalize to [0,1], apply augmentation."""
        img = np.load(path).astype(np.float32)

        # Ensure correct size
        if img.shape[0] != target_size or img.shape[1] != target_size:
            import cv2
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        p_low, p_high = np.percentile(img, (1, 99))
        if p_high > p_low:
            img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            img = img / 255.0 if img.max() > 1.0 else img

        # Online augmentation
        if self.augment:
            if np.random.random() < 0.5:
                img = np.fliplr(img).copy()
            # Small intensity jitter
            jitter = np.random.uniform(-0.05, 0.05)
            img = np.clip(img + jitter, 0.0, 1.0)
            # Small Gaussian noise
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

        # Convert to tensor: (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        t = self.triplets[idx]

        ref_64 = self._load_and_preprocess(t["ref_64"], 64)
        ref_128 = self._load_and_preprocess(t["ref_128"], 128)
        pos_64 = self._load_and_preprocess(t["pos_64"], 64)
        pos_128 = self._load_and_preprocess(t["pos_128"], 128)
        neg_64 = self._load_and_preprocess(t["neg_64"], 64)
        neg_128 = self._load_and_preprocess(t["neg_128"], 128)

        return {
            "ref_64": ref_64, "ref_128": ref_128,
            "pos_64": pos_64, "pos_128": pos_128,
            "neg_64": neg_64, "neg_128": neg_128,
            "ncc_margin": t["neg_ncc"] - t["pos_ncc"],  # How close the negative is to positive in NCC
        }


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        ref_64 = batch["ref_64"].to(device)
        ref_128 = batch["ref_128"].to(device)
        pos_64 = batch["pos_64"].to(device)
        pos_128 = batch["pos_128"].to(device)
        neg_64 = batch["neg_64"].to(device)
        neg_128 = batch["neg_128"].to(device)

        # Forward pass
        z_ref = model(ref_64, ref_128)
        z_pos = model(pos_64, pos_128)
        z_neg = model(neg_64, neg_128)

        loss = criterion(z_ref, z_pos, z_neg)

        # Accuracy: is positive closer than negative?
        d_pos = 1.0 - torch.nn.functional.cosine_similarity(z_ref, z_pos)
        d_neg = 1.0 - torch.nn.functional.cosine_similarity(z_ref, z_neg)
        correct = (d_pos < d_neg).sum().item()
        total_correct += correct
        total_samples += ref_64.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * ref_64.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            ref_64 = batch["ref_64"].to(device)
            ref_128 = batch["ref_128"].to(device)
            pos_64 = batch["pos_64"].to(device)
            pos_128 = batch["pos_128"].to(device)
            neg_64 = batch["neg_64"].to(device)
            neg_128 = batch["neg_128"].to(device)

            z_ref = model(ref_64, ref_128)
            z_pos = model(pos_64, pos_128)
            z_neg = model(neg_64, neg_128)

            loss = criterion(z_ref, z_pos, z_neg)

            d_pos = 1.0 - torch.nn.functional.cosine_similarity(z_ref, z_pos)
            d_neg = 1.0 - torch.nn.functional.cosine_similarity(z_ref, z_neg)
            correct = (d_pos < d_neg).sum().item()
            total_correct += correct
            total_samples += ref_64.size(0)

            total_loss += loss.item() * ref_64.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Siamese Hard-Negative Re-Ranker")
    parser.add_argument("--data-dir", type=str, default="data/hcr_train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    full_dataset = HardNegativeTripletDataset(manifest_path, augment=True)
    val_dataset = HardNegativeTripletDataset(manifest_path, augment=False)

    # Train/val split
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    indices = list(range(n_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # Model
    model = MultiScaleSiameseEncoder(local_dim=64, context_dim=64).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Training
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n{'='*70}")
    print(f"  TRAINING SIAMESE STRUCTURAL RE-RANKER")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Margin: {args.margin}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {lr_now:.6f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, os.path.join(args.output_dir, "siamese_best.pt"))

        # Also save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }, os.path.join(args.output_dir, "siamese_latest.pt"))

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")
    print(f"  Model saved to: {args.output_dir}/siamese_best.pt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
