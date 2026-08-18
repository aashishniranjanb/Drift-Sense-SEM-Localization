"""
PACE Group Candidate List Ranking Trainer
Trains ProcessAwareContextEncoder using Group Softmax Cross-Entropy Loss over Top-20 candidates per sample.
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

from pace_model import ProcessAwareContextEncoder, GroupListRankingLoss


class PACEGroupDataset(Dataset):
    def __init__(self, manifest_path: str, augment: bool = True):
        self.augment = augment
        self.groups = []

        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.groups.append({
                    "filepath": row["group_file"],
                    "target_idx": int(row["target_idx"]),
                })

        print(f"Loaded {len(self.groups)} PACE candidate groups from {manifest_path}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g_info = self.groups[idx]
        data = np.load(g_info["filepath"])

        ref_64 = data["ref_64"].astype(np.float32)       # (64, 64)
        ref_128 = data["ref_128"].astype(np.float32)     # (128, 128)
        ref_ovl = data["ref_ovl"].astype(np.float32)     # (4, 32, 32)

        cand_64 = data["cand_64"].astype(np.float32)     # (K, 64, 64)
        cand_128 = data["cand_128"].astype(np.float32)   # (K, 128, 128)
        cand_ovl = data["cand_ovl"].astype(np.float32)   # (K, 4, 32, 32)
        cand_ncc = data["cand_ncc"].astype(np.float32)   # (K,)
        target_idx = int(data["target_idx"])

        if self.augment and np.random.random() < 0.5:
            ref_64 = np.fliplr(ref_64).copy()
            ref_128 = np.fliplr(ref_128).copy()
            cand_64 = np.array([np.fliplr(c).copy() for c in cand_64])
            cand_128 = np.array([np.fliplr(c).copy() for c in cand_128])

        return {
            "ref_64": torch.from_numpy(ref_64).unsqueeze(0),       # (1, 64, 64)
            "ref_128": torch.from_numpy(ref_128).unsqueeze(0),     # (1, 128, 128)
            "ref_ovl": torch.from_numpy(ref_ovl),                  # (4, 32, 32)
            "cand_64": torch.from_numpy(cand_64).unsqueeze(1),     # (K, 1, 64, 64)
            "cand_128": torch.from_numpy(cand_128).unsqueeze(1),   # (K, 1, 128, 128)
            "cand_ovl": torch.from_numpy(cand_ovl),                # (K, 4, 32, 32)
            "cand_ncc": torch.from_numpy(cand_ncc),                # (K,)
            "target_idx": target_idx
        }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0

    for batch in dataloader:
        ref_64 = batch["ref_64"].to(device)
        ref_128 = batch["ref_128"].to(device)
        ref_ovl = batch["ref_ovl"].to(device)

        cand_64 = batch["cand_64"][0].to(device)
        cand_128 = batch["cand_128"][0].to(device)
        cand_ovl = batch["cand_ovl"][0].to(device)
        cand_ncc = batch["cand_ncc"][0].to(device)
        target_idx = batch["target_idx"].to(device)

        z_ref = model.forward_encoder(ref_64, ref_128, ref_ovl)
        z_cands = model.forward_encoder(cand_64, cand_128, cand_ovl)

        scores = model(z_ref, z_cands, cand_ncc)
        loss = criterion(scores, target_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        top1_pred = scores.argmax(dim=1).item()
        top3_preds = scores.topk(min(3, scores.size(1)), dim=1).indices[0].cpu().numpy()

        if top1_pred == target_idx.item():
            total_top1 += 1
        if target_idx.item() in top3_preds:
            total_top3 += 1

        total_samples += 1

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = total_top1 / max(1, total_samples) * 100
    top3_acc = total_top3 / max(1, total_samples) * 100
    return avg_loss, top1_acc, top3_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            ref_64 = batch["ref_64"].to(device)
            ref_128 = batch["ref_128"].to(device)
            ref_ovl = batch["ref_ovl"].to(device)

            cand_64 = batch["cand_64"][0].to(device)
            cand_128 = batch["cand_128"][0].to(device)
            cand_ovl = batch["cand_ovl"][0].to(device)
            cand_ncc = batch["cand_ncc"][0].to(device)
            target_idx = batch["target_idx"].to(device)

            z_ref = model.forward_encoder(ref_64, ref_128, ref_ovl)
            z_cands = model.forward_encoder(cand_64, cand_128, cand_ovl)

            scores = model(z_ref, z_cands, cand_ncc)
            loss = criterion(scores, target_idx)

            total_loss += loss.item()
            top1_pred = scores.argmax(dim=1).item()
            top3_preds = scores.topk(min(3, scores.size(1)), dim=1).indices[0].cpu().numpy()

            if top1_pred == target_idx.item():
                total_top1 += 1
            if target_idx.item() in top3_preds:
                total_top3 += 1

            total_samples += 1

    avg_loss = total_loss / max(1, total_samples)
    top1_acc = total_top1 / max(1, total_samples) * 100
    top3_acc = total_top3 / max(1, total_samples) * 100
    return avg_loss, top1_acc, top3_acc


def main():
    parser = argparse.ArgumentParser(description="Train PACE Group Candidate List Ranker")
    parser.add_argument("--data-dir", type=str, default="data/pace_train")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest '{manifest_path}' not found!")
        return

    full_dataset = PACEGroupDataset(manifest_path, augment=True)
    val_dataset = PACEGroupDataset(manifest_path, augment=False)

    n_total = len(full_dataset)
    n_val = int(n_total * 0.15)
    n_train = n_total - n_val

    indices = list(range(n_total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)

    model = ProcessAwareContextEncoder().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PACE Model Parameters: {n_params:,}")

    criterion = GroupListRankingLoss(temperature=args.temp)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_top1 = 0.0

    print(f"\n{'='*75}")
    print(f"  TRAINING PACE GROUP CANDIDATE LIST RANKER")
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Temperature: {args.temp}")
    print(f"{'='*75}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_top1, tr_top3 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_top1, val_top3 = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Top-1: {tr_top1:5.1f}% | "
              f"Val Loss: {val_loss:.4f} Top-1: {val_top1:5.1f}% Top-3: {val_top3:5.1f}% | {elapsed:.1f}s")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_loss": val_loss,
            }, os.path.join(args.output_dir, "pace_best.pt"))

    print(f"\n{'='*75}")
    print(f"  PACE TRAINING COMPLETE")
    print(f"  Best Validation Top-1 Candidate Ranking Accuracy: {best_val_top1:.1f}%")
    print(f"  Model saved to: {args.output_dir}/pace_best.pt")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
