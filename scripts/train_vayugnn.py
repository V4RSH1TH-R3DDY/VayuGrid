#!/usr/bin/env python3
"""
Training Script for VayuGNN.

Generates GNN training samples from the simulator using GraphDatasetGenerator,
and trains the VayuGNN model using VayuGNNTrainer.

Usage:
    python scripts/train_vayugnn.py \
        --scenario scenarios/phase1_default.json \
        --epochs 50 \
        --checkpoint-dir outputs/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset

from ai.gnn.dataset import GNNSample, GraphDatasetGenerator
from ai.gnn.vayu_gnn import (
    GNNTrainConfig,
    VayuGNN,
    VayuGNNLoss,
    VayuGNNTrainer,
    _make_dummy_snapshots,
)
from simulator.config import load_simulator_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VayuGNN model")
    parser.add_argument("--scenario", default="scenarios/phase1_default.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a quick smoke test with dummy data")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of simulator episodes for data generation")
    parser.add_argument("--use-pecan", action="store_true",
                        help="Use Pecan Street real load profiles (overrides scenario config)")
    parser.add_argument("--city", default="bangalore",
                        choices=["bangalore", "chennai", "delhi", "hyderabad", "kochi"],
                        help="City for Pecan/NSRDB data")
    return parser.parse_args()


def collate_fn(batch: list[GNNSample]) -> tuple[list[list[Any]], dict[str, torch.Tensor]]:
    snap_seqs = [s.snapshots for s in batch]
    targets = {
        "target_overload": torch.stack([s.target_overload for s in batch]),
        "target_voltage": torch.stack([s.target_voltage for s in batch]),
        "target_risk": torch.stack([s.target_risk for s in batch]),
        "target_duck": torch.stack([s.target_duck for s in batch]),
    }
    return snap_seqs, targets


def train(args: argparse.Namespace) -> VayuGNN:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"[train_vayugnn] Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = VayuGNN().to(device)
    loss_fn = VayuGNNLoss()
    train_cfg = GNNTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_path=str(checkpoint_dir / "vayu_gnn_best.pt"),
    )
    trainer = VayuGNNTrainer(model=model, cfg=train_cfg, loss_fn=loss_fn, device=device)

    if args.dry_run:
        print("Dry-run mode: using dummy data")
        dummy_samples = [
            GNNSample(
                snapshots=_make_dummy_snapshots(),
                target_overload=torch.rand(30),
                target_voltage=torch.rand(30),
                target_risk=torch.rand(1),
                target_duck=torch.rand(96),
            )
            for _ in range(32)
        ]
        train_loader = DataLoader(
            cast(Dataset, dummy_samples), batch_size=4, shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            cast(Dataset, dummy_samples), batch_size=4,
            collate_fn=collate_fn,
        )
    else:
        print("Loading simulator config and generating dataset...")
        print(f"  Episodes: {args.num_episodes}")
        sim_cfg = load_simulator_config(args.scenario)

        if args.use_pecan:
            pecan_path = (
                f"data/processed/pecan_india/{args.city}/2019/"
                f"pecan_wired_{args.city}_2019.csv"
            )
            sim_cfg.load_profile.use_pecan_profiles = True
            sim_cfg.load_profile.pecan_profile_file = pecan_path
            sim_cfg.load_profile.city = args.city
            print(f"  Using Pecan Street data: {pecan_path}")

        generator = GraphDatasetGenerator(sim_cfg)
        train_samples, val_samples, test_samples = generator.generate_dataset(
            num_episodes=args.num_episodes
        )

        print(
            f"  Train: {len(train_samples)} | Val: {len(val_samples)}"
            f" | Test: {len(test_samples)}"
        )

        train_loader = DataLoader(
            cast(Dataset, train_samples), batch_size=4, shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            cast(Dataset, val_samples), batch_size=4,
            collate_fn=collate_fn,
        )

    print(f"Training for {args.epochs} epochs...")
    trainer.fit(train_loader, val_loader)

    print(f"\nTraining complete. Best checkpoint: {train_cfg.checkpoint_path}")

    if not args.dry_run:
        fpr = trainer.compute_false_positive_rate(val_loader)
        print(f"  Final false-positive rate: {fpr*100:.2f}% (gate: <1%)")

    return model


if __name__ == "__main__":
    train(parse_args())
