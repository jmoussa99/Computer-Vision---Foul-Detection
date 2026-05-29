#!/usr/bin/env python3
"""Train offence/action heads on cached TAdaFormer features."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class TadaHeads(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.offence = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 4))
        self.action = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 8))

    def forward(self, x):
        x = self.shared(x)
        return self.offence(x), self.action(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classification heads on cached TAdaFormer features.")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--valid-features", required=True)
    parser.add_argument("--output", default="outputs/tada_heads/tada_heads.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_records = torch.load(args.train_features, map_location="cpu")
    valid_records = torch.load(args.valid_features, map_location="cpu")
    x_train, y_off_train, y_act_train = _stack_records(train_records)
    x_valid, y_off_valid, y_act_valid = _stack_records(valid_records)

    device = torch.device(args.device)
    model = TadaHeads(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(x_train, y_off_train, y_act_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        for features, target_offence, target_action in loader:
            features = features.to(device)
            target_offence = target_offence.to(device)
            target_action = target_action.to(device)
            pred_offence, pred_action = model(features)
            loss = criterion(pred_offence, target_offence) + criterion(pred_action, target_action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        off_acc, act_acc = _evaluate(model, x_valid, y_off_valid, y_act_valid, device)
        print(f"epoch={epoch + 1} valid_offence_acc={off_acc:.4f} valid_action_acc={act_acc:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_dim": x_train.shape[1]}, output)
    print(f"Wrote heads to {output}")
    return 0


def _stack_records(records):
    features = torch.stack([record["features"].float() for record in records])
    target_offence = torch.stack([record["target_offence"] for record in records]).argmax(dim=1)
    target_action = torch.stack([record["target_action"] for record in records]).argmax(dim=1)
    return features, target_offence, target_action


def _evaluate(model, features, target_offence, target_action, device):
    model.eval()
    with torch.no_grad():
        pred_offence, pred_action = model(features.to(device))
        off_acc = (pred_offence.argmax(dim=1).cpu() == target_offence).float().mean().item()
        act_acc = (pred_action.argmax(dim=1).cpu() == target_action).float().mean().item()
    return off_acc, act_acc


if __name__ == "__main__":
    raise SystemExit(main())
