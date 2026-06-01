#!/usr/bin/env python3
"""Train an Upper/Under body-part head on frozen deep-net features.

The deep net stays the foul detector; this head turns its pooled feature
("context") into a body-part prediction, which the classical CV stage cannot do
well on its own. Reports plain and balanced accuracy plus a confusion matrix so
results are comparable to the majority-class baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


INDEX_TO_BODYPART = {0: "Upper body", 1: "Under body"}


class BodyPartHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--valid-features", required=True)
    parser.add_argument("--output", default="outputs/bodypart_head/bodypart_head.pth")
    parser.add_argument("--report", default="outputs/bodypart_head/bodypart_head_report.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x_train, y_train = _stack(torch.load(args.train_features, map_location="cpu", weights_only=False))
    x_valid, y_valid = _stack(torch.load(args.valid_features, map_location="cpu", weights_only=False))
    feature_dim = x_train.shape[1]

    counts = torch.bincount(y_train, minlength=2).float()
    class_weights = (counts.sum() / (2.0 * counts.clamp(min=1))).to(device)
    print(f"Train: {len(y_train)} (Upper={int(counts[0])}, Under={int(counts[1])}); Valid: {len(y_valid)}")
    print(f"Class weights: {class_weights.tolist()}")

    valid_counts = torch.bincount(y_valid, minlength=2).float()
    majority_acc = float(valid_counts.max() / valid_counts.sum())
    print(f"Valid majority-class accuracy baseline: {majority_acc:.4f}")

    model = BodyPartHead(feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )

    best = {"balanced_accuracy": -1.0}
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            loss = criterion(model(features), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = _evaluate(model, x_valid, y_valid, device)
        if metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = {**metrics, "epoch": epoch + 1}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch={epoch + 1} acc={metrics['accuracy']:.4f} "
                f"balanced_acc={metrics['balanced_accuracy']:.4f}"
            )

    report = {
        "feature_dim": feature_dim,
        "train_samples": int(len(y_train)),
        "valid_samples": int(len(y_valid)),
        "majority_baseline_accuracy": majority_acc,
        "best": best,
    }
    print("\n=== BEST ===")
    print(json.dumps(best, indent=2))
    print(f"Majority baseline: {majority_acc:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feature_dim": feature_dim, "hidden_dim": args.hidden_dim}, out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved head -> {out}\nReport -> {args.report}")
    return 0


def _stack(records):
    features = torch.stack([r["features"].float() for r in records])
    targets = torch.tensor([int(r["bodypart"]) for r in records], dtype=torch.long)
    return features, targets


def _evaluate(model, x_valid, y_valid, device):
    model.eval()
    with torch.no_grad():
        preds = model(x_valid.to(device)).argmax(dim=1).cpu()
    accuracy = float((preds == y_valid).float().mean())
    confusion = [[0, 0], [0, 0]]
    per_class_correct = [0, 0]
    per_class_total = [0, 0]
    for pred, truth in zip(preds.tolist(), y_valid.tolist()):
        confusion[truth][pred] += 1
        per_class_total[truth] += 1
        if pred == truth:
            per_class_correct[truth] += 1
    recalls = [c / t if t else 0.0 for c, t in zip(per_class_correct, per_class_total)]
    balanced = sum(recalls) / len(recalls)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "per_class_recall": {INDEX_TO_BODYPART[i]: recalls[i] for i in range(2)},
        "confusion_truth_to_pred": {
            INDEX_TO_BODYPART[t]: {INDEX_TO_BODYPART[p]: confusion[t][p] for p in range(2)} for t in range(2)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
