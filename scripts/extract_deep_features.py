#!/usr/bin/env python3
"""Cache frozen deep-net features + Bodypart labels for head training.

The deep net (foul detector) stays frozen and provides its pooled multi-view
feature as "context". We pair each action's feature with its ground-truth
Bodypart label (Upper/Under body) so a small head can be trained on top.

Works with the original VARS video backbones, such as mvit_v2_s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VARS model"))

import torchvision.transforms as transforms

from dataset import MultiViewDataset
from model import MVNetwork


BODYPART_TO_INDEX = {"Upper body": 0, "Under body": 1}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="outputs/deep_features")
    parser.add_argument("--splits", nargs="+", default=["Train", "Valid"])
    parser.add_argument("--pre-model", default="mvit_v2_s")
    parser.add_argument("--pooling-type", default="attention")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1, help="Augmented passes over Train (>=1).")
    # Frame sampling (mvit defaults match the VARS interface for 14_model).
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--end-frame", type=int, default=85)
    parser.add_argument("--fps", type=int, default=21)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = MVNetwork(
        net_name=args.pre_model,
        agr_type=args.pooling_type,
    ).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {len(state) - len(unexpected)}/{len(state)} tensors (missing={len(missing)}, unexpected={len(unexpected)}).")
    model.eval()

    transform_aug, transform_model = _transforms(args)

    for split in args.splits:
        annotations = _load_annotations(args.dataset, split)
        repeats = args.repeats if split == "Train" else 1
        use_aug = split == "Train" and repeats > 1
        dataset = MultiViewDataset(
            path=args.dataset,
            start=args.start_frame,
            end=args.end_frame,
            fps=args.fps,
            split=split,
            num_views=5,
            transform=transform_aug if use_aug else None,
            transform_model=transform_model,
        )
        count = len(dataset) if args.max_actions is None else min(args.max_actions, len(dataset))

        records = []
        skipped_label = 0
        skipped_error = 0
        with torch.no_grad():
            for repeat in range(repeats):
                for idx in range(count):
                    try:
                        sample = dataset[idx]
                    except Exception as exc:  # noqa: BLE001 - skip short/corrupt clips
                        skipped_error += 1
                        if skipped_error <= 5:
                            print(f"  {split}: skip index {idx}: {exc}")
                        continue
                    videos, action_id = sample[2], sample[-1]
                    aid = str(action_id)
                    bodypart = annotations.get(aid, {}).get("Bodypart", "").strip()
                    if bodypart not in BODYPART_TO_INDEX:
                        skipped_label += 1
                        continue
                    mvclips = videos.unsqueeze(0).to(device).float()
                    out = model.extract_features(mvclips)
                    features = out[0] if isinstance(out, tuple) else out
                    records.append({
                        "action": aid,
                        "repeat": repeat,
                        "features": features.squeeze(0).detach().cpu(),
                        "bodypart": BODYPART_TO_INDEX[bodypart],
                    })
                    if len(records) % 200 == 0:
                        print(f"  {split}: {len(records)} records...")
                if repeats > 1:
                    print(f"  {split}: finished repeat {repeat + 1}/{repeats} ({len(records)} records)")
        out_path = output / f"{split}_bodypart_features.pt"
        torch.save(records, out_path)
        print(
            f"Wrote {len(records)} records "
            f"({skipped_label} skipped no-label, {skipped_error} skipped load-error) -> {out_path}"
        )
    return 0


def _transforms(args):
    transform_aug = transforms.Compose([
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
    ])

    from torchvision.models.video import (
        MC3_18_Weights,
        MViT_V2_S_Weights,
        R2Plus1D_18_Weights,
        R3D_18_Weights,
        S3D_Weights,
    )

    weights = {
        "r3d_18": R3D_18_Weights,
        "s3d": S3D_Weights,
        "mc3_18": MC3_18_Weights,
        "r2plus1d_18": R2Plus1D_18_Weights,
        "mvit_v2_s": MViT_V2_S_Weights,
    }.get(args.pre_model, R2Plus1D_18_Weights)
    return transform_aug, weights.KINETICS400_V1.transforms()


def _load_annotations(dataset_root: str, split: str) -> dict:
    path = Path(dataset_root) / split / "annotations.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f).get("Actions", {})


if __name__ == "__main__":
    raise SystemExit(main())
