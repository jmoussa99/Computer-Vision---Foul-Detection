#!/usr/bin/env python3
"""Extract cached TAdaFormer features for heads-only stage-two training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[1]
VARS_MODEL = ROOT / "VARS model"
sys.path.insert(0, str(VARS_MODEL))

from dataset import MultiViewDataset
from model import MVNetwork


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache TAdaFormer features with repeated random augmentations.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="outputs/tada_features")
    parser.add_argument("--splits", nargs="+", default=["Train", "Valid"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--sample-frames", type=int, default=16)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--input-height", type=int, default=280)
    parser.add_argument("--input-width", type=int, default=490)
    parser.add_argument("--tada-timm-model", default="vit_large_patch14_clip_224.openai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = MVNetwork(
        net_name="tadaformer_l14",
        agr_type="max",
        tada_pretrained=False,
        tada_input_size=(args.input_height, args.input_width),
        tada_timm_model=args.tada_timm_model,
    ).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint, strict=False)
    model.eval()

    transform_aug = transforms.Compose([
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
        transforms.RandomHorizontalFlip(),
    ])
    transform_model = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize((args.input_height, args.input_width), antialias=True),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    for split in args.splits:
        dataset = MultiViewDataset(
            path=args.dataset,
            start=args.start_frame,
            end=125,
            fps=25,
            split=split,
            num_views=5,
            transform=transform_aug,
            transform_model=transform_model,
            sample_frames=args.sample_frames,
            temporal_stride=args.temporal_stride,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        records = []
        with torch.no_grad():
            for repeat in range(args.repeats):
                for targets_offence, targets_action, mvclips, view_ids, action in loader:
                    mvclips = mvclips.to(device).float()
                    view_ids = view_ids.to(device)
                    features, _ = model.extract_features(mvclips, view_ids=view_ids)
                    records.append({
                        "action": action[0],
                        "repeat": repeat,
                        "features": features.squeeze(0).cpu(),
                        "target_offence": targets_offence.squeeze(0).cpu(),
                        "target_action": targets_action.squeeze(0).cpu(),
                    })
        torch.save(records, output / f"{split}_features.pt")
        print(f"Wrote {len(records)} feature records to {output / f'{split}_features.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
