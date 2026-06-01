#!/usr/bin/env python3
"""Draw red contact boxes only for actions the deep model predicts as fouls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_foul_detection.dataset import iter_actions
from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig
from cv_foul_detection.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate red contact-box overlays with VARS foul predictions.")
    parser.add_argument("--dataset", required=True, help="Root folder containing Train/Valid/Test/Chall or action_* folders.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON produced by VARS model evaluation.")
    parser.add_argument("--output", default="outputs/model_gated_contact_boxes")
    parser.add_argument("--splits", nargs="+", default=["Test"])
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--min-contact-area", type=int, default=450)
    parser.add_argument("--field-top-ratio", type=float, default=0.18)
    parser.add_argument("--contact-distance-ratio", type=float, default=0.08)
    parser.add_argument("--contact-motion-p95", type=float, default=8.0)
    parser.add_argument("--contact-pause-seconds", type=float, default=1.0)
    parser.add_argument("--contact-box-padding", type=int, default=12)
    args = parser.parse_args()

    with Path(args.predictions).open(encoding="utf-8") as f:
        predictions = json.load(f).get("Actions", {})

    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            min_contact_area=args.min_contact_area,
            field_top_ratio=args.field_top_ratio,
            contact_distance_ratio=args.contact_distance_ratio,
            contact_motion_p95=args.contact_motion_p95,
            contact_pause_seconds=args.contact_pause_seconds,
            contact_box_padding=args.contact_box_padding,
        )
    )

    output = Path(args.output)
    index = []
    processed = 0
    for action in iter_actions(args.dataset, args.splits):
        if args.max_actions is not None and processed >= args.max_actions:
            break
        prediction = predictions.get(str(action.action_id), {})
        if prediction.get("Offence", "").lower() in ("", "no offence"):
            continue

        action_out = output / action.split / f"action_{action.action_id}"
        clip_features = []
        for clip in action.clips:
            overlay_path = action_out / f"{clip.stem}_overlay.mp4"
            clip_features.append(extractor.extract_clip(clip, overlay_path=overlay_path))

        feature_path = action_out / "features.json"
        write_json(
            feature_path,
            {
                "split": action.split,
                "action_id": action.action_id,
                "prediction": prediction,
                "clips": clip_features,
            },
        )
        index.append({"split": action.split, "action_id": action.action_id, "features": str(feature_path)})
        processed += 1

    write_json(output / "index.json", index)
    print(f"Rendered {processed} predicted-foul actions. Index: {output / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
