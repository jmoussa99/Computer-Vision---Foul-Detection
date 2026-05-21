#!/usr/bin/env python3
"""Extract visual tracking and contact-cue features from SoccerNet-MVFoul clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_foul_detection.dataset import iter_actions
from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig
from cv_foul_detection.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract motion, tracking, and possible-contact visual descriptors.")
    parser.add_argument("--dataset", required=True, help="Root folder containing Train/Valid/Test/Chall.")
    parser.add_argument("--output", default="outputs/cv_features", help="Output directory.")
    parser.add_argument("--splits", nargs="+", default=["Train", "Valid", "Test", "Chall"])
    parser.add_argument("--max-actions", type=int, default=None, help="Limit the number of actions for quick experiments.")
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--visualize", action="store_true", help="Save annotated tracking/contact videos for each sampled clip.")
    parser.add_argument("--contact-distance-ratio", type=float, default=0.08, help="Normalized distance threshold for drawing close interaction/contact lines.")
    parser.add_argument("--contact-motion-p95", type=float, default=8.0, help="Optical-flow p95 threshold for labeling a close interaction as possible contact.")
    args = parser.parse_args()

    output = Path(args.output)
    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            contact_distance_ratio=args.contact_distance_ratio,
            contact_motion_p95=args.contact_motion_p95,
        )
    )
    actions = iter_actions(args.dataset, args.splits)
    processed = 0
    index = []

    for action in actions:
        if args.max_actions is not None and processed >= args.max_actions:
            break
        action_out = output / action.split / f"action_{action.action_id}"
        clip_features = []
        for clip in action.clips:
            overlay = action_out / f"{clip.stem}_overlay.mp4" if args.visualize else None
            clip_features.append(extractor.extract_clip(clip, overlay_path=overlay))

        payload = {
            "split": action.split,
            "action_id": action.action_id,
            "clips": clip_features,
        }
        feature_path = action_out / "features.json"
        write_json(feature_path, payload)
        index.append({"split": action.split, "action_id": action.action_id, "features": str(feature_path)})
        processed += 1

    write_json(output / "index.json", index)
    print(f"Processed {processed} actions. Feature index: {output / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
