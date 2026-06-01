#!/usr/bin/env python3
"""Tune the CV body-part recognizer against the dataset's Bodypart labels.

The expensive work (video decode, optical flow, tracking, pose) is cached once
per action. Threshold configs are then swept cheaply in memory via coordinate
ascent to maximize coarse Upper/Under-body accuracy.

Two phases (the cache is reused automatically when present):
  1. Build cache: analyze each foul clip's frames + run pose on multi-player
     frames, store per-frame tracks/motion/poses.
  2. Sweep: replay peak-contact selection + body-part assignment for many
     configs using the cached data, score vs ground-truth Bodypart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv_foul_detection.bodypart import assign_bodypart
from cv_foul_detection.dataset import iter_actions
from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig, FrameState, MotionDetection
from cv_foul_detection.io import write_json
from cv_foul_detection.pose import PlayerPose, PoseEstimator

import json


CANDIDATES = {
    "field_top_ratio": [0.0, 0.1, 0.18, 0.25, 0.35],
    "contact_distance_ratio": [0.05, 0.08, 0.12, 0.18, 0.25],
    "contact_motion_p95": [0.0, 4.0, 8.0, 12.0, 18.0],
    "min_contact_area": [150, 300, 450, 800, 1500],
    "person_score": [0.4, 0.5, 0.7, 0.85],
    "kp_score": [1.0, 2.0, 3.0, 4.0],
}
BASELINE = {
    "field_top_ratio": 0.18,
    "contact_distance_ratio": 0.08,
    "contact_motion_p95": 8.0,
    "min_contact_area": 450,
    "person_score": 0.7,
    "kp_score": 2.0,
}
STRATEGIES = ["contact", "closest"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", default=["Valid"])
    parser.add_argument("--cache", default="outputs/tune/bodypart_cache.pt")
    parser.add_argument("--output", default="outputs/tune/bodypart_tune_report.json")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cache even if it exists.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-actions", type=int, default=None, help="Limit cached actions (quick tests).")
    # Structural params fixed at cache build time.
    parser.add_argument("--max-frames", type=int, default=126)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--min-track-area", type=int, default=150)
    parser.add_argument("--cache-person-score", type=float, default=0.3)
    parser.add_argument("--rounds", type=int, default=2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cache_path = Path(args.cache)

    if args.rebuild or not cache_path.exists():
        build_cache(args, cache_path)

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    data = _load_in_memory(cache)
    meta = cache["meta"]
    print(f"Loaded {len(data)} cached foul actions (truths: {_truth_counts(data)}).")
    print(f"Structural build params: {meta}")

    base_acc, base_cov, base_asg = _evaluate(BASELINE, "contact", data, meta)
    print(f"Baseline (contact, defaults): acc={base_acc:.4f} coverage={base_cov:.4f} assigned_acc={base_asg:.4f}")

    best = {"accuracy": base_acc, "coverage": base_cov, "assigned_accuracy": base_asg, "strategy": "contact", **BASELINE}
    per_strategy = {}

    for strategy in STRATEGIES:
        config = dict(BASELINE)
        acc, cov, asg = _evaluate(config, strategy, data, meta)
        for _ in range(args.rounds):
            for param, values in CANDIDATES.items():
                if strategy == "closest" and param == "contact_motion_p95":
                    continue
                best_value, best_acc, best_cov, best_asg = config[param], acc, cov, asg
                for value in values:
                    trial = dict(config)
                    trial[param] = value
                    t_acc, t_cov, t_asg = _evaluate(trial, strategy, data, meta)
                    if (t_acc, t_cov) > (best_acc, best_cov):
                        best_value, best_acc, best_cov, best_asg = value, t_acc, t_cov, t_asg
                config[param] = best_value
                acc, cov, asg = best_acc, best_cov, best_asg
        per_strategy[strategy] = {"accuracy": acc, "coverage": cov, "assigned_accuracy": asg, **config}
        print(f"Best [{strategy}]: acc={acc:.4f} coverage={cov:.4f} assigned_acc={asg:.4f} :: {config}")
        if (acc, cov) > (best["accuracy"], best["coverage"]):
            best = {"accuracy": acc, "coverage": cov, "assigned_accuracy": asg, "strategy": strategy, **config}

    report = {
        "samples": len(data),
        "truth_counts": _truth_counts(data),
        "build_meta": meta,
        "baseline": {"accuracy": base_acc, "coverage": base_cov, "assigned_accuracy": base_asg, "strategy": "contact", **BASELINE},
        "per_strategy": per_strategy,
        "best": best,
        "recommended_flags": _recommended_flags(best, meta),
    }
    write_json(args.output, report)
    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2))
    print(f"\nReport: {args.output}")
    print("Recommended recognizer flags:\n  " + " ".join(report["recommended_flags"]))
    return 0


def build_cache(args, cache_path: Path) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            min_track_area=args.min_track_area,
        )
    )
    pose = PoseEstimator(device=device, person_score_threshold=args.cache_person_score)
    annotations = _load_annotations(args.dataset, args.splits)

    actions_cache = {}
    processed = 0
    for action in iter_actions(args.dataset, args.splits):
        if args.max_actions is not None and processed >= args.max_actions:
            break
        truth = annotations.get(action.split, {}).get(str(action.action_id), {}).get("Bodypart", "").strip()
        if truth not in ("Upper body", "Under body"):
            continue
        live_clip = action.clips[0]
        try:
            states, images = extractor.analyze_frames(live_clip, keep_images=True)
            poses = {}
            for state in states:
                if len(state.tracks) >= 2:
                    detected = pose.estimate(images[state.frame_index])
                    poses[state.frame_index] = [
                        {"box": p.box, "score": p.score, "keypoints": p.keypoints} for p in detected
                    ]
        except Exception as exc:  # noqa: BLE001 - skip unreadable/corrupt clips
            print(f"  skip {action.split}/{action.action_id}: {exc}")
            continue
        actions_cache[f"{action.split}/{action.action_id}"] = {
            "truth": truth,
            "states": [_serialize_state(s) for s in states],
            "poses": poses,
        }
        processed += 1
        if processed % 25 == 0:
            print(f"  cached {processed} foul actions...")

    cache = {
        "meta": {
            "max_frames": args.max_frames,
            "frame_stride": args.frame_stride,
            "resize_width": args.resize_width,
            "min_track_area": args.min_track_area,
        },
        "actions": actions_cache,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Built cache with {len(actions_cache)} actions -> {cache_path}")


def _serialize_state(state: FrameState) -> dict:
    return {
        "frame_index": state.frame_index,
        "height": state.height,
        "width": state.width,
        "motion_p95": state.motion_p95,
        "tracks": {tid: (det.centroid, det.bbox, det.area) for tid, det in state.tracks.items()},
    }


def _load_in_memory(cache: dict) -> list[dict]:
    data = []
    for record in cache["actions"].values():
        states = [
            FrameState(
                frame_index=s["frame_index"],
                height=s["height"],
                width=s["width"],
                motion_p95=s["motion_p95"],
                tracks={
                    tid: MotionDetection(centroid=tuple(c), bbox=tuple(b), area=float(a))
                    for tid, (c, b, a) in s["tracks"].items()
                },
            )
            for s in record["states"]
        ]
        poses = {
            int(fi): [
                PlayerPose(box=tuple(p["box"]), score=float(p["score"]), keypoints=np.asarray(p["keypoints"], dtype=np.float32))
                for p in players
            ]
            for fi, players in record["poses"].items()
        }
        data.append({"truth": record["truth"], "states": states, "poses": poses})
    return data


def _evaluate(config: dict, strategy: str, data: list[dict], meta: dict) -> tuple[float, float, float]:
    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=meta["max_frames"],
            frame_stride=meta["frame_stride"],
            resize_width=meta["resize_width"],
            min_track_area=meta["min_track_area"],
            min_contact_area=config["min_contact_area"],
            field_top_ratio=config["field_top_ratio"],
            contact_distance_ratio=config["contact_distance_ratio"],
            contact_motion_p95=config["contact_motion_p95"],
            peak_strategy=strategy,
        )
    )
    person_score = config["person_score"]
    kp_score = config["kp_score"]
    total = len(data)
    correct = 0
    assigned = 0
    for record in data:
        selection = extractor.select_peak_state(record["states"])
        point = _selection_point(selection)
        if point is None:
            continue
        players = [p for p in record["poses"].get(selection.frame_index, []) if p.score >= person_score]
        if not players:
            continue
        result = assign_bodypart(point, players, keypoint_score_threshold=kp_score)
        if result is None:
            continue
        assigned += 1
        if result.coarse == record["truth"]:
            correct += 1
    accuracy = correct / total if total else 0.0
    coverage = assigned / total if total else 0.0
    assigned_acc = correct / assigned if assigned else 0.0
    return accuracy, coverage, assigned_acc


def _selection_point(selection):
    if selection is None:
        return None
    if selection.contact_point is not None:
        return selection.contact_point
    if selection.contact_box is not None:
        x1, y1, x2, y2 = selection.contact_box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return None


def _recommended_flags(best: dict, meta: dict) -> list[str]:
    return [
        f"--peak-strategy {best['strategy']}",
        f"--field-top-ratio {best['field_top_ratio']}",
        f"--contact-distance-ratio {best['contact_distance_ratio']}",
        f"--contact-motion-p95 {best['contact_motion_p95']}",
        f"--min-contact-area {best['min_contact_area']}",
        f"--person-score-threshold {best['person_score']}",
        f"--keypoint-score-threshold {best['kp_score']}",
        f"--max-frames {meta['max_frames']}",
        f"--frame-stride {meta['frame_stride']}",
    ]


def _truth_counts(data: list[dict]) -> dict:
    counts = {"Upper body": 0, "Under body": 0}
    for record in data:
        counts[record["truth"]] = counts.get(record["truth"], 0) + 1
    return counts


def _load_annotations(dataset_root: str, splits: list[str]) -> dict:
    result = {}
    for split in splits:
        path = Path(dataset_root) / split / "annotations.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                result[split] = json.load(f).get("Actions", {})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
