#!/usr/bin/env python3
"""Recognize WHERE a foul lands on the body using CV, with the deep net as context.

The deep net is the foul DETECTOR (offence/severity). This script runs only on
actions the deep net flags as fouls and uses classical CV (motion/contact) plus
pose estimation to localize the contact to a body region (leg, arm, back, ...).

Foul predictions can be supplied as a precomputed JSON (``--predictions``) or
produced inline on the GPU from a checkpoint (``--weights``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cv_foul_detection.bodypart import COARSE_BY_FINE, assign_bodypart
from cv_foul_detection.contact import find_pose_contact
from cv_foul_detection.deep_saliency import compute_occlusion_saliency, crop_geometry, peak_uv
from cv_foul_detection.dataset import iter_actions
from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig, PeakContact
from cv_foul_detection.io import write_json
from cv_foul_detection.pose import COCO_SKELETON, PlayerPose, PoseEstimator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset root with Train/Valid/Test/Chall folders.")
    parser.add_argument("--splits", nargs="+", default=["Valid"])
    parser.add_argument("--output", default="outputs/foul_bodypart")
    parser.add_argument("--device", default=None, help="Torch device for pose/deep net (default: cuda if available).")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--render-video", action="store_true", help="Also write the full motion red-box overlay video.")
    parser.add_argument("--eval", action="store_true", help="Score coarse body-part vs annotation Bodypart.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="Deep-net prediction JSON (predicitions_*.json).")
    source.add_argument("--weights", help="Deep-net checkpoint to run inline on the GPU for foul detection.")
    parser.add_argument("--detect-limit", type=int, default=None, help="Cap actions scored by the inline deep net (quick tests).")
    parser.add_argument(
        "--bodypart-head",
        default="",
        help="Trained deep body-part head (scripts/train_bodypart_head.py). Requires --weights; "
        "predicts coarse Upper/Under body from the deep feature.",
    )

    # Contact / motion tuning (mirrors FeatureConfig).
    # Defaults below are tuned on the Valid split vs ground-truth Bodypart
    # (see scripts/tune_bodypart.py).
    parser.add_argument("--max-frames", type=int, default=126)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--min-contact-area", type=int, default=450)
    parser.add_argument("--field-top-ratio", type=float, default=0.35)
    parser.add_argument("--contact-distance-ratio", type=float, default=0.18)
    parser.add_argument("--contact-motion-p95", type=float, default=18.0)
    parser.add_argument("--contact-box-padding", type=int, default=12)
    parser.add_argument("--peak-strategy", default="contact", choices=["contact", "closest"])

    # Where the red box comes from.
    parser.add_argument(
        "--contact-source",
        default="pose",
        choices=["pose", "motion", "deep"],
        help="pose: closest two-player skeleton contact (default); motion: legacy MOG2; "
        "deep: where the deep net looks (occlusion saliency), snapped to the nearest player. Requires --weights.",
    )
    parser.add_argument("--saliency-grid", type=int, default=7, help="deep: occlusion grid resolution.")
    parser.add_argument("--saliency-resize-shorter", type=int, default=256, help="deep: backbone resize shorter side.")
    parser.add_argument("--saliency-alpha", type=float, default=0.45, help="deep: heatmap overlay opacity.")
    parser.add_argument("--contact-box-scale", type=float, default=0.4, help="Pose contact box size vs player height.")
    parser.add_argument("--max-pose-frames", type=int, default=32, help="Frames posed per clip when locating contact.")
    parser.add_argument("--contact-center-frac", type=float, default=0.6, help="Central time window searched for contact.")
    parser.add_argument(
        "--contact-motion-weight",
        type=float,
        default=0.6,
        help="How strongly local motion (a collision) biases the contact pair [0..1]; 0 disables it.",
    )
    parser.add_argument(
        "--contact-center-weight",
        type=float,
        default=0.3,
        help="How strongly frame-centeredness biases the contact pair [0..1]; 0 disables it.",
    )
    parser.add_argument(
        "--require-two-players",
        action="store_true",
        help="Only keep actions with a clear two-player contact (pose). Skips the action instead of "
        "falling back to the single-blob motion box.",
    )
    parser.add_argument(
        "--max-contact-distance-ratio",
        type=float,
        default=0.35,
        help="With --require-two-players, reject contacts where the two closest keypoints are farther "
        "apart than this fraction of player height (0 disables the gate; e.g. 0.25 = must nearly touch).",
    )

    # Pose tuning (defaults from scripts/tune_bodypart.py --mode pose on Valid).
    parser.add_argument("--person-score-threshold", type=float, default=0.6)
    parser.add_argument("--keypoint-score-threshold", type=float, default=2.0)

    # Deep-net inline config (only used with --weights).
    parser.add_argument("--pre-model", default="mvit_v2_s")
    parser.add_argument("--pooling-type", default="attention")
    parser.add_argument("--start-frame", type=int, default=65)
    parser.add_argument("--end-frame", type=int, default=85)
    parser.add_argument("--fps", type=int, default=21)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    device = _resolve_device(args.device)
    predictions = _load_predictions(args, device)

    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            min_contact_area=args.min_contact_area,
            field_top_ratio=args.field_top_ratio,
            contact_distance_ratio=args.contact_distance_ratio,
            contact_motion_p95=args.contact_motion_p95,
            contact_box_padding=args.contact_box_padding,
            peak_strategy=args.peak_strategy,
        )
    )
    pose = PoseEstimator(
        device=device,
        person_score_threshold=args.person_score_threshold,
        keypoint_score_threshold=args.keypoint_score_threshold,
    )

    output = Path(args.output)
    index = []
    eval_pairs: list[tuple[str, str]] = []
    eval_pairs_deep: list[tuple[str, str]] = []
    annotations = _load_all_annotations(args.dataset, args.splits) if args.eval else {}
    processed = 0
    skipped_no_contact = 0

    for action in iter_actions(args.dataset, args.splits):
        if args.max_actions is not None and processed >= args.max_actions:
            break
        prediction = predictions.get(str(action.action_id), {})
        if prediction.get("Offence", "").strip().lower() in ("", "no offence"):
            continue

        live_clip = action.clips[0]
        located = _locate_contact(live_clip, extractor, pose, args, prediction)
        if located is None:
            if args.require_two_players:
                skipped_no_contact += 1
                print(f"[{action.split}] action_{action.action_id}: skipped (no clear two-player contact)")
            continue

        contact_point = located["contact_point"]
        players = located["players"]
        assignment = (
            assign_bodypart(contact_point, players, keypoint_score_threshold=args.keypoint_score_threshold)
            if contact_point is not None and players
            else None
        )

        action_out = output / action.split / f"action_{action.action_id}"
        action_out.mkdir(parents=True, exist_ok=True)

        context_text = _context_text(prediction)
        overlay = _draw_overlay(
            located["frame"],
            players,
            located["contact_box"],
            contact_point,
            assignment,
            context_text,
            args.keypoint_score_threshold,
            prediction.get("BodypartDeep"),
            saliency=located.get("saliency"),
            saliency_rect=located.get("saliency_rect"),
            saliency_alpha=args.saliency_alpha,
        )
        overlay_image = action_out / "contact_bodypart.png"
        cv2.imwrite(str(overlay_image), overlay)

        video_path = None
        if args.render_video:
            video_path = action_out / f"{live_clip.stem}_overlay.mp4"
            extractor.extract_clip(live_clip, overlay_path=video_path)

        record = {
            "split": action.split,
            "action_id": action.action_id,
            "clip": str(live_clip),
            "deep_prediction": {k: v for k, v in prediction.items() if k not in ("Saliency", "SaliencyCrop")},
            "contact_source": args.contact_source,
            "contact_frame_index": located["frame_index"],
            "contact_box": located["contact_box"],
            "players_detected": len(players),
            "bodypart": _assignment_dict(assignment),
            "bodypart_deep": prediction.get("BodypartDeep"),
            "overlay_image": str(overlay_image),
            "overlay_video": str(video_path) if video_path else None,
        }
        write_json(action_out / "bodypart.json", record)
        index.append(record)
        processed += 1

        if args.eval:
            truth = annotations.get(action.split, {}).get(str(action.action_id), {}).get("Bodypart", "").strip()
            if truth in ("Upper body", "Under body"):
                if assignment is not None:
                    eval_pairs.append((assignment.coarse, truth))
                if prediction.get("BodypartDeep"):
                    eval_pairs_deep.append((prediction["BodypartDeep"], truth))

        print(
            f"[{action.split}] action_{action.action_id}: "
            f"{_assignment_dict(assignment)['fine'] if assignment else 'unknown'} "
            f"({len(players)} players, source={args.contact_source})"
        )

    write_json(output / "index.json", index)
    print(f"Wrote {processed} body-part records. Index: {output / 'index.json'}")
    if args.require_two_players:
        print(f"Skipped {skipped_no_contact} actions without a clear two-player contact.")

    if args.eval:
        report = _eval_report(eval_pairs)
        report["deep_head"] = _eval_report(eval_pairs_deep) if eval_pairs_deep else None
        write_json(output / "bodypart_eval.json", report)
        print(
            f"CV pose coarse accuracy vs Bodypart: {report['accuracy']:.4f} "
            f"on {report['samples']} labelled actions."
        )
        if report["deep_head"]:
            print(
                f"Deep head coarse accuracy vs Bodypart: {report['deep_head']['accuracy']:.4f} "
                f"on {report['deep_head']['samples']} labelled actions."
            )
    return 0


def _resolve_device(device: str | None) -> str:
    import torch

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _locate_contact(live_clip, extractor, pose, args, prediction=None) -> dict | None:
    """Return the contact frame, point, box and posed players.

    Pose source: the place where two players' skeletons are closest. Deep
    source: where the deep net looks (occlusion saliency) snapped to the nearest
    player. Motion source: the legacy MOG2 peak. Pose/deep fall back to motion if
    they cannot localize.
    """
    if args.contact_source == "deep":
        located = _locate_contact_deep(live_clip, extractor, pose, args, prediction or {})
        if located is not None:
            return located
        # Fall through to motion if saliency is unavailable.

    if args.contact_source == "pose":
        frames = extractor.read_frames(live_clip)
        contact = find_pose_contact(
            frames,
            pose,
            keypoint_score_threshold=args.keypoint_score_threshold,
            box_scale=args.contact_box_scale,
            max_pose_frames=args.max_pose_frames,
            center_frac=args.contact_center_frac,
            max_distance_ratio=args.max_contact_distance_ratio if args.require_two_players else 0.0,
            motion_weight=args.contact_motion_weight,
            center_weight=args.contact_center_weight,
        )
        if contact is not None:
            return {
                "frame": contact.frame,
                "frame_index": contact.frame_index,
                "contact_point": contact.point,
                "contact_box": list(contact.box),
                "players": contact.players,
            }
        if args.require_two_players:
            # No clean two-player contact: drop the action rather than fall back.
            return None

    peak = extractor.extract_peak_contact(live_clip)
    if peak is None:
        return None
    point = peak.contact_point
    if point is None and peak.contact_box is not None:
        x1, y1, x2, y2 = peak.contact_box
        point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return {
        "frame": peak.frame,
        "frame_index": peak.frame_index,
        "contact_point": point,
        "contact_box": list(peak.contact_box) if peak.contact_box is not None else None,
        "players": pose.estimate(peak.frame),
    }


def _locate_contact_deep(live_clip, extractor, pose, args, prediction: dict) -> dict | None:
    saliency = prediction.get("Saliency")
    if not saliency:
        return None
    salmap = np.asarray(saliency, dtype=np.float32)

    frames = extractor.read_frames(live_clip)
    if not frames:
        return None
    mid = frames[len(frames) // 2]
    frame_index, frame = mid
    display_h, display_w = frame.shape[:2]

    orig_h, orig_w = _original_frame_size(live_clip)
    geometry = crop_geometry(args.pre_model, orig_h, orig_w, resize_shorter=args.saliency_resize_shorter)
    scale_x = display_w / float(orig_w)
    scale_y = display_h / float(orig_h)

    u, v = peak_uv(salmap)
    px, py = geometry.uv_to_point(u, v)
    peak_point = (px * scale_x, py * scale_y)

    players = pose.estimate(frame)
    contact_point = _snap_to_player(peak_point, players, args.keypoint_score_threshold) or peak_point

    half = max(0.18 * _frame_player_height(players), 24.0)
    contact_box = [
        int(max(contact_point[0] - half, 0)),
        int(max(contact_point[1] - half, 0)),
        int(min(contact_point[0] + half, display_w - 1)),
        int(min(contact_point[1] + half, display_h - 1)),
    ]

    crop_box = (
        int(geometry.x1 * scale_x),
        int(geometry.y1 * scale_y),
        int(geometry.x2 * scale_x),
        int(geometry.y2 * scale_y),
    )
    return {
        "frame": frame,
        "frame_index": frame_index,
        "contact_point": contact_point,
        "contact_box": contact_box,
        "players": players,
        "saliency": salmap,
        "saliency_rect": crop_box,
    }


def _snap_to_player(point, players: list[PlayerPose], kp_threshold: float):
    best = None
    best_dist = float("inf")
    for player in players:
        kp = player.keypoints
        for j in range(kp.shape[0]):
            if kp[j, 2] < kp_threshold:
                continue
            dist = float(np.hypot(kp[j, 0] - point[0], kp[j, 1] - point[1]))
            if dist < best_dist:
                best_dist = dist
                best = (float(kp[j, 0]), float(kp[j, 1]))
    return best


def _frame_player_height(players: list[PlayerPose]) -> float:
    heights = [p.box[3] - p.box[1] for p in players]
    return max(float(np.median(heights)) if heights else 0.0, 1.0)


def _original_frame_size(clip_path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(clip_path))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if width <= 0 or height <= 0:
        return (720, 1280)
    return (int(height), int(width))


def _assignment_dict(assignment) -> dict:
    if assignment is None:
        return {"fine": "unknown", "coarse": "unknown", "distance": None, "confidence": None, "player_index": None}
    return {
        "fine": assignment.fine,
        "coarse": assignment.coarse,
        "distance": round(assignment.distance, 2),
        "confidence": round(assignment.confidence, 4),
        "player_index": assignment.player_index,
    }


def _context_text(prediction: dict) -> str:
    offence = prediction.get("Offence", "Offence")
    severity = prediction.get("Severity", "")
    action_class = prediction.get("Action class", "")
    parts = [offence]
    if severity:
        parts.append(f"sev {severity}")
    if action_class:
        parts.append(action_class)
    return " | ".join(p for p in parts if p)


def _draw_overlay(
    frame: np.ndarray,
    players: list[PlayerPose],
    contact_box,
    contact_point,
    assignment,
    context_text: str,
    kp_threshold: float,
    bodypart_deep: str | None = None,
    saliency=None,
    saliency_rect=None,
    saliency_alpha: float = 0.45,
) -> np.ndarray:
    overlay = frame.copy()
    if saliency is not None and saliency_rect is not None:
        overlay = _blend_saliency(overlay, saliency, saliency_rect, saliency_alpha)
    highlight = assignment.player_index if assignment is not None else -1

    for idx, player in enumerate(players):
        colour = (0, 215, 255) if idx == highlight else (160, 160, 160)
        kp = player.keypoints
        for a, b in COCO_SKELETON:
            if kp[a, 2] >= kp_threshold and kp[b, 2] >= kp_threshold:
                pa = (int(kp[a, 0]), int(kp[a, 1]))
                pb = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(overlay, pa, pb, colour, 2)
        for j in range(kp.shape[0]):
            if kp[j, 2] >= kp_threshold:
                cv2.circle(overlay, (int(kp[j, 0]), int(kp[j, 1])), 3, colour, -1)

    if contact_box is not None:
        x1, y1, x2, y2 = contact_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if contact_point is not None:
        cx, cy = int(contact_point[0]), int(contact_point[1])
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

    fine = assignment.fine if assignment is not None else "unknown"
    coarse = assignment.coarse if assignment is not None else "unknown"
    _put_label(overlay, f"Foul contact: {fine} ({coarse})", 24)
    if context_text:
        _put_label(overlay, f"Deep net: {context_text}", 52)
    if bodypart_deep:
        _put_label(overlay, f"Body part (deep head): {bodypart_deep}", 80)
    return overlay


def _blend_saliency(image: np.ndarray, saliency: np.ndarray, rect, alpha: float) -> np.ndarray:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, image.shape[1] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y1 = max(0, min(y1, image.shape[0] - 1))
    y2 = max(y1 + 1, min(y2, image.shape[0]))

    sal = saliency.astype(np.float32)
    sal = np.clip(sal, 0.0, None)
    if sal.max() > 1e-9:
        sal = sal / sal.max()
    sal_u8 = (sal * 255.0).astype(np.uint8)
    heat = cv2.resize(sal_u8, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    heat_colour = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    region = image[y1:y2, x1:x2]
    weight = (heat.astype(np.float32) / 255.0 * alpha)[:, :, None]
    blended = region * (1.0 - weight) + heat_colour * weight
    image[y1:y2, x1:x2] = blended.astype(np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return image


def _put_label(image: np.ndarray, text: str, y: int) -> None:
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _load_predictions(args, device: str) -> dict:
    if args.predictions:
        with Path(args.predictions).open(encoding="utf-8") as f:
            return json.load(f).get("Actions", {})
    return _predict_with_weights(args, device)


def _predict_with_weights(args, device: str) -> dict:
    import torch

    sys.path.insert(0, str(ROOT / "VARS model"))
    from config.classes import INVERSE_EVENT_DICTIONARY
    from dataset import MultiViewDataset
    from model import MVNetwork

    torch_device = torch.device(device)
    model = MVNetwork(
        net_name=args.pre_model,
        agr_type=args.pooling_type,
    ).to(torch_device)
    checkpoint = torch.load(args.weights, map_location=torch_device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    print(f"Loaded {loaded}/{len(state)} checkpoint tensors (missing={len(missing)}, unexpected={len(unexpected)}).")

    bodypart_head = _load_bodypart_head(args.bodypart_head, torch_device) if args.bodypart_head else None
    index_to_bodypart = {0: "Upper body", 1: "Under body"}

    transform_model = _backbone_transform(args)
    model.eval()

    predictions: dict[str, dict] = {}
    severity_map = {0: ("No offence", ""), 1: ("Offence", "1.0"), 2: ("Offence", "3.0"), 3: ("Offence", "5.0")}
    for split in args.splits:
        dataset = MultiViewDataset(
            path=args.dataset,
            start=args.start_frame,
            end=args.end_frame,
            fps=args.fps,
            split=split,
            num_views=5,
            transform=None,
            transform_model=transform_model,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                if args.detect_limit is not None and len(predictions) >= args.detect_limit:
                    break
                mvclips, action_id = batch[2], batch[-1]
                mvclips = mvclips.to(torch_device).float()
                offence_logits, action_logits, _ = model(mvclips, None)
                sev = int(torch.argmax(offence_logits.detach().cpu(), dim=-1).item())
                act = int(torch.argmax(action_logits.detach().cpu(), dim=-1).item())
                offence, severity = severity_map[sev]
                entry = {
                    "Action class": INVERSE_EVENT_DICTIONARY["action_class"][act],
                    "Offence": offence,
                    "Severity": severity,
                }
                if bodypart_head is not None:
                    out = model.extract_features(mvclips)
                    features = out[0] if isinstance(out, tuple) else out
                    bp = int(torch.argmax(bodypart_head(features).detach().cpu(), dim=-1).item())
                    entry["BodypartDeep"] = index_to_bodypart[bp]
                if args.contact_source == "deep" and offence != "No offence":
                    salmap = compute_occlusion_saliency(model, mvclips, view=0, grid=args.saliency_grid)
                    entry["Saliency"] = salmap.tolist()
                    entry["SaliencyCrop"] = int(mvclips.shape[-1])
                predictions[str(action_id[0])] = entry
    return predictions


def _load_bodypart_head(path: str, device):
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_bodypart_head import BodyPartHead

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head = BodyPartHead(checkpoint["feature_dim"], hidden_dim=checkpoint.get("hidden_dim", 256))
    head.load_state_dict(checkpoint["state_dict"])
    head.eval().to(device)
    return head


def _backbone_transform(args):
    """Return the preprocessing transform matching the chosen VARS backbone.

    Mirrors the preprocessing in ``VARS model/main.py`` so a checkpoint runs
    with the same transforms it was trained on.
    """

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
    return weights.KINETICS400_V1.transforms()


def _load_all_annotations(dataset_root: str, splits: list[str]) -> dict:
    result: dict[str, dict] = {}
    for split in splits:
        path = Path(dataset_root) / split / "annotations.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                result[split] = json.load(f).get("Actions", {})
    return result


def _eval_report(pairs: list[tuple[str, str]]) -> dict:
    correct = sum(1 for pred, truth in pairs if pred == truth)
    total = len(pairs)
    confusion: dict[str, dict[str, int]] = {}
    for pred, truth in pairs:
        confusion.setdefault(truth, Counter())[pred] += 1
    confusion_serializable = {truth: dict(counts) for truth, counts in confusion.items()}
    return {
        "samples": total,
        "accuracy": (correct / total) if total else 0.0,
        "labels": ["Upper body", "Under body"],
        "confusion_truth_to_pred": confusion_serializable,
    }


if __name__ == "__main__":
    raise SystemExit(main())
