"""Locate the foul contact point from player poses.

Instead of trusting raw motion blobs, this finds the moment and place where two
players' skeletons are closest -- i.e. where bodily contact actually occurs --
and returns a tight box centered there. The deep net has already decided the
clip is a foul, and clips are centered on the foul, so the closest player
approach (optionally within the central time window) is the contact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .pose import PlayerPose, PoseEstimator


@dataclass
class PoseContact:
    frame_index: int
    frame: np.ndarray
    point: tuple[float, float]
    box: tuple[int, int, int, int]
    distance: float
    players: list[PlayerPose]
    player_indices: tuple[int, int]


def find_pose_contact(
    frames: list[tuple[int, np.ndarray]],
    pose: PoseEstimator,
    keypoint_score_threshold: float = 2.0,
    box_scale: float = 0.4,
    max_pose_frames: int = 24,
    center_frac: float = 0.6,
    max_distance_ratio: float = 0.0,
    motion_weight: float = 0.6,
    center_weight: float = 0.3,
    saliency_fn: Callable[[float, float], float] | None = None,
    saliency_weight: float = 0.0,
) -> PoseContact | None:
    """Return the most foul-like two-player contact across the frames.

    Each candidate player pair is scored by how *close* their skeletons are,
    how much *motion* (a collision) is happening at the contact point, and how
    *central* it is in the frame -- so the box lands on the actual foul rather
    than on any two nearby bodies. ``center_frac`` restricts the search to the
    central time window; ``max_distance_ratio`` (>0) rejects pairs that never
    actually touch. ``motion_weight``/``center_weight`` in [0, 1] control how
    strongly motion/centeredness influence the choice (0 disables that cue).

    ``saliency_fn`` (hybrid mode) maps a frame point to a deep-saliency value in
    [0, 1]; with ``saliency_weight`` > 0 the search is biased toward where the
    deep net looks, so the tightest contact *inside the foul region* wins.

    Returns ``None`` if no frame has two posed players touching closely enough.
    """
    if not frames:
        return None

    motion_by_index = _motion_maps(frames)
    window = _center_window(frames, center_frac)
    window = _subsample(window, max_pose_frames)

    best: PoseContact | None = None
    best_score = -1.0
    for frame_index, frame in window:
        players = pose.estimate(frame)
        candidate = _score_contacts(
            players,
            keypoint_score_threshold,
            frame.shape,
            motion_by_index.get(frame_index),
            box_scale,
            max_distance_ratio,
            motion_weight,
            center_weight,
            saliency_fn,
            saliency_weight,
        )
        if candidate is None:
            continue
        score, distance, point, (i, j) = candidate
        if score > best_score:
            box = _contact_box(point, players[i], players[j], box_scale, frame.shape)
            best_score = score
            best = PoseContact(
                frame_index=frame_index,
                frame=frame,
                point=point,
                box=box,
                distance=distance,
                players=players,
                player_indices=(i, j),
            )
    return best


def find_pose_contact_frame(
    frame: np.ndarray,
    frame_index: int,
    players: list[PlayerPose],
    keypoint_score_threshold: float = 2.0,
    box_scale: float = 0.4,
    max_distance_ratio: float = 0.0,
    center_weight: float = 0.3,
    saliency_fn: Callable[[float, float], float] | None = None,
    saliency_weight: float = 0.0,
) -> PoseContact | None:
    """Return the best two-player contact in one already-posed frame."""
    if len(players) < 2:
        return None
    candidate = _score_contacts(
        players,
        keypoint_score_threshold,
        frame.shape,
        None,
        box_scale,
        max_distance_ratio,
        motion_weight=0.0,
        center_weight=center_weight,
        saliency_fn=saliency_fn,
        saliency_weight=saliency_weight,
    )
    if candidate is None:
        return None
    score, distance, point, (i, j) = candidate
    return PoseContact(
        frame_index=frame_index,
        frame=frame,
        point=point,
        box=_contact_box(point, players[i], players[j], box_scale, frame.shape),
        distance=distance,
        players=players,
        player_indices=(i, j),
    )


def _motion_maps(frames: list[tuple[int, np.ndarray]]) -> dict[int, np.ndarray | None]:
    """Per-frame motion magnitude (abs diff with the previous sampled frame)."""
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for _, frame in frames]
    motion: dict[int, np.ndarray | None] = {}
    for k, (frame_index, _) in enumerate(frames):
        motion[frame_index] = None if k == 0 else cv2.absdiff(grays[k], grays[k - 1])
    return motion


def _score_contacts(
    players: list[PlayerPose],
    threshold: float,
    shape,
    motion_map: np.ndarray | None,
    box_scale: float,
    max_distance_ratio: float,
    motion_weight: float,
    center_weight: float,
    saliency_fn: Callable[[float, float], float] | None = None,
    saliency_weight: float = 0.0,
):
    """Best-scoring player pair in one frame, or ``None``."""
    points = [_visible_points(p, threshold) for p in players]
    height, width = shape[:2]
    cx, cy = width / 2.0, height / 2.0
    half_diag = max(0.5 * float(np.hypot(width, height)), 1.0)
    mp95 = float(np.percentile(motion_map, 95)) if motion_map is not None else 0.0

    best = None
    for i in range(len(players)):
        if len(points[i]) == 0:
            continue
        for j in range(i + 1, len(players)):
            if len(points[j]) == 0:
                continue
            diff = points[i][:, None, :] - points[j][None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            flat = int(np.argmin(dists))
            a, b = divmod(flat, dists.shape[1])
            distance = float(dists[a, b])
            avg_height = _pair_height(players[i], players[j])
            if max_distance_ratio > 0.0 and distance > max_distance_ratio * avg_height:
                continue
            point = (
                float((points[i][a, 0] + points[j][b, 0]) / 2.0),
                float((points[i][a, 1] + points[j][b, 1]) / 2.0),
            )
            closeness = float(np.exp(-(distance / avg_height) / 0.3))
            motion_score = _local_motion(motion_map, mp95, point, box_scale, avg_height, width, height)
            motion_factor = (1.0 - motion_weight) + motion_weight * motion_score
            center_score = 1.0 - min(float(np.hypot(point[0] - cx, point[1] - cy)) / half_diag, 1.0)
            center_factor = (1.0 - center_weight) + center_weight * center_score
            score = closeness * motion_factor * center_factor
            if saliency_weight > 0.0 and saliency_fn is not None:
                sal = float(np.clip(saliency_fn(point[0], point[1]), 0.0, 1.0))
                score *= (1.0 - saliency_weight) + saliency_weight * sal
            if best is None or score > best[0]:
                best = (score, distance, point, (i, j))
    return best


def _local_motion(motion_map, mp95, point, box_scale, avg_height, width, height) -> float:
    if motion_map is None or mp95 <= 1e-6:
        return 0.0
    half = max(box_scale * 0.5 * avg_height, 16.0)
    x1 = int(max(point[0] - half, 0))
    x2 = int(min(point[0] + half, width - 1))
    y1 = int(max(point[1] - half, 0))
    y2 = int(min(point[1] + half, height - 1))
    region = motion_map[y1 : y2 + 1, x1 : x2 + 1]
    if region.size == 0:
        return 0.0
    return min(float(region.mean()) / mp95, 1.0)


def _pair_height(player_a: PlayerPose, player_b: PlayerPose) -> float:
    height_a = player_a.box[3] - player_a.box[1]
    height_b = player_b.box[3] - player_b.box[1]
    return max((height_a + height_b) / 2.0, 1.0)


def _center_window(frames, center_frac):
    if center_frac >= 1.0:
        return frames
    n = len(frames)
    margin = int(n * (1.0 - center_frac) / 2.0)
    window = frames[margin : n - margin]
    return window if window else frames


def _subsample(frames, max_frames):
    if len(frames) <= max_frames:
        return frames
    idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int)
    return [frames[i] for i in idxs]


def _visible_points(player: PlayerPose, threshold: float) -> np.ndarray:
    kp = player.keypoints
    return kp[kp[:, 2] >= threshold][:, :2]


def _closest_pair(players: list[PlayerPose], threshold: float):
    points = [_visible_points(p, threshold) for p in players]
    best = None
    for i in range(len(players)):
        if len(points[i]) == 0:
            continue
        for j in range(i + 1, len(players)):
            if len(points[j]) == 0:
                continue
            diff = points[i][:, None, :] - points[j][None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            flat = int(np.argmin(dists))
            a, b = divmod(flat, dists.shape[1])
            distance = float(dists[a, b])
            if best is None or distance < best[0]:
                midpoint = (
                    float((points[i][a, 0] + points[j][b, 0]) / 2.0),
                    float((points[i][a, 1] + points[j][b, 1]) / 2.0),
                )
                best = (distance, midpoint, (i, j))
    return best


def _contact_box(point, player_a: PlayerPose, player_b: PlayerPose, box_scale: float, shape):
    avg_height = _pair_height(player_a, player_b)
    half = max(box_scale * 0.5 * avg_height, 16.0)
    cx, cy = point
    x1 = int(max(cx - half, 0))
    y1 = int(max(cy - half, 0))
    x2 = int(min(cx + half, shape[1] - 1))
    y2 = int(min(cy + half, shape[0] - 1))
    return x1, y1, x2, y2
