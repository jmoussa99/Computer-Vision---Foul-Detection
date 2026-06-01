"""Locate the foul contact point from player poses.

Instead of trusting raw motion blobs, this finds the moment and place where two
players' skeletons are closest -- i.e. where bodily contact actually occurs --
and returns a tight box centered there. The deep net has already decided the
clip is a foul, and clips are centered on the foul, so the closest player
approach (optionally within the central time window) is the contact.
"""

from __future__ import annotations

from dataclasses import dataclass

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
) -> PoseContact | None:
    """Return the closest two-player contact across (a window of) the frames.

    ``center_frac`` restricts the search to the central fraction of the clip,
    where the foul occurs. Returns ``None`` if no frame has two posed players.
    """
    if not frames:
        return None

    window = _center_window(frames, center_frac)
    window = _subsample(window, max_pose_frames)

    best: PoseContact | None = None
    for frame_index, frame in window:
        players = pose.estimate(frame)
        pair = _closest_pair(players, keypoint_score_threshold)
        if pair is None:
            continue
        distance, point, (i, j) = pair
        if best is None or distance < best.distance:
            box = _contact_box(point, players[i], players[j], box_scale, frame.shape)
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
    height_a = player_a.box[3] - player_a.box[1]
    height_b = player_b.box[3] - player_b.box[1]
    avg_height = max((height_a + height_b) / 2.0, 1.0)
    half = max(box_scale * 0.5 * avg_height, 16.0)
    cx, cy = point
    x1 = int(max(cx - half, 0))
    y1 = int(max(cy - half, 0))
    x2 = int(min(cx + half, shape[1] - 1))
    y2 = int(min(cy + half, shape[0] - 1))
    return x1, y1, x2, y2
