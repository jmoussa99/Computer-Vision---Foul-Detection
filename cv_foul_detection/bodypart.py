"""Map a contact point to a body region from player keypoints.

Given the contact location found by the motion/contact CV pipeline and the
player skeletons from :mod:`cv_foul_detection.pose`, this assigns the body part
where the foul contact most likely lands (leg, arm, back, etc.) and a coarse
Upper/Under body label compatible with the dataset's ``Bodypart`` annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import PlayerPose


# Fine body regions reported by the recognizer.
FINE_PARTS = ("head", "shoulder", "arm", "back/torso", "hip", "leg", "foot")

# Coarse mapping onto the dataset "Bodypart" taxonomy.
COARSE_BY_FINE = {
    "head": "Upper body",
    "shoulder": "Upper body",
    "arm": "Upper body",
    "back/torso": "Upper body",
    "hip": "Under body",
    "leg": "Under body",
    "foot": "Under body",
}


@dataclass(frozen=True)
class BodyPartAssignment:
    fine: str
    coarse: str
    player_index: int
    distance: float
    confidence: float


def _seg_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def _points_distance(point: np.ndarray, points: list[np.ndarray]) -> float:
    if not points:
        return float("inf")
    return min(float(np.linalg.norm(point - p)) for p in points)


def _segments_distance(point: np.ndarray, segments: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not segments:
        return float("inf")
    return min(_seg_distance(point, a, b) for a, b in segments)


def _region_distances(point: np.ndarray, kp: dict[str, tuple[float, float]]) -> dict[str, float]:
    def pt(name: str) -> np.ndarray | None:
        value = kp.get(name)
        return np.asarray(value, dtype=np.float32) if value is not None else None

    def points(*names: str) -> list[np.ndarray]:
        return [p for p in (pt(n) for n in names) if p is not None]

    def segments(*pairs: tuple[str, str]) -> list[tuple[np.ndarray, np.ndarray]]:
        result = []
        for first, second in pairs:
            a, b = pt(first), pt(second)
            if a is not None and b is not None:
                result.append((a, b))
        return result

    distances: dict[str, float] = {}
    distances["head"] = _points_distance(point, points("nose", "left_eye", "right_eye", "left_ear", "right_ear"))
    distances["shoulder"] = _points_distance(point, points("left_shoulder", "right_shoulder"))
    distances["arm"] = _segments_distance(
        point,
        segments(
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
        ),
    )

    torso_segments = segments(
        ("left_shoulder", "right_shoulder"),
        ("right_shoulder", "right_hip"),
        ("right_hip", "left_hip"),
        ("left_hip", "left_shoulder"),
    )
    shoulder_mid = _midpoint(pt("left_shoulder"), pt("right_shoulder"))
    hip_mid = _midpoint(pt("left_hip"), pt("right_hip"))
    if shoulder_mid is not None and hip_mid is not None:
        torso_segments.append((shoulder_mid, hip_mid))
    distances["back/torso"] = _segments_distance(point, torso_segments)

    distances["hip"] = _points_distance(point, points("left_hip", "right_hip"))
    distances["leg"] = _segments_distance(
        point,
        segments(
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ),
    )
    distances["foot"] = _points_distance(point, points("left_ankle", "right_ankle"))
    return distances


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _player_scale(player: PlayerPose) -> float:
    x1, y1, x2, y2 = player.box
    return max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)


def assign_bodypart(
    contact_point: tuple[float, float],
    players: list[PlayerPose],
    keypoint_score_threshold: float = 2.0,
) -> BodyPartAssignment | None:
    """Assign the body part nearest to ``contact_point`` across ``players``.

    Returns ``None`` when no player has enough visible keypoints.
    """
    point = np.asarray(contact_point, dtype=np.float32)
    best: BodyPartAssignment | None = None
    for index, player in enumerate(players):
        kp = player.visible_keypoints(keypoint_score_threshold)
        if not kp:
            continue
        distances = _region_distances(point, kp)
        fine, distance = min(distances.items(), key=lambda item: item[1])
        if not np.isfinite(distance):
            continue
        scale = _player_scale(player)
        confidence = float(np.exp(-distance / (0.5 * scale)))
        if best is None or distance < best.distance:
            best = BodyPartAssignment(
                fine=fine,
                coarse=COARSE_BY_FINE[fine],
                player_index=index,
                distance=distance,
                confidence=confidence,
            )
    return best
