"""Player pose estimation using torchvision Keypoint R-CNN (GPU-friendly).

This module locates players in a frame and returns COCO-17 keypoints per
player. It is used by the body-part recognizer to map a contact point to the
nearest body region (leg, arm, back, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
)


@dataclass(frozen=True)
class PlayerPose:
    """A detected player with bounding box and COCO-17 keypoints."""

    box: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray  # shape (17, 3): x, y, per-keypoint score

    def visible_keypoints(self, min_score: float) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y, score = self.keypoints[idx]
            if score >= min_score:
                result[name] = (float(x), float(y))
        return result


class PoseEstimator:
    """Wraps a pretrained Keypoint R-CNN and runs it on the requested device."""

    def __init__(
        self,
        device: str | None = None,
        person_score_threshold: float = 0.7,
        keypoint_score_threshold: float = 2.0,
        max_players: int = 8,
    ) -> None:
        import torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.person_score_threshold = person_score_threshold
        self.keypoint_score_threshold = keypoint_score_threshold
        self.max_players = max_players

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model.eval().to(self.device)

    def estimate(self, frame_bgr: np.ndarray) -> list[PlayerPose]:
        """Return detected players for a single BGR (OpenCV) frame."""
        torch = self._torch
        rgb = frame_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255.0
        with torch.no_grad():
            outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        keypoints = outputs["keypoints"].cpu().numpy()
        keypoint_scores = outputs["keypoints_scores"].cpu().numpy()

        players: list[PlayerPose] = []
        for box, score, kps, kp_scores in zip(boxes, scores, keypoints, keypoint_scores):
            if score < self.person_score_threshold:
                continue
            kp = np.zeros((17, 3), dtype=np.float32)
            kp[:, 0] = kps[:, 0]
            kp[:, 1] = kps[:, 1]
            kp[:, 2] = kp_scores
            players.append(
                PlayerPose(
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(score),
                    keypoints=kp,
                )
            )
            if len(players) >= self.max_players:
                break
        return players
