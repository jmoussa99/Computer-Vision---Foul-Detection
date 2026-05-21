"""Visual CV feature extraction for movement tracking and contact cues."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class FeatureConfig:
    max_frames: int = 125
    frame_stride: int = 2
    resize_width: int = 640
    min_track_area: int = 180
    contact_distance_ratio: float = 0.08
    contact_motion_p95: float = 8.0
    trail_length: int = 24


@dataclass(frozen=True)
class MotionDetection:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float


@dataclass(frozen=True)
class InteractionCue:
    first_id: int
    second_id: int
    distance_ratio: float
    possible_contact: bool


class CentroidTracker:
    def __init__(self, max_missing: int = 8, max_distance: float = 80.0) -> None:
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: dict[int, MotionDetection] = {}
        self.missing: dict[int, int] = {}
        self.tracks: dict[int, list[tuple[float, float]]] = {}

    def update(self, detections: list[MotionDetection]) -> dict[int, MotionDetection]:
        unmatched = set(range(len(detections)))
        updated: dict[int, MotionDetection] = {}

        for object_id, detection in list(self.objects.items()):
            if not unmatched:
                self.missing[object_id] = self.missing.get(object_id, 0) + 1
                continue
            unmatched_list = list(unmatched)
            distances = [np.linalg.norm(np.array(detection.centroid) - np.array(detections[i].centroid)) for i in unmatched_list]
            best_local = int(np.argmin(distances))
            best_det = unmatched_list[best_local]
            if distances[best_local] <= self.max_distance:
                updated[object_id] = detections[best_det]
                self.tracks.setdefault(object_id, []).append(detections[best_det].centroid)
                self.missing[object_id] = 0
                unmatched.remove(best_det)
            else:
                self.missing[object_id] = self.missing.get(object_id, 0) + 1

        for det_idx in unmatched:
            object_id = self.next_id
            self.next_id += 1
            updated[object_id] = detections[det_idx]
            self.missing[object_id] = 0
            self.tracks[object_id] = [detections[det_idx].centroid]

        for object_id in list(self.objects):
            if object_id not in updated and self.missing.get(object_id, 0) > self.max_missing:
                self.objects.pop(object_id, None)
                self.missing.pop(object_id, None)

        self.objects.update(updated)
        return self.objects


class ClipFeatureExtractor:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def extract_clip(self, clip_path: str | Path, overlay_path: str | Path | None = None) -> dict[str, Any]:
        clip_path = Path(clip_path)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {clip_path}")

        bg = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=24, detectShadows=False)
        tracker = CentroidTracker()
        writer = None
        prev_gray: np.ndarray | None = None
        motion_mean: list[float] = []
        motion_p95: list[float] = []
        close_interaction_counts: list[int] = []
        possible_contact_counts: list[int] = []
        min_interaction_distances: list[float] = []
        close_motion_p95: list[float] = []
        frame_count = 0
        sampled = 0

        while sampled < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if (frame_count - 1) % self.config.frame_stride != 0:
                continue
            sampled += 1

            frame = self._resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self._moving_detections(bg.apply(frame))
            tracks = tracker.update(detections)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                p95 = float(np.percentile(mag, 95))
                motion_mean.append(float(np.mean(mag)))
                motion_p95.append(p95)
            else:
                p95 = 0.0

            interaction_cues = self._interaction_cues(tracks, frame.shape, p95)
            close_count = len(interaction_cues)
            contact_count = sum(cue.possible_contact for cue in interaction_cues)
            min_distance = min((cue.distance_ratio for cue in interaction_cues), default=1.0)
            close_interaction_counts.append(close_count)
            possible_contact_counts.append(contact_count)
            min_interaction_distances.append(min_distance)
            if close_count > 0:
                close_motion_p95.append(p95)
            prev_gray = gray

            if overlay_path is not None:
                if writer is None:
                    overlay_path = Path(overlay_path)
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(overlay_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(cap.get(cv2.CAP_PROP_FPS) / self.config.frame_stride, 1),
                        (frame.shape[1], frame.shape[0]),
                    )
                writer.write(self._overlay(frame, tracks, tracker.tracks, interaction_cues, p95))

        cap.release()
        if writer is not None:
            writer.release()

        track_lengths = [len(points) for points in tracker.tracks.values()]
        return {
            "clip": str(clip_path),
            "config": asdict(self.config),
            "frames_sampled": sampled,
            "motion_mean": _safe_mean(motion_mean),
            "motion_p95_mean": _safe_mean(motion_p95),
            "close_interactions_mean": _safe_mean(close_interaction_counts),
            "possible_contacts_mean": _safe_mean(possible_contact_counts),
            "interaction_min_distance_mean": _safe_mean(min_interaction_distances),
            "close_motion_p95_mean": _safe_mean(close_motion_p95),
            "tracked_objects": len(track_lengths),
            "long_tracks": int(sum(length >= 8 for length in track_lengths)),
            "track_length_mean": _safe_mean(track_lengths),
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= self.config.resize_width:
            return frame
        scale = self.config.resize_width / frame.shape[1]
        return cv2.resize(frame, (self.config.resize_width, int(frame.shape[0] * scale)))

    def _moving_detections(self, mask: np.ndarray) -> list[MotionDetection]:
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_track_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                MotionDetection(
                    centroid=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                    bbox=(x, y, w, h),
                    area=float(area),
                )
            )
        return detections

    def _interaction_cues(self, tracks: dict[int, MotionDetection], shape: tuple[int, ...], motion_p95: float) -> list[InteractionCue]:
        if len(tracks) < 2:
            return []
        diagonal = float(np.hypot(shape[0], shape[1]))
        cues = []
        track_items = list(tracks.items())
        for idx, (first_id, first) in enumerate(track_items[:-1]):
            for second_id, second in track_items[idx + 1 :]:
                distance = float(np.linalg.norm(np.array(first.centroid) - np.array(second.centroid)) / diagonal)
                if distance <= self.config.contact_distance_ratio:
                    cues.append(
                        InteractionCue(
                            first_id=first_id,
                            second_id=second_id,
                            distance_ratio=distance,
                            possible_contact=motion_p95 >= self.config.contact_motion_p95,
                        )
                    )
        return cues

    def _overlay(
        self,
        frame: np.ndarray,
        tracks: dict[int, MotionDetection],
        histories: dict[int, list[tuple[float, float]]],
        interaction_cues: list[InteractionCue],
        motion_p95: float,
    ) -> np.ndarray:
        overlay = frame.copy()

        for object_id, detection in tracks.items():
            x, y, w, h = detection.bbox
            center = (int(detection.centroid[0]), int(detection.centroid[1]))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (46, 204, 113), 2)
            cv2.circle(overlay, center, 4, (46, 204, 113), -1)
            cv2.putText(overlay, f"P{object_id}", (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 204, 113), 1)

            history = histories.get(object_id, [])[-self.config.trail_length :]
            for first, second in zip(history, history[1:]):
                cv2.line(overlay, (int(first[0]), int(first[1])), (int(second[0]), int(second[1])), (255, 255, 255), 2)

        possible_contacts = 0
        for cue in interaction_cues:
            first = tracks.get(cue.first_id)
            second = tracks.get(cue.second_id)
            if first is None or second is None:
                continue
            first_center = (int(first.centroid[0]), int(first.centroid[1]))
            second_center = (int(second.centroid[0]), int(second.centroid[1]))
            color = (0, 0, 255) if cue.possible_contact else (0, 165, 255)
            thickness = 4 if cue.possible_contact else 2
            cv2.line(overlay, first_center, second_center, color, thickness)
            label_pos = ((first_center[0] + second_center[0]) // 2, (first_center[1] + second_center[1]) // 2)
            label = "possible contact" if cue.possible_contact else "close"
            cv2.putText(overlay, label, (label_pos[0] + 6, label_pos[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if cue.possible_contact:
                possible_contacts += 1

        status = f"motion p95={motion_p95:.1f} | close pairs={len(interaction_cues)} | possible contact={possible_contacts}"
        cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 560), 40), (0, 0, 0), -1)
        cv2.putText(overlay, status, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if possible_contacts:
            cv2.rectangle(overlay, (8, 46), (240, 78), (0, 0, 180), -1)
            cv2.putText(overlay, "POSSIBLE CONTACT", (18, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay


def _safe_mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0
