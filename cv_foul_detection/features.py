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
    min_contact_area: int = 450
    field_top_ratio: float = 0.18
    contact_distance_ratio: float = 0.08
    contact_motion_p95: float = 8.0
    contact_pause_seconds: float = 1.0
    contact_box_padding: int = 12
    # How to pick the representative contact frame:
    #   "contact": strongest motion spike among possible-contact frames (default),
    #   "closest": closest two-player approach among interaction frames.
    peak_strategy: str = "contact"


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


@dataclass
class FrameState:
    """Per-frame motion/tracking summary, decoupled from the frame image.

    Caching a list of these (plus the frame images or cached poses) lets the
    peak-contact selection be replayed for many threshold configs without
    re-decoding video.
    """

    frame_index: int
    height: int
    width: int
    tracks: dict[int, MotionDetection]
    motion_p95: float


@dataclass
class PeakSelection:
    """Result of choosing the representative contact frame (no image)."""

    frame_index: int
    contact_point: tuple[float, float] | None
    contact_box: tuple[int, int, int, int] | None
    player_boxes: tuple[tuple[int, int, int, int], ...]
    motion_p95: float
    distance_ratio: float
    has_contact: bool


@dataclass
class PeakContact:
    """Representative contact moment located in a clip.

    Used by the body-part recognizer to know where (and on which frame) to run
    pose estimation. ``frame`` is the resized BGR frame at ``frame_index``.
    """

    frame_index: int
    frame: np.ndarray
    contact_point: tuple[float, float] | None
    contact_box: tuple[int, int, int, int] | None
    player_boxes: tuple[tuple[int, int, int, int], ...]
    motion_p95: float
    distance_ratio: float
    has_contact: bool


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
        previous_contact = False
        output_fps = 1.0

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
                    output_fps = max(cap.get(cv2.CAP_PROP_FPS) / self.config.frame_stride, 1)
                    writer = cv2.VideoWriter(
                        str(overlay_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                overlay = self._overlay(frame, tracks, interaction_cues)
                writer.write(overlay)
                if contact_count > 0 and not previous_contact:
                    for _ in range(int(round(output_fps * self.config.contact_pause_seconds))):
                        writer.write(overlay)
                previous_contact = contact_count > 0

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

    def analyze_frames(
        self, clip_path: str | Path, keep_images: bool = False
    ) -> tuple[list[FrameState], dict[int, np.ndarray]]:
        """Decode a clip once and return per-frame motion/tracking states.

        Set ``keep_images`` to also return the resized BGR frames keyed by
        frame index (needed to run pose estimation on the chosen frame).
        """
        clip_path = Path(clip_path)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {clip_path}")

        bg = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=24, detectShadows=False)
        tracker = CentroidTracker()
        prev_gray: np.ndarray | None = None
        frame_count = 0
        sampled = 0
        states: list[FrameState] = []
        images: dict[int, np.ndarray] = {}

        while sampled < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if (frame_count - 1) % self.config.frame_stride != 0:
                continue

            frame = self._resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self._moving_detections(bg.apply(frame))
            tracks = tracker.update(detections)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                p95 = float(np.percentile(mag, 95))
            else:
                p95 = 0.0
            prev_gray = gray

            states.append(
                FrameState(
                    frame_index=sampled,
                    height=frame.shape[0],
                    width=frame.shape[1],
                    tracks=dict(tracks),
                    motion_p95=p95,
                )
            )
            if keep_images:
                images[sampled] = frame.copy()
            sampled += 1

        cap.release()
        return states, images

    def select_peak_state(self, states: list[FrameState]) -> PeakSelection | None:
        """Pick the representative contact frame from per-frame states.

        Honors ``config.peak_strategy``. Returns ``None`` only for empty input.
        """
        best_contact: PeakSelection | None = None
        best_close: PeakSelection | None = None
        best_motion: PeakSelection | None = None

        for state in states:
            shape = (state.height, state.width, 3)
            if best_motion is None or state.motion_p95 > best_motion.motion_p95:
                best_motion = PeakSelection(
                    frame_index=state.frame_index,
                    contact_point=None,
                    contact_box=None,
                    player_boxes=(),
                    motion_p95=state.motion_p95,
                    distance_ratio=1.0,
                    has_contact=False,
                )

            cues = self._interaction_cues(state.tracks, shape, state.motion_p95)
            if not cues:
                continue
            primary = min(cues, key=lambda cue: cue.distance_ratio)
            first = state.tracks.get(primary.first_id)
            second = state.tracks.get(primary.second_id)
            if first is None or second is None:
                continue
            candidate = self._build_peak(shape, state.frame_index, primary, first, second, state.motion_p95)
            if primary.possible_contact and (best_contact is None or state.motion_p95 > best_contact.motion_p95):
                best_contact = candidate
            if best_close is None or primary.distance_ratio < best_close.distance_ratio:
                best_close = candidate

        if self.config.peak_strategy == "closest":
            return best_close or best_contact or best_motion
        return best_contact or best_close or best_motion

    def extract_peak_contact(self, clip_path: str | Path) -> PeakContact | None:
        """Find the most representative contact moment in a clip.

        Returns the chosen frame (per ``config.peak_strategy``) with its image,
        contact point/box, and contributing player boxes. Returns ``None`` only
        if the clip has no frames.
        """
        states, images = self.analyze_frames(clip_path, keep_images=True)
        selection = self.select_peak_state(states)
        if selection is None:
            return None
        frame = images.get(selection.frame_index)
        if frame is None:
            return None
        return PeakContact(
            frame_index=selection.frame_index,
            frame=frame,
            contact_point=selection.contact_point,
            contact_box=selection.contact_box,
            player_boxes=selection.player_boxes,
            motion_p95=selection.motion_p95,
            distance_ratio=selection.distance_ratio,
            has_contact=selection.has_contact,
        )

    def _build_peak(
        self,
        shape: tuple[int, ...],
        frame_index: int,
        cue: InteractionCue,
        first: MotionDetection,
        second: MotionDetection,
        motion_p95: float,
    ) -> PeakSelection:
        box = self._contact_box(first, second, shape)
        point = (
            (first.centroid[0] + second.centroid[0]) / 2.0,
            (first.centroid[1] + second.centroid[1]) / 2.0,
        )
        return PeakSelection(
            frame_index=frame_index,
            contact_point=point,
            contact_box=box,
            player_boxes=(first.bbox, second.bbox),
            motion_p95=motion_p95,
            distance_ratio=cue.distance_ratio,
            has_contact=cue.possible_contact,
        )

    def read_frames(self, clip_path: str | Path) -> list[tuple[int, np.ndarray]]:
        """Decode and return sampled, resized BGR frames (no motion analysis).

        Cheaper than analyze_frames; used by the pose-based contact locator.
        """
        clip_path = Path(clip_path)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {clip_path}")
        frames: list[tuple[int, np.ndarray]] = []
        frame_count = 0
        sampled = 0
        while sampled < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if (frame_count - 1) % self.config.frame_stride != 0:
                continue
            frames.append((sampled, self._resize(frame)))
            sampled += 1
        cap.release()
        return frames

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
                if first.area < self.config.min_contact_area or second.area < self.config.min_contact_area:
                    continue
                if not self._in_playing_area(first, shape) or not self._in_playing_area(second, shape):
                    continue
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

    def _in_playing_area(self, detection: MotionDetection, shape: tuple[int, ...]) -> bool:
        return detection.centroid[1] >= shape[0] * self.config.field_top_ratio

    def _overlay(
        self,
        frame: np.ndarray,
        tracks: dict[int, MotionDetection],
        interaction_cues: list[InteractionCue],
    ) -> np.ndarray:
        overlay = frame.copy()
        for cue in interaction_cues:
            if not cue.possible_contact:
                continue
            first = tracks.get(cue.first_id)
            second = tracks.get(cue.second_id)
            if first is None or second is None:
                continue
            x1, y1, x2, y2 = self._contact_box(first, second, frame.shape)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
        return overlay

    def _contact_box(self, first: MotionDetection, second: MotionDetection, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        fx, fy, fw, fh = first.bbox
        sx, sy, sw, sh = second.bbox
        padding = self.config.contact_box_padding
        x1 = max(min(fx, sx) - padding, 0)
        y1 = max(min(fy, sy) - padding, 0)
        x2 = min(max(fx + fw, sx + sw) + padding, shape[1] - 1)
        y2 = min(max(fy + fh, sy + sh) + padding, shape[0] - 1)
        return x1, y1, x2, y2


def _safe_mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0
