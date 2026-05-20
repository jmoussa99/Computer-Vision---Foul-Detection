"""Classical CV feature extraction for multi-view foul clips.

The extractor intentionally produces compact, explainable descriptors that can be
saved next to the VARS deep-video baseline or used for reports and ablations.
"""

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
    canny_low: int = 80
    canny_high: int = 160
    min_track_area: int = 180
    orb_features: int = 1000


class CentroidTracker:
    def __init__(self, max_missing: int = 8, max_distance: float = 80.0) -> None:
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: dict[int, tuple[float, float]] = {}
        self.missing: dict[int, int] = {}
        self.tracks: dict[int, list[tuple[float, float]]] = {}

    def update(self, detections: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
        unmatched = set(range(len(detections)))
        updated: dict[int, tuple[float, float]] = {}

        for object_id, centroid in list(self.objects.items()):
            if not unmatched:
                self.missing[object_id] = self.missing.get(object_id, 0) + 1
                continue
            distances = [np.linalg.norm(np.array(centroid) - np.array(detections[i])) for i in unmatched]
            best_local = int(np.argmin(distances))
            best_det = list(unmatched)[best_local]
            if distances[best_local] <= self.max_distance:
                updated[object_id] = detections[best_det]
                self.tracks.setdefault(object_id, []).append(detections[best_det])
                self.missing[object_id] = 0
                unmatched.remove(best_det)
            else:
                self.missing[object_id] = self.missing.get(object_id, 0) + 1

        for det_idx in unmatched:
            object_id = self.next_id
            self.next_id += 1
            updated[object_id] = detections[det_idx]
            self.missing[object_id] = 0
            self.tracks[object_id] = [detections[det_idx]]

        for object_id in list(self.objects):
            if object_id not in updated and self.missing.get(object_id, 0) > self.max_missing:
                self.objects.pop(object_id, None)
                self.missing.pop(object_id, None)

        self.objects.update(updated)
        return self.objects


class ClipFeatureExtractor:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.orb = cv2.ORB_create(nfeatures=self.config.orb_features)

    def extract_clip(self, clip_path: str | Path, overlay_path: str | Path | None = None) -> dict[str, Any]:
        clip_path = Path(clip_path)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {clip_path}")

        bg = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=24, detectShadows=False)
        tracker = CentroidTracker()
        writer = None
        prev_gray: np.ndarray | None = None
        edge_density: list[float] = []
        motion_mean: list[float] = []
        motion_p95: list[float] = []
        close_interaction_counts: list[int] = []
        min_interaction_distances: list[float] = []
        close_motion_p95: list[float] = []
        keypoints_per_frame: list[int] = []
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
            edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
            edge_density.append(float(np.count_nonzero(edges) / edges.size))

            keypoints = self.orb.detect(gray, None)
            keypoints_per_frame.append(len(keypoints))

            detections = self._moving_centroids(bg.apply(frame))
            tracks = tracker.update(detections)
            min_distance, close_count = self._interaction_stats(detections, frame.shape)
            min_interaction_distances.append(min_distance)
            close_interaction_counts.append(close_count)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                p95 = float(np.percentile(mag, 95))
                motion_mean.append(float(np.mean(mag)))
                motion_p95.append(p95)
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
                writer.write(self._overlay(frame, edges, tracks))

        cap.release()
        if writer is not None:
            writer.release()

        track_lengths = [len(points) for points in tracker.tracks.values()]
        return {
            "clip": str(clip_path),
            "config": asdict(self.config),
            "frames_sampled": sampled,
            "edge_density_mean": _safe_mean(edge_density),
            "edge_density_std": _safe_std(edge_density),
            "motion_mean": _safe_mean(motion_mean),
            "motion_p95_mean": _safe_mean(motion_p95),
            "close_interactions_mean": _safe_mean(close_interaction_counts),
            "interaction_min_distance_mean": _safe_mean(min_interaction_distances),
            "close_motion_p95_mean": _safe_mean(close_motion_p95),
            "orb_keypoints_mean": _safe_mean(keypoints_per_frame),
            "tracked_objects": len(track_lengths),
            "long_tracks": int(sum(length >= 8 for length in track_lengths)),
            "track_length_mean": _safe_mean(track_lengths),
        }

    def compare_views(self, clip_a: str | Path, clip_b: str | Path) -> dict[str, Any]:
        frame_a = self._first_frame(clip_a)
        frame_b = self._first_frame(clip_b)
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        kp_a, desc_a = self.orb.detectAndCompute(gray_a, None)
        kp_b, desc_b = self.orb.detectAndCompute(gray_b, None)
        if desc_a is None or desc_b is None:
            return {"matches": 0, "inliers": 0, "homography": None}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)
        if len(matches) < 4:
            return {"matches": len(matches), "inliers": 0, "homography": None}

        src = np.float32([kp_a[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return {
            "matches": len(matches),
            "inliers": inliers,
            "homography": homography.tolist() if homography is not None else None,
        }

    def stitch_pair(self, clip_a: str | Path, clip_b: str | Path, output_path: str | Path) -> dict[str, Any]:
        frame_a = self._first_frame(clip_a)
        frame_b = self._first_frame(clip_b)
        comparison = self.compare_views(clip_a, clip_b)
        homography = comparison.get("homography")
        if homography is None:
            raise ValueError("Not enough local-feature inliers to estimate a homography.")
        h = np.array(homography, dtype=np.float64)
        width = frame_a.shape[1] + frame_b.shape[1]
        height = max(frame_a.shape[0], frame_b.shape[0])
        panorama = cv2.warpPerspective(frame_a, h, (width, height))
        panorama[: frame_b.shape[0], : frame_b.shape[1]] = frame_b
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), panorama)
        return {**comparison, "stitch": str(output_path)}

    def stereo_disparity(self, clip_left: str | Path, clip_right: str | Path, output_path: str | Path) -> dict[str, Any]:
        left = cv2.cvtColor(self._first_frame(clip_left), cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(self._first_frame(clip_right), cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=15)
        disparity = stereo.compute(left, right).astype(np.float32) / 16.0
        norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), norm)
        return {
            "disparity": str(output_path),
            "disparity_mean": float(np.mean(disparity)),
            "disparity_std": float(np.std(disparity)),
        }

    def _first_frame(self, clip_path: str | Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(clip_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise FileNotFoundError(f"Could not read first frame: {clip_path}")
        return self._resize(frame)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= self.config.resize_width:
            return frame
        scale = self.config.resize_width / frame.shape[1]
        return cv2.resize(frame, (self.config.resize_width, int(frame.shape[0] * scale)))

    def _moving_centroids(self, mask: np.ndarray) -> list[tuple[float, float]]:
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) < self.config.min_track_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroids.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
        return centroids

    def _interaction_stats(self, centroids: list[tuple[float, float]], shape: tuple[int, ...]) -> tuple[float, int]:
        if len(centroids) < 2:
            return 1.0, 0
        diagonal = float(np.hypot(shape[0], shape[1]))
        min_distance = 1.0
        close_count = 0
        for idx, first in enumerate(centroids[:-1]):
            for second in centroids[idx + 1 :]:
                distance = float(np.linalg.norm(np.array(first) - np.array(second)) / diagonal)
                min_distance = min(min_distance, distance)
                if distance < 0.08:
                    close_count += 1
        return min_distance, close_count

    def _overlay(self, frame: np.ndarray, edges: np.ndarray, tracks: dict[int, tuple[float, float]]) -> np.ndarray:
        overlay = frame.copy()
        overlay[edges > 0] = (0, 255, 255)
        for object_id, (x, y) in tracks.items():
            center = (int(x), int(y))
            cv2.circle(overlay, center, 5, (0, 255, 0), -1)
            cv2.putText(overlay, str(object_id), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return overlay


def _safe_mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float] | list[int]) -> float:
    return float(np.std(values)) if values else 0.0
