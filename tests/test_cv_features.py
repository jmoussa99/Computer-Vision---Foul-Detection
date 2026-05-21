from pathlib import Path

import cv2
import numpy as np

from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig


def test_extract_clip_features_from_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    for idx in range(12):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (8 + idx * 3, 20), (28 + idx * 3, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    extractor = ClipFeatureExtractor(FeatureConfig(max_frames=12, frame_stride=1, resize_width=96))
    features = extractor.extract_clip(video_path)

    assert features["frames_sampled"] == 12
    assert features["motion_mean"] > 0
    assert features["tracked_objects"] >= 1
    assert "possible_contacts_mean" in features
