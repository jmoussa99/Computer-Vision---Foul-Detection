#!/usr/bin/env python3
"""Calibrate a camera from chessboard images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_foul_detection.calibration import calibrate_from_chessboard
from cv_foul_detection.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate camera intrinsics/extrinsics from chessboard calibration images.")
    parser.add_argument("--images", required=True, help="Folder containing chessboard images.")
    parser.add_argument("--output", default="outputs/calibration/camera.json")
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0)
    args = parser.parse_args()

    payload = calibrate_from_chessboard(
        args.images,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size=args.square_size,
    )
    write_json(args.output, payload)
    print(f"Calibration written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
