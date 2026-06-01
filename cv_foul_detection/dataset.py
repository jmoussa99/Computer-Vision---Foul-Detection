"""Dataset discovery utilities for the SoccerNet-MVFoul folder layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Real clips are named clip_<int>.mp4. The dataset also ships decoy files such as
# clip_1blabla.mp4 that must be ignored.
_CLIP_PATTERN = re.compile(r"^clip_(\d+)$")
_ACTION_PATTERN = re.compile(r"^action_(\d+)$")


@dataclass(frozen=True)
class ActionClips:
    split: str
    action_id: str
    clips: tuple[Path, ...]


def _numeric_suffix(name: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.match(name)
    return int(match.group(1)) if match else None


def iter_actions(dataset_root: str | Path, splits: Iterable[str] = ("Train", "Valid", "Test", "Chall")) -> Iterable[ActionClips]:
    root = Path(dataset_root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        action_dirs = [d for d in split_dir.glob("action_*") if _numeric_suffix(d.name, _ACTION_PATTERN) is not None]
        for action_dir in sorted(action_dirs, key=lambda p: _numeric_suffix(p.name, _ACTION_PATTERN)):
            clip_files = [c for c in action_dir.glob("clip_*.mp4") if _numeric_suffix(c.stem, _CLIP_PATTERN) is not None]
            clips = tuple(sorted(clip_files, key=lambda p: _numeric_suffix(p.stem, _CLIP_PATTERN)))
            if clips:
                yield ActionClips(split=split, action_id=action_dir.name.split("_")[-1], clips=clips)
