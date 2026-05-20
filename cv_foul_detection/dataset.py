"""Dataset discovery utilities for the SoccerNet-MVFoul folder layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ActionClips:
    split: str
    action_id: str
    clips: tuple[Path, ...]


def iter_actions(dataset_root: str | Path, splits: Iterable[str] = ("Train", "Valid", "Test", "Chall")) -> Iterable[ActionClips]:
    root = Path(dataset_root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for action_dir in sorted(split_dir.glob("action_*"), key=lambda p: int(p.name.split("_")[-1])):
            clips = tuple(sorted(action_dir.glob("clip_*.mp4"), key=lambda p: int(p.stem.split("_")[-1])))
            if clips:
                yield ActionClips(split=split, action_id=action_dir.name.split("_")[-1], clips=clips)
