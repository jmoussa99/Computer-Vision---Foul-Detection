"""Classical ML baseline over visual tracking/contact descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


CORE_CLIP_FEATURES = (
    "motion_mean",
    "motion_p95_mean",
    "close_interactions_mean",
    "possible_contacts_mean",
    "interaction_min_distance_mean",
    "close_motion_p95_mean",
    "tracked_objects",
    "long_tracks",
    "track_length_mean",
)

FEATURE_SETS = {
    "core": {
        "clip_features": CORE_CLIP_FEATURES,
        "description": "Visual foul-analysis set: motion, tracked movement, close interactions, and possible-contact cues.",
    },
}


def load_feature_dataset(
    feature_root: str | Path,
    dataset_root: str | Path,
    split: str,
    target: str,
    feature_set: str = "core",
) -> tuple[np.ndarray, list[str]]:
    feature_root = Path(feature_root)
    dataset_root = Path(dataset_root)
    _validate_feature_set(feature_set)
    annotations = _load_annotations(dataset_root / split / "annotations.json")
    x_values = []
    y_values = []
    for feature_file in sorted((feature_root / split).glob("action_*/features.json")):
        with feature_file.open(encoding="utf-8") as f:
            payload = json.load(f)
        action_id = str(payload["action_id"])
        if action_id not in annotations:
            continue
        label = _target_label(annotations[action_id], target)
        if not label:
            continue
        x_values.append(_vectorize(payload, feature_set))
        y_values.append(label)
    return np.asarray(x_values, dtype=np.float32), y_values


def train_random_forest(
    feature_root: str | Path,
    dataset_root: str | Path,
    target: str = "offence_severity",
    train_split: str = "Train",
    eval_split: str = "Valid",
    feature_set: str = "core",
    random_state: int = 7,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    _validate_feature_set(feature_set)
    x_train, y_train = load_feature_dataset(feature_root, dataset_root, train_split, target, feature_set)
    x_eval, y_eval = load_feature_dataset(feature_root, dataset_root, eval_split, target, feature_set)
    if len(y_train) == 0:
        raise ValueError(f"No labelled training samples found for split {train_split}.")
    if len(y_eval) == 0:
        raise ValueError(f"No labelled evaluation samples found for split {eval_split}.")

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    report = {
        "target": target,
        "train_split": train_split,
        "eval_split": eval_split,
        "train_samples": len(y_train),
        "eval_samples": len(y_eval),
        "feature_set": feature_set,
        "feature_set_description": FEATURE_SETS[feature_set]["description"],
        "accuracy": float(accuracy_score(y_eval, predictions)),
        "classification_report": classification_report(y_eval, predictions, output_dict=True, zero_division=0),
        "feature_keys": _feature_names(feature_set),
    }
    return model, report


def feature_dim(feature_set: str = "core") -> int:
    _validate_feature_set(feature_set)
    return len(_feature_names(feature_set))


def feature_names(feature_set: str = "core") -> list[str]:
    _validate_feature_set(feature_set)
    return _feature_names(feature_set)


def vectorize_payload(payload: dict[str, Any], feature_set: str = "core") -> list[float]:
    _validate_feature_set(feature_set)
    return _vectorize(payload, feature_set)


def _load_annotations(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["Actions"]


def _target_label(annotation: dict[str, Any], target: str) -> str:
    if target == "action":
        return annotation.get("Action class", "")
    if target == "offence":
        return annotation.get("Offence", "")
    if target == "severity":
        return annotation.get("Severity", "")
    if target == "offence_severity":
        offence = annotation.get("Offence", "")
        severity = annotation.get("Severity", "")
        if offence in ("No Offence", "No offence"):
            return "No offence"
        if offence == "Offence" and severity:
            return f"Offence severity {severity}"
        return offence
    raise ValueError("target must be one of: action, offence, severity, offence_severity")


def _vectorize(payload: dict[str, Any], feature_set: str) -> list[float]:
    settings = FEATURE_SETS[feature_set]
    clip_feature_keys = settings["clip_features"]
    clip_vectors = []
    for clip in payload.get("clips", []):
        clip_vectors.append([float(clip.get(key, 0.0)) for key in clip_feature_keys])
    if not clip_vectors:
        return [0.0] * len(_feature_names(feature_set))

    clip_array = np.asarray(clip_vectors, dtype=np.float32)
    return np.concatenate([clip_array.mean(axis=0), clip_array.max(axis=0)]).tolist()


def _feature_names(feature_set: str) -> list[str]:
    settings = FEATURE_SETS[feature_set]
    clip_feature_keys = settings["clip_features"]
    names = [f"mean_{key}" for key in clip_feature_keys]
    names.extend(f"max_{key}" for key in clip_feature_keys)
    return names


def _validate_feature_set(feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of: {', '.join(FEATURE_SETS)}")
