"""TokaMark task metadata: task configs, train-split stats, splits, outliers.

All files in ``paper/tokamark/metadata/`` are verbatim copies fetched on
2026-08-17 from the official benchmark repo
(github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark, ``main`` branch):

- ``task_2-1.yaml``                      <- src/tokamark/tasks_configs/group_2_magnetics_dynamics/
- ``random_split_signals_stats.yaml``    <- src/tokamark/metadata/  (OFFICIAL train-split
  per-signal mean/std/dt — we normalize with these, we do NOT recompute stats)
- ``TokaMark_data_splits.csv``           <- src/tokamark/metadata/  (11,573 shots,
  train 8,963 / val 1,115 / test 1,110 + 5 CV folds + ip_negative flags)
- ``random_split_outlier_metadata.yaml`` <- src/MAST_tools/metadata/  (shot -> list of
  "source-signal" outlier channels to drop, used when remove_outliers=True)

``build_task_metadata`` mirrors ``tokamark.tasks.get_task_metadata`` exactly
(dt and values_shape come from the stats YAML, not from the data itself).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

METADATA_DIR = Path(__file__).resolve().parent / "metadata"

TASK_CONFIG_FILES = {"task_2-1": "task_2-1.yaml"}
SIGNAL_STATS_FILE = METADATA_DIR / "random_split_signals_stats.yaml"
SPLITS_CSV_FILE = METADATA_DIR / "TokaMark_data_splits.csv"
OUTLIER_METADATA_FILE = METADATA_DIR / "random_split_outlier_metadata.yaml"

VALID_SPLITS = (
    "train",
    "val",
    "test",
    "fold_1_train", "fold_1_val", "fold_2_train", "fold_2_val",
    "fold_3_train", "fold_3_val", "fold_4_train", "fold_4_val",
    "fold_5_train", "fold_5_val",
)


def signal_key(source: str, signal: str) -> str:
    """Return the official '{source}-{signal}' key for one signal."""
    return f"{source}-{signal}"


def load_task_config(task_name: str = "task_2-1") -> dict[str, Any]:
    """Load one task-config YAML (verbatim copy of the official file)."""
    if task_name not in TASK_CONFIG_FILES:
        raise ValueError(f"unknown task {task_name!r}; available: {sorted(TASK_CONFIG_FILES)}")
    path = METADATA_DIR / TASK_CONFIG_FILES[task_name]
    config = _load_yaml(path)
    for required in ("task_type", "task_window_segmenter", "stride_window"):
        if required not in config:
            raise ValueError(f"{path} is missing required key {required!r}")
    return config


def load_signal_stats() -> dict[str, Any]:
    """Load the OFFICIAL train-split per-signal stats (mean/std/dt/values_shape)."""
    stats = _load_yaml(SIGNAL_STATS_FILE)
    for key, entry in stats.items():
        for field in ("dt", "values_shape", "mean", "std"):
            if field not in entry:
                raise ValueError(f"stats entry {key!r} is missing field {field!r}")
    return stats


def load_split_shot_ids(split: str = "train") -> list[int]:
    """Return shot IDs for one official split column, in CSV (deterministic) order."""
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; valid: {VALID_SPLITS}")
    if not SPLITS_CSV_FILE.is_file():
        raise FileNotFoundError(f"splits CSV not found: {SPLITS_CSV_FILE}")
    shot_ids = []
    with open(SPLITS_CSV_FILE, newline="") as handle:
        for row in csv.DictReader(handle):
            if row[split] == "True":
                shot_ids.append(int(row["shot_id"]))
    if not shot_ids:
        raise ValueError(f"split {split!r} matched zero shots — corrupt CSV?")
    return shot_ids


def load_outlier_metadata() -> dict[int, list[str]]:
    """Load the official shot -> ['source-signal', ...] outlier map (random split)."""
    raw = _load_yaml(OUTLIER_METADATA_FILE)
    return {int(shot_id): list(signals or []) for shot_id, signals in raw.items()}


def role_keys(task_config: dict[str, Any]) -> dict[str, list[str]]:
    """Return {'input'|'actuator'|'output': ['source-signal', ...]} for a task."""
    segmenter = task_config["task_window_segmenter"]
    return {
        role: [signal_key(source, signal) for source, signal in (segmenter[f"{role}_keys"] or [])]
        for role in ("input", "actuator", "output")
    }


def unique_source_signals(task_config: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered unique (source, signal) pairs across all roles (official unicity rule)."""
    names = task_config["sources_and_signals"]
    combined = [
        tuple(pair)
        for role in ("input_name", "actuator_name", "output_name")
        for pair in (names.get(role) or [])
    ]
    return [pair for index, pair in enumerate(combined) if pair not in combined[:index]]


def build_task_metadata(task_config: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``tokamark.tasks.get_task_metadata`` (fresh copies; no stats mutation).

    Returns {'stride_in_secs': float, 'input': {key: meta}, 'actuator': ..., 'output': ...}
    where meta = stats entry + length_in_seconds/length_in_time_stamps/stride_in_time_stamps.
    """
    segmenter = task_config["task_window_segmenter"]
    input_length = float(segmenter["input_length"])
    output_length = float(segmenter["output_length"])
    delta = float(segmenter["delta"])
    stride_in_secs = float(task_config["stride_window"])
    role_lengths = {
        "input": input_length,
        "actuator": input_length + delta + output_length,
        "output": output_length,
    }

    keys = role_keys(task_config)
    metadata: dict[str, Any] = {"stride_in_secs": stride_in_secs}
    for role, length_in_secs in role_lengths.items():
        metadata[role] = {}
        for key in keys[role]:
            if key not in stats:
                raise KeyError(f"signal {key!r} has no entry in {SIGNAL_STATS_FILE.name}")
            entry = dict(stats[key])  # copy: the official code aliases/mutates these
            entry["values_shape"] = tuple(entry["values_shape"])
            entry["length_in_seconds"] = length_in_secs
            entry["length_in_time_stamps"] = int(np.round(length_in_secs / entry["dt"]))
            entry["stride_in_time_stamps"] = int(np.round(stride_in_secs / entry["dt"]))
            metadata[role][key] = entry
    return metadata


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must parse to a non-empty mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with open(path) as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict) or not content:
        raise ValueError(f"{path} did not parse to a non-empty mapping")
    return content
