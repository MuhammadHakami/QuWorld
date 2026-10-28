"""Streaming-first TokaMark G2 loader (task 2-1 magnetics forecasting).

Reads per-shot Zarr v2 stores lazily over anonymous S3 (zarr 3 + s3fs,
``anon=True``, endpoint https://s3.echo.stfc.ac.uk, store
``s3://mast/tokamark/v1/{shot_id}.zarr`` with consolidated metadata) — no bulk
mirror. Only the task's signals are ever fetched (~2.4 MB compressed per shot
for task 2-1 out of a ~236 MB store).

Pipeline per shot (mirrors the official ``tokamark`` + ``MAST_tools`` code —
see ``paper/tokamark/task_meta.py`` for file provenance):

1. fetch raw signals from S3 (or the local LRU npz cache, see below);
2. outlier removal: official ``random_split_outlier_metadata.yaml`` empties
   listed 'source-signal' channels per shot (``remove_outliers=True`` default);
3. EFIT-rating mask: equilibrium samples where ``equilibrium/ip_rating == 0``
   are set to NaN; if ip_rating is absent, all equilibrium signals are dropped
   (``remove_bad_efit_rating=True`` default) — as in ``MastDataset.__getitem__``;
4. signal transforms, matching ``build_common_signal_transform_map`` for task
   2-1: x_point_r/z clipped to NaN outside [-2, 2] BEFORE scaling, then every
   signal z-scored with the OFFICIAL train-split stats
   (``random_split_signals_stats.yaml``; we do not recompute stats). NaN
   filling is off (official default ``use_nan_filling=False``);
5. windowing, mirroring ``TokaMarkDataset`` exactly (markovian): global start
   from inputs+actuators, global end from outputs, NaN-padding to that
   interval on each signal's stats-dt grid, ``t_cuts = arange(start+0.005,
   end-0.025, 0.001)``, per-signal integer index math on the stats dt, indices
   clipped, single-channel (1, T) windows squeezed to (T,).

Cache: ``data/tokamark_cache/`` holds one compressed ``.npz`` of RAW decoded
signals per shot (~2-3 MB), capped at 5 GiB, least-recently-used eviction
(mtime is refreshed on every hit, oldest files deleted first). Raw values are
cached so flag/normalization changes never invalidate the cache.

Torch-free by design (CPU/network only); wrap in a torch IterableDataset at
training time. Windows are dicts of numpy arrays in the official layout:
``{"shot_id", "window_index", "t_cut", "input"|"actuator"|"output":
{key: {"time", "values"}}}``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator
from zlib import error as zlib_error  # np.load on a truncated npz can raise this

import numpy as np
import zarr
from zarr.storage import FsspecStore, WrapperStore

if __package__ in (None, ""):  # allow `python paper/tokamark/dataset.py`-style use
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import task_meta
else:
    from . import task_meta  # type: ignore[no-redef]

# ======================================================================================
# Constants (endpoint verified anonymously 2026-08-17; see README.md)

S3_ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
S3_STORE_URL = "s3://mast/tokamark/v1/{shot_id}.zarr"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "tokamark_cache"
DEFAULT_CACHE_CAP_BYTES = 5 * 1024**3  # 5 GiB
IP_RATING_KEY = "equilibrium-ip_rating"
X_POINT_CLIP_KEYS = ("equilibrium-x_point_r", "equilibrium-x_point_z")
X_POINT_CLIP_LIMIT = 2.0  # official ClipXPointTransform: |value| > 2 -> NaN


def _empty_entry() -> dict[str, np.ndarray]:
    """Official placeholder for a missing/removed signal."""
    return {"time": np.array([]), "values": np.array([])}


# ======================================================================================
# S3 access with byte accounting


class ByteCounter:
    """Mutable tally of bytes/objects fetched from the remote store."""

    def __init__(self) -> None:
        self.nbytes = 0
        self.ngets = 0

    def add(self, nbytes: int) -> None:
        self.nbytes += nbytes
        self.ngets += 1


class CountingStore(WrapperStore):
    """Zarr store wrapper counting every byte fetched from the wrapped store.

    zarr re-instantiates stores via ``_with_store`` (e.g. for read-only
    copies), so the counter is a shared object that survives those copies.
    """

    def __init__(self, store: Any, counter: ByteCounter | None = None) -> None:
        super().__init__(store)
        self.counter = counter if counter is not None else ByteCounter()

    def _with_store(self, store: Any) -> "CountingStore":
        return CountingStore(store, counter=self.counter)

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        result = await super().get(key, prototype, byte_range)
        if result is not None:
            self.counter.add(len(result))
        return result


def open_shot_root(shot_id: int, counter: ByteCounter | None = None) -> zarr.Group:
    """Lazily open one shot's consolidated Zarr v2 store over anonymous S3."""
    if shot_id <= 0:
        raise ValueError(f"shot_id must be positive, got {shot_id}")
    url = S3_STORE_URL.format(shot_id=shot_id)
    try:
        store = FsspecStore.from_url(
            url,
            storage_options={"anon": True, "endpoint_url": S3_ENDPOINT_URL, "asynchronous": True},
        )
        wrapped = CountingStore(store, counter=counter) if counter is not None else store
        return zarr.open_group(store=wrapped, mode="r", use_consolidated=True)
    except (OSError, KeyError, ValueError) as err:
        raise RuntimeError(f"failed to open {url} via {S3_ENDPOINT_URL}: {err}") from err


def _find_time_dim(dims: Any) -> str | None:
    """First dimension name starting with 'time' (official coord-discovery rule)."""
    if not isinstance(dims, (list, tuple)):
        return None
    for dim in dims:
        if str(dim).startswith("time"):
            return str(dim)
    return None


def fetch_shot_signals(
    shot_id: int,
    source_signals: list[tuple[str, str]],
    counter: ByteCounter | None = None,
    include_ip_rating: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Fetch RAW (untransformed) signals for one shot from S3.

    Returns {'source-signal': {'time': (T,), 'values': (C, T)}}; missing
    signals get empty arrays (official convention). 1-D signals are expanded
    to (1, T) exactly like ``MastDataset.__getitem__``. When any equilibrium
    signal is requested and ``include_ip_rating``, ``equilibrium-ip_rating``
    is fetched too (raw, for the later EFIT mask).
    """
    root = open_shot_root(shot_id, counter=counter)
    wanted = list(source_signals)
    if include_ip_rating and any(source == "equilibrium" for source, _ in wanted):
        wanted.append(("equilibrium", "ip_rating"))

    signals: dict[str, dict[str, np.ndarray]] = {}
    for source, signal in wanted:
        key = task_meta.signal_key(source, signal)
        signals[key] = _read_signal(root, source, signal, shot_id)
    return signals


def _read_signal(root: zarr.Group, source: str, signal: str, shot_id: int) -> dict[str, np.ndarray]:
    """Read one signal + its time axis; empty entry if absent/time-less."""
    path = f"{source}/{signal}"
    try:
        array = root[path]
    except KeyError:
        return _empty_entry()
    time_dim = _find_time_dim(array.attrs.get("_ARRAY_DIMENSIONS"))
    if time_dim is None:
        return _empty_entry()  # not time-dependent -> unusable for windowing
    try:
        times = np.asarray(root[f"{source}/{time_dim}"][:])
        values = np.asarray(array[:])
    except (KeyError, OSError, ValueError) as err:
        raise RuntimeError(f"failed reading {path} of shot {shot_id}: {err}") from err
    if values.ndim == 1:
        values = np.expand_dims(values, axis=0)
    return {"time": times, "values": values}


# ======================================================================================
# Local on-demand LRU cache (raw signals, one npz per shot)


class ShotSignalCache:
    """Disk LRU cache: one compressed npz of raw signals per shot.

    Every hit refreshes the file's mtime; after every store the oldest files
    are evicted until total size <= cap. Corrupt files are deleted and treated
    as misses. Cache stores RAW values, so normalization/flag changes never
    require invalidation.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 cap_bytes: int = DEFAULT_CACHE_CAP_BYTES) -> None:
        if cap_bytes <= 0:
            raise ValueError(f"cap_bytes must be positive, got {cap_bytes}")
        self.cache_dir = Path(cache_dir)
        self.cap_bytes = cap_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, shot_id: int) -> Path:
        return self.cache_dir / f"{shot_id}.npz"

    def load(self, shot_id: int) -> dict[str, dict[str, np.ndarray]] | None:
        """Return cached raw signals, refreshing LRU recency; None on miss."""
        path = self._path(shot_id)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                signals: dict[str, dict[str, np.ndarray]] = {}
                for name in payload.files:
                    key, _, field = name.rpartition("::")
                    if field not in ("time", "values") or not key:
                        raise ValueError(f"unexpected npz entry {name!r}")
                    signals.setdefault(key, {})[field] = payload[name]
        except (OSError, ValueError, zlib_error) as err:
            print(f"[tokamark cache] dropping corrupt {path.name}: {err}", file=sys.stderr)
            path.unlink(missing_ok=True)
            return None
        os.utime(path)  # refresh recency
        return signals

    def store(self, shot_id: int, signals: dict[str, dict[str, np.ndarray]]) -> None:
        """Atomically write one shot's raw signals, then enforce the size cap."""
        arrays = {
            f"{key}::{field}": entry[field]
            for key, entry in signals.items()
            for field in ("time", "values")
        }
        tmp_name = None
        try:
            fd, tmp_name = tempfile.mkstemp(suffix=".npz.tmp", dir=self.cache_dir)
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(tmp_name, self._path(shot_id))
        except OSError as err:  # cache is best-effort; streaming still works
            print(f"[tokamark cache] write failed for shot {shot_id}: {err}", file=sys.stderr)
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
            return
        self._evict()

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.cache_dir.glob("*.npz"))

    def _evict(self) -> None:
        """Delete oldest-mtime files until total size fits the cap (LRU).

        The newest file is never evicted, so a single shot larger than the cap
        stays cached (cap briefly exceeded) instead of thrashing.
        """
        entries = []
        for path in self.cache_dir.glob("*.npz"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in entries)
        for _, size, path in sorted(entries)[:-1]:  # oldest first; newest protected
            if total <= self.cap_bytes:
                break
            path.unlink(missing_ok=True)
            total -= size


# ======================================================================================
# Official signal-level post-processing (outliers, EFIT mask, transforms)


def remove_outlier_signals(
    signals: dict[str, Any], shot_id: int, outlier_map: dict[int, list[str]]
) -> dict[str, Any]:
    """Empty out this shot's official outlier channels (new dict; no mutation)."""
    outliers = set(outlier_map.get(shot_id, []))
    return {
        key: (_empty_entry() if key in outliers else entry)
        for key, entry in signals.items()
    }


def apply_efit_rating_mask(signals: dict[str, Any]) -> dict[str, Any]:
    """NaN-mask equilibrium samples where ip_rating == 0 (official semantics).

    If ip_rating is missing/empty, every equilibrium signal is dropped, exactly
    like ``load_signals = False`` in ``MastDataset.__getitem__``.
    """
    rating = signals.get(IP_RATING_KEY, _empty_entry())["values"]
    rating_ok = rating.size > 0
    bad_mask = (rating[0] == 0) if rating_ok else None  # rating stored as (1, T)

    out: dict[str, Any] = {}
    for key, entry in signals.items():
        if key == IP_RATING_KEY or not key.startswith("equilibrium-"):
            out[key] = entry
            continue
        if not rating_ok:
            out[key] = _empty_entry()
            continue
        values = entry["values"]
        if values.size == 0 or values.shape[-1] != bad_mask.shape[0]:
            out[key] = entry  # missing signal or clock mismatch: leave untouched
            continue
        shape = (1,) * (values.ndim - 1) + (bad_mask.shape[0],)
        out[key] = {"time": entry["time"],
                    "values": np.where(bad_mask.reshape(shape), np.nan, values)}
    return out


def apply_signal_transforms(
    signals: dict[str, Any], stats: dict[str, Any], normalize: bool = True
) -> dict[str, Any]:
    """Official per-signal transform chain for task 2-1 (new dict; no mutation).

    x_point_r/z: clip |value| > 2 to NaN BEFORE scaling; then z-score every
    signal with the official train-split mean/std. ip_rating passes through
    untouched (it is metadata, not a model signal).
    """
    out: dict[str, Any] = {}
    for key, entry in signals.items():
        values = entry["values"]
        if key == IP_RATING_KEY or values.size == 0:
            out[key] = entry
            continue
        if key in X_POINT_CLIP_KEYS:
            values = np.where(np.abs(values) > X_POINT_CLIP_LIMIT, np.nan, values)
        if normalize:
            if key not in stats:
                raise KeyError(f"no train-split stats for signal {key!r}")
            values = (values - stats[key]["mean"]) / stats[key]["std"]
        out[key] = {"time": entry["time"], "values": values}
    return out


# ======================================================================================
# Windowing — mirrors tokamark.tools.TokaMark_dataset.TokaMarkDataset exactly


class WindowBuilder:
    """Builds task windows from one shot's processed signals.

    Faithful port of ``TokaMarkDataset._process_shot`` / ``_build_window`` /
    ``_pad_*`` (markovian forecasting): identical global-time rules, NaN
    padding, ``arange`` cuts, ``int(round(...))`` index math on the stats dt,
    index clipping, and single-channel squeezing.
    """

    def __init__(self, task_config: dict[str, Any], task_metadata: dict[str, Any]) -> None:
        segmenter = task_config["task_window_segmenter"]
        self.task_type = task_config["task_type"]
        keys = task_meta.role_keys(task_config)
        self.input_keys = keys["input"]
        self.actuator_keys = keys["actuator"]
        self.output_keys = keys["output"]
        self.input_length = float(segmenter["input_length"])
        self.output_length = float(segmenter["output_length"])
        self.delta = float(segmenter["delta"])
        self.stride = float(task_metadata["stride_in_secs"])
        self.task_metadata = task_metadata
        self.data_metadata = {
            key: {"dt": meta["dt"], "shape_values": tuple(meta["values_shape"])}
            for role in ("input", "actuator", "output")
            for key, meta in task_metadata[role].items()
        }

    # ----------------------------------------------------------------------------------
    def global_times(self, sample: dict[str, Any]) -> tuple[float, float] | None:
        """(global_start, global_end) per official rules; None -> skip shot."""
        starts = []
        for key in self.input_keys + self.actuator_keys:
            first = _first_valid_time(sample[key])
            if first is not None:
                starts.append(first)
        if not starts:
            return None
        ends = []
        for key in self.output_keys:
            last = _last_valid_time(sample[key])
            if last is not None:
                ends.append(last)
        if not ends:
            return None
        start, end = float(np.min(starts)), float(np.max(ends))
        if not (np.isfinite(start) and np.isfinite(end)):
            return None
        if start + self.input_length >= end - self.delta - self.output_length:
            return None  # too short for even one window
        return start, end

    def t_cuts(self, sample: dict[str, Any]) -> np.ndarray:
        """All window cut times for one shot (empty when the shot is skipped)."""
        bounds = self.global_times(sample)
        if bounds is None:
            return np.array([])
        start, end = bounds
        return np.arange(start + self.input_length, end - self.delta - self.output_length,
                         self.stride)

    def count_windows(self, sample: dict[str, Any]) -> int:
        return int(self.t_cuts(sample).size)

    # ----------------------------------------------------------------------------------
    def iter_windows(
        self, sample: dict[str, Any], shot_id: int,
        test_mode: bool = False, filled_test_mode: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield window dicts in the official layout, in temporal order."""
        bounds = self.global_times(sample)
        if bounds is None:
            return
        start, end = bounds
        padded = self._pad_sample(sample, start, end)
        cuts = np.arange(start + self.input_length, end - self.delta - self.output_length,
                         self.stride)
        for index, t_cut in enumerate(cuts):
            window = {
                "input": self._build_window(padded, start, float(t_cut), "input"),
                "actuator": self._build_window(padded, start, float(t_cut), "actuator"),
                "output": self._build_window(padded, start, float(t_cut), "output"),
                "t_cut": float(t_cut),
                "shot_id": shot_id,
                "window_index": index,
            }
            if test_mode and not _window_valid(window, filled_test_mode):
                continue
            yield window

    # ----------------------------------------------------------------------------------
    def _build_window(
        self, sample: dict[str, Any], global_start_time: float, t_cut: float, role: str
    ) -> dict[str, dict[str, np.ndarray]]:
        """Port of ``TokaMarkDataset._build_window`` (identical index math)."""
        keys = {"input": self.input_keys, "actuator": self.actuator_keys,
                "output": self.output_keys}[role]
        out = {}
        for key in keys:
            meta = self.task_metadata[role][key]
            dt = meta["dt"]
            shape_values = tuple(meta["values_shape"])
            ts_in = int(round(self.input_length / dt))
            ts_out = int(round(self.output_length / dt))
            ts_delta = int(round(self.delta / dt))
            ts_len = {"input": ts_in, "actuator": ts_in + ts_delta + ts_out, "output": ts_out}

            times = sample[key]["time"]
            values = sample[key]["values"]
            if times.size == 0 or values.size == 0:
                selected_times = np.full(ts_len[role], np.nan)
                selected_values = np.full(shape_values + (ts_len[role],), np.nan)
            else:
                if role == "input":
                    end_idx = int(round((t_cut - times[0]) / dt))
                    if self.task_type == "markovian":
                        idx = np.arange(end_idx - ts_in, end_idx)
                    else:
                        start_idx = int(round((global_start_time - times[0]) / dt))
                        idx = np.arange(start_idx, end_idx)
                elif role == "output":
                    start_idx = int(round((t_cut + self.delta - times[0]) / dt))
                    idx = np.arange(start_idx, start_idx + ts_out)
                else:  # actuator: input span + (delta + output) span
                    cut_idx = int(round((t_cut - times[0]) / dt))
                    if self.task_type == "markovian":
                        idx_in = np.arange(cut_idx - ts_in, cut_idx)
                    else:
                        start_idx = int(round((global_start_time - times[0]) / dt))
                        idx_in = np.arange(start_idx, cut_idx)
                    idx = np.concatenate([idx_in, np.arange(cut_idx, cut_idx + ts_delta + ts_out)])
                idx = np.clip(idx, 0, len(times) - 1)
                selected_times = times[idx]
                selected_values = values[..., idx]

            if selected_values.ndim == 2 and selected_values.shape[0] == 1:
                selected_values = selected_values[0]
            out[key] = {"time": selected_times, "values": selected_values}
        return out

    # ----------------------------------------------------------------------------------
    def _pad_sample(self, sample: dict[str, Any], t_start: float, t_end: float) -> dict[str, Any]:
        """NaN-pad every signal to [t_start, t_end] on its dt grid (new dicts)."""
        padded = {}
        for key, entry in sample.items():
            times, values = entry["time"], entry["values"]
            if times.size == 0 or values.size == 0 or key not in self.data_metadata:
                padded[key] = entry
                continue
            meta = self.data_metadata[key]
            times, values = _pad_series(times, values, meta["dt"], t_start, t_end,
                                        meta["shape_values"])
            padded[key] = {"time": times, "values": values}
        return padded


def _pad_series(
    times: np.ndarray, values: np.ndarray, dt: float,
    t_start: float, t_end: float, shape_values: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``TokaMarkDataset._pad_time_series_to_interval``."""
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(float)
    if times[0] > t_start:
        n_pad = int(np.ceil((times[0] - t_start) / dt)) + 1
        times = np.concatenate([times[0] - dt * np.arange(n_pad, 0, -1), times])
        values = np.concatenate([np.full(shape_values + (n_pad,), np.nan), values], axis=-1)
    if times[-1] < t_end:
        n_pad = int(np.ceil((t_end - times[-1]) / dt)) + 1
        times = np.concatenate([times, times[-1] + dt * np.arange(1, n_pad + 1)])
        values = np.concatenate([values, np.full(shape_values + (n_pad,), np.nan)], axis=-1)
    return times, values


def _first_valid_time(entry: dict[str, np.ndarray]) -> float | None:
    """First time where any channel is non-NaN (None if signal empty/all-NaN)."""
    times, values = entry["time"], entry["values"]
    if times.size == 0 or values.size == 0:
        return None
    valid = ~np.all(np.isnan(values), axis=tuple(range(values.ndim - 1)))
    if not np.any(valid):
        return None
    return float(times[valid][0])


def _last_valid_time(entry: dict[str, np.ndarray]) -> float | None:
    """Last time where any channel is non-NaN (None if signal empty/all-NaN)."""
    times, values = entry["time"], entry["values"]
    if times.size == 0 or values.size == 0:
        return None
    valid = ~np.all(np.isnan(values), axis=tuple(range(values.ndim - 1)))
    if not np.any(valid):
        return None
    return float(times[valid][-1])


def _window_valid(window: dict[str, Any], filled: bool) -> bool:
    """Official test-mode filters (sparse by default, 'filled' variant optional)."""

    def any_nans(role: dict[str, Any]) -> list[bool]:
        return [bool(np.isnan(np.asarray(role[k]["values"])).any()) for k in role]

    def all_nans(role: dict[str, Any]) -> list[bool]:
        return [bool(np.isnan(np.asarray(role[k]["values"])).all()) for k in role]

    if filled:
        return not (all(any_nans(window["input"])) and all(any_nans(window["actuator"]))) \
            and not any(any_nans(window["output"]))
    return not (all(all_nans(window["input"])) and all(all_nans(window["actuator"]))) \
        and not any(all_nans(window["output"]))


# ======================================================================================
# Public streaming dataset


class TokamarkStreamDataset:
    """Streaming task-2-1 window dataset over anonymous S3 with local LRU cache.

    Usage::

        ds = TokamarkStreamDataset(split="train")
        for window in ds:            # dicts of numpy arrays, official layout
            ...
        ds.bytes_moved               # exact bytes fetched from S3 so far
    """

    def __init__(
        self,
        task_name: str = "task_2-1",
        split: str = "train",
        shots: list[int] | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        cache_cap_bytes: int = DEFAULT_CACHE_CAP_BYTES,
        normalize: bool = True,
        remove_outliers: bool = True,
        remove_bad_efit_rating: bool = True,
        test_mode: bool = False,
        filled_test_mode: bool = False,
        max_shots: int | None = None,
    ) -> None:
        self.task_name = task_name
        self.task_config = task_meta.load_task_config(task_name)
        self.stats = task_meta.load_signal_stats()
        self.task_metadata = task_meta.build_task_metadata(self.task_config, self.stats)
        self.source_signals = task_meta.unique_source_signals(self.task_config)
        self.window_builder = WindowBuilder(self.task_config, self.task_metadata)
        self.outlier_map = task_meta.load_outlier_metadata() if remove_outliers else {}

        self.shots = list(shots) if shots is not None else task_meta.load_split_shot_ids(split)
        if max_shots is not None:
            self.shots = self.shots[:max_shots]
        if not self.shots:
            raise ValueError("no shots to stream")

        self.cache = ShotSignalCache(cache_dir=cache_dir, cap_bytes=cache_cap_bytes)
        self.counter = ByteCounter()
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.remove_bad_efit_rating = remove_bad_efit_rating
        self.test_mode = test_mode
        self.filled_test_mode = filled_test_mode

    # ----------------------------------------------------------------------------------
    def load_shot_raw(self, shot_id: int) -> dict[str, Any]:
        """Raw signals for one shot: LRU cache first, S3 on miss (then cached)."""
        cached = self.cache.load(shot_id)
        if cached is not None:
            return cached
        signals = fetch_shot_signals(shot_id, self.source_signals, counter=self.counter)
        self.cache.store(shot_id, signals)
        return signals

    def load_shot(self, shot_id: int) -> dict[str, Any]:
        """Fully processed signals: outliers -> EFIT mask -> clip+z-score."""
        signals = self.load_shot_raw(shot_id)
        if self.remove_outliers:
            signals = remove_outlier_signals(signals, shot_id, self.outlier_map)
        if self.remove_bad_efit_rating:
            signals = apply_efit_rating_mask(signals)
        signals = apply_signal_transforms(signals, self.stats, normalize=self.normalize)
        return {key: entry for key, entry in signals.items() if key != IP_RATING_KEY}

    def iter_shot(self, shot_id: int) -> Iterator[dict[str, Any]]:
        """Yield all task windows of one shot (official layout, temporal order)."""
        sample = self.load_shot(shot_id)
        yield from self.window_builder.iter_windows(
            sample, shot_id, test_mode=self.test_mode, filled_test_mode=self.filled_test_mode
        )

    def count_shot_windows(self, shot_id: int) -> int:
        """Window count for one shot WITHOUT materializing windows (pre-filter)."""
        return self.window_builder.count_windows(self.load_shot(shot_id))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for shot_id in self.shots:
            yield from self.iter_shot(shot_id)

    @property
    def bytes_moved(self) -> int:
        """Exact bytes fetched from S3 by this dataset instance (cache misses only)."""
        return self.counter.nbytes


# ======================================================================================

def _smoke_test() -> int:  # pragma: no cover - manual entry point
    """Tiny smoke test: one shot, first window. Run verify_stream.py for the real check."""
    dataset = TokamarkStreamDataset(split="train", max_shots=1)
    shot_id = dataset.shots[0]
    started = time.time()
    n_windows = dataset.count_shot_windows(shot_id)
    window = next(iter(dataset.iter_shot(shot_id)))
    print(f"shot {shot_id}: {n_windows} windows, first t_cut={window['t_cut']:.4f}s, "
          f"{time.time() - started:.1f}s, {dataset.bytes_moved / 1e6:.2f} MB from S3")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
