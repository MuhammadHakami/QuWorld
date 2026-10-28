#!/usr/bin/env python
"""Stream ONE MAST shot's TokaMark G2 (magnetics-dynamics) signals from the
public FAIR-MAST S3 bucket and verify anonymous access.

Access pattern (verified 2026-08-17):
    https://s3.echo.stfc.ac.uk/mast/tokamark/v1/{shot_id}.zarr/...
    - Zarr v2 store, consolidated metadata at {shot}.zarr/.zmetadata
    - Anonymous GET (no credentials, no signing) works over plain HTTPS
    - Chunks are blosc-compressed (inner codec lz4, byte-shuffle)

DEPENDENCY NOTE: the repo venv has NO s3fs / zarr / xarray / numcodecs /
aiohttp, so this script uses only urllib (stdlib) + numpy. It includes a
minimal pure-Python blosc/LZ4 decoder so small 1-D arrays can be fully
decoded and sanity-checked without any new dependencies. For real training,
install: zarr, s3fs, xarray, numcodecs (see README.md, "Dependency gap").

Usage:
    uv run --no-sync python paper/tokamark/fetch_sample.py [shot_id]
    (default shot_id: 30421, a shot used in the official TokaMark docs)

Exit codes: 0 = access verified, 1 = failure (message on stderr).
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.error
import urllib.request

import numpy as np

# ======================================================================================
# Constants (source: github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark,
# src/MAST_tools/utils/store_utils.py)

S3_HTTPS_ENDPOINT = "https://s3.echo.stfc.ac.uk"
DATASET_PATH = "mast/tokamark/v1"  # bucket "mast", prefix "tokamark/v1"
DEFAULT_SHOT_ID = 30421  # example shot from official TokaMark docs
HTTP_TIMEOUT_S = 60.0

# G2 / task 2-1 signal lists, copied verbatim from
# src/tokamark/tasks_configs/group_2_magnetics_dynamics/task_2-1.yaml
G2_INPUT_SIGNALS = (
    "magnetics/flux_loop_flux",
    "magnetics/b_field_pol_probe_ccbv_field",
    "magnetics/b_field_pol_probe_obr_field",
    "magnetics/b_field_pol_probe_obv_field",
    "magnetics/b_field_tor_probe_saddle_voltage",
    "pf_active/coil_current",
    "pf_active/solenoid_current",
    "summary/ip",
)
G2_ACTUATOR_SIGNALS = (
    "pf_active/coil_voltage",
    "summary/power_nbi",
)
G2_OUTPUT_SIGNALS_TASK_2_1 = (
    "pf_active/coil_current",
    "pf_active/solenoid_current",
    "summary/ip",
    "equilibrium/elongation",
    "equilibrium/elongation_axis",
    "equilibrium/triangularity_upper",
    "equilibrium/triangularity_lower",
    "equilibrium/x_point_r",
    "equilibrium/x_point_z",
    "equilibrium/minor_radius",
    "equilibrium/magnetic_axis_r",
    "equilibrium/magnetic_axis_z",
    "equilibrium/q95",
    "equilibrium/beta_tor",
    "equilibrium/beta_pol",
    "equilibrium/beta_normal",
    "equilibrium/bvac_rmag",
    "equilibrium/bphi_rmag",
)
# Extra outputs for tasks 2-2 / 2-3 (same inputs/actuators as 2-1)
G2_OUTPUT_SIGNALS_TASK_2_2 = ("equilibrium/lcfs_r", "equilibrium/lcfs_z")
G2_OUTPUT_SIGNALS_TASK_2_3 = ("equilibrium/psi",)

# Small 1-D arrays we fully decode as a value-level sanity check
DECODE_SIGNALS = (
    "summary/time",
    "summary/ip",
    "magnetics/time",
    "equilibrium/time",
    "equilibrium/q95",
)

# blosc v1 chunk format (c-blosc), 16-byte header
BLOSC_HEADER_SIZE = 16
BLOSC_FLAG_SHUFFLE = 0x01
BLOSC_FLAG_MEMCPYED = 0x02
BLOSC_CODEC_NAMES = {0: "blosclz", 1: "lz4", 2: "snappy", 3: "zlib", 4: "zstd"}


# ======================================================================================
# HTTP helpers


def shot_url(shot_id: int, suffix: str) -> str:
    """Build the HTTPS URL for a key inside one shot's zarr store."""
    return f"{S3_HTTPS_ENDPOINT}/{DATASET_PATH}/{shot_id}.zarr/{suffix}"


def fetch_bytes(url: str) -> bytes:
    """GET a URL anonymously; raise RuntimeError with a clear message on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": "quworld-tokamark-fetch/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {url}")
            return response.read()
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"HTTP {err.code} ({err.reason}) for {url}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error for {url}: {err.reason}") from err


def fetch_consolidated_metadata(shot_id: int) -> dict:
    """Fetch and validate the consolidated zarr v2 metadata for one shot."""
    raw = fetch_bytes(shot_url(shot_id, ".zmetadata"))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f".zmetadata for shot {shot_id} is not valid JSON") from err
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise RuntimeError(f".zmetadata for shot {shot_id} has no 'metadata' mapping")
    return metadata


# ======================================================================================
# Minimal blosc/LZ4 decoding (pure Python + numpy; no numcodecs required)


def parse_blosc_header(chunk: bytes) -> dict:
    """Parse the 16-byte c-blosc header; return an immutable-style summary dict."""
    if len(chunk) < BLOSC_HEADER_SIZE:
        raise RuntimeError(f"chunk too short ({len(chunk)} bytes) for a blosc header")
    nbytes, blocksize, cbytes = struct.unpack("<III", chunk[4:16])
    flags = chunk[2]
    header = {
        "version": chunk[0],
        "flags": flags,
        "typesize": chunk[3],
        "nbytes": nbytes,
        "blocksize": blocksize,
        "cbytes": cbytes,
        "shuffle": bool(flags & BLOSC_FLAG_SHUFFLE),
        "memcpyed": bool(flags & BLOSC_FLAG_MEMCPYED),
        "codec": BLOSC_CODEC_NAMES.get((flags >> 5) & 0x7, "unknown"),
    }
    if header["cbytes"] != len(chunk):
        raise RuntimeError(f"blosc header cbytes={cbytes} != chunk length {len(chunk)}")
    return header


def lz4_decompress_block(src: bytes) -> bytes:
    """Decode one raw LZ4 block (no frame). Pure Python; fine for small arrays."""
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        literal_length = token >> 4
        if literal_length == 15:
            while True:
                extra = src[i]
                i += 1
                literal_length += extra
                if extra != 255:
                    break
        dst += src[i : i + literal_length]
        i += literal_length
        if i >= n:
            break  # last literals: block ends without a match
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0:
            raise RuntimeError("invalid LZ4 stream: zero match offset")
        match_length = token & 0xF
        if match_length == 15:  # extension bytes follow
            while True:
                extra = src[i]
                i += 1
                match_length += extra
                if extra != 255:
                    break
        match_length += 4
        start = len(dst) - offset
        for j in range(match_length):
            dst.append(dst[start + j])
    return bytes(dst)


def unshuffle(data: bytes, typesize: int) -> bytes:
    """Invert blosc byte-shuffle: typesize byte-streams -> interleaved elements."""
    if typesize <= 1 or len(data) % typesize != 0:
        return data
    array = np.frombuffer(data, dtype=np.uint8).reshape(typesize, -1)
    return array.T.tobytes()


def decode_blosc_chunk(chunk: bytes) -> bytes:
    """Decode a full blosc chunk (memcpyed or lz4 blocks). Returns raw bytes."""
    header = parse_blosc_header(chunk)
    if header["memcpyed"]:
        body = chunk[BLOSC_HEADER_SIZE : BLOSC_HEADER_SIZE + header["nbytes"]]
        return unshuffle(body, header["typesize"]) if header["shuffle"] else body
    if header["codec"] != "lz4":
        raise RuntimeError(
            f"inner codec '{header['codec']}' not supported by this minimal decoder "
            "(install numcodecs/zarr for full support)"
        )
    nbytes, blocksize = header["nbytes"], header["blocksize"]
    n_blocks = -(-nbytes // blocksize)  # ceil division
    starts = struct.unpack(f"<{n_blocks}I", chunk[BLOSC_HEADER_SIZE : BLOSC_HEADER_SIZE + 4 * n_blocks])
    decoded = bytearray()
    for block_index, start in enumerate(starts):
        expected = min(blocksize, nbytes - block_index * blocksize)
        block_raw = decode_blosc_block(chunk, start, expected, header["typesize"])
        if header["shuffle"]:
            block_raw = unshuffle(block_raw, header["typesize"])
        decoded += block_raw
    if len(decoded) != nbytes:
        raise RuntimeError(f"blosc decode produced {len(decoded)} bytes, expected {nbytes}")
    return bytes(decoded)


def decode_blosc_block(chunk: bytes, start: int, expected: int, typesize: int) -> bytes:
    """Decode one blosc block at `start`, handling c-blosc "split" mode.

    With shuffle active, c-blosc typically splits a block into `typesize`
    sub-streams, each stored as [int32 csize][payload] back to back. Any
    (sub-)stream whose csize equals its uncompressed size is stored raw.
    Split layout is tried first (the common case here), then unsplit.
    """

    def decode_parts(part_sizes: tuple[int, ...]) -> bytes:
        position = start
        parts = []
        for part_expected in part_sizes:
            (csize,) = struct.unpack("<I", chunk[position : position + 4])
            payload = chunk[position + 4 : position + 4 + csize]
            part = payload if csize == part_expected else lz4_decompress_block(payload)
            if len(part) != part_expected:
                raise RuntimeError(f"blosc part decoded to {len(part)} bytes, expected {part_expected}")
            parts.append(part)
            position += 4 + csize
        return b"".join(parts)

    layouts = []
    if typesize > 1 and expected % typesize == 0:
        layouts.append((expected // typesize,) * typesize)  # split
    layouts.append((expected,))  # unsplit
    errors = []
    for layout in layouts:
        try:
            return decode_parts(layout)
        except (RuntimeError, IndexError, struct.error) as err:
            errors.append(f"{len(layout)}-part: {err!r}")
    raise RuntimeError(f"blosc block decode failed for all layouts ({'; '.join(errors)})")


def decode_1d_array(shot_id: int, name: str, zarray: dict) -> np.ndarray | None:
    """Fully decode a 1-D array stored as a single chunk; None if not decodable."""
    shape, chunks = zarray["shape"], zarray["chunks"]
    if len(shape) != 1 or chunks[0] < shape[0]:
        return None  # only single-chunk 1-D arrays handled here
    raw = decode_blosc_chunk(fetch_bytes(shot_url(shot_id, f"{name}/0")))
    return np.frombuffer(raw, dtype=np.dtype(zarray["dtype"]))[: shape[0]]


# ======================================================================================
# Reporting


def summarize_signals(metadata: dict, names: tuple[str, ...], label: str) -> int:
    """Print shape/chunk/dtype/compressor for each signal; return #missing."""
    print(f"\n  [{label}]")
    missing = 0
    for name in names:
        zarray = metadata.get(f"{name}/.zarray")
        if zarray is None:
            print(f"    {name:48s}  ABSENT in this shot")
            missing += 1
            continue
        compressor = (zarray.get("compressor") or {}).get("id", "none")
        dims = (metadata.get(f"{name}/.zattrs") or {}).get("_ARRAY_DIMENSIONS", "?")
        print(
            f"    {name:48s}  shape={str(zarray['shape']):16s} chunks={str(zarray['chunks']):16s}"
            f" dtype={zarray['dtype']} comp={compressor} dims={dims}"
        )
    return missing


def report_decoded_values(shot_id: int, metadata: dict) -> None:
    """Decode a few small 1-D arrays and print value-level sanity checks."""
    print("\n  [decoded values (pure-Python blosc/LZ4, no extra deps)]")
    for name in DECODE_SIGNALS:
        zarray = metadata.get(f"{name}/.zarray")
        if zarray is None:
            print(f"    {name:24s}  ABSENT")
            continue
        try:
            values = decode_1d_array(shot_id, name, zarray)
        except (RuntimeError, IndexError, struct.error) as err:
            print(f"    {name:24s}  decode failed: {err!r}")
            continue
        if values is None:
            print(f"    {name:24s}  skipped (multi-chunk or >1-D)")
            continue
        finite = values[np.isfinite(values)]
        stats = (
            f"n={values.size} range=[{finite.min():+.4g}, {finite.max():+.4g}]"
            if finite.size
            else f"n={values.size} all-NaN"
        )
        if name.endswith("/time") and finite.size > 2:
            dt = float(np.median(np.diff(finite)))
            stats += f" median_dt={dt*1e3:.3f} ms (~{1.0/dt:.0f} Hz)"
        print(f"    {name:24s}  {stats}")


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 1
    try:
        shot_id = int(argv[1]) if len(argv) == 2 else DEFAULT_SHOT_ID
    except ValueError:
        print(f"error: shot_id must be an integer, got {argv[1]!r}", file=sys.stderr)
        return 1
    if shot_id <= 0:
        print(f"error: shot_id must be positive, got {shot_id}", file=sys.stderr)
        return 1

    print(f"TokaMark G2 sample fetch — shot {shot_id}")
    print(f"  store: {S3_HTTPS_ENDPOINT}/{DATASET_PATH}/{shot_id}.zarr (anonymous HTTPS)")
    try:
        metadata = fetch_consolidated_metadata(shot_id)
    except RuntimeError as err:
        print(f"FAILED to fetch consolidated metadata: {err}", file=sys.stderr)
        return 1
    n_arrays = sum(1 for key in metadata if key.endswith("/.zarray"))
    groups = sorted({key.split("/")[0] for key in metadata if "/" in key})
    print(f"  consolidated metadata OK: {n_arrays} arrays, groups: {', '.join(groups)}")

    missing = 0
    missing += summarize_signals(metadata, G2_INPUT_SIGNALS, "task 2-1/2-2/2-3 inputs")
    missing += summarize_signals(metadata, G2_ACTUATOR_SIGNALS, "actuators")
    missing += summarize_signals(metadata, G2_OUTPUT_SIGNALS_TASK_2_1, "task 2-1 outputs")
    missing += summarize_signals(metadata, G2_OUTPUT_SIGNALS_TASK_2_2, "task 2-2 outputs")
    missing += summarize_signals(metadata, G2_OUTPUT_SIGNALS_TASK_2_3, "task 2-3 outputs")

    try:
        report_decoded_values(shot_id, metadata)
    except RuntimeError as err:
        print(f"  value decode step failed (metadata access still OK): {err}", file=sys.stderr)

    verdict = "VERIFIED" if missing == 0 else f"VERIFIED with {missing} absent signal(s)"
    print(f"\n  S3 anonymous access: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
