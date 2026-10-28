"""Fetch a DaRUS dataset (Pelzer 1HP heat plume) file by file, keeping the dataverse folder layout and checking md5.

darus-3653 reuses file names across its folders (Inputs/RUN_0-1.pt, Labels/RUN_0-1.pt, ...), so a flat download loses
two thirds of the data; this keeps every file under its directoryLabel.

    uv run python paper/geothermal/fetch_darus.py                         # darus-3653 -> data/geothermal/darus-3653/
    uv run python paper/geothermal/fetch_darus.py --doi 10.18419/darus-3650 --out data/geothermal/darus-3650
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request

API = "https://darus.uni-stuttgart.de/api"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", default="10.18419/darus-3653")
    ap.add_argument("--out", default=str(ROOT / "data/geothermal/darus-3653"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    with urllib.request.urlopen(f"{API}/datasets/:persistentId/?persistentId=doi:{a.doi}", timeout=60) as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    bad = 0
    for i, f in enumerate(files, 1):
        df = f["dataFile"]
        dst = out / f.get("directoryLabel", "") / df["filename"]
        want = df.get("md5") or (df.get("checksum") or {}).get("value")
        if dst.exists() and (not want or md5(dst) == want):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{API}/access/datafile/{df['id']}", dst)
        if want and md5(dst) != want:
            bad += 1
            print(f"md5 mismatch: {dst}")
        if i % 50 == 0:
            print(f"{i}/{len(files)} files")
    print(f"{len(files)} files in {out} ({bad} md5 mismatches)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
