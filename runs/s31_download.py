"""Doc 35 §31/§31a acquisition: the two prospective Well rows, TRAIN + VALID only (TEST stays sealed until the V1/V2
rule is locked), and their official UNetConvNext checkpoints, each pinned to the revision recorded in doc 35 §31a.
After each download every file is hashed (sha256) into runs/s31/hashes_<name>.txt; those bytes are what the models lock to.

    .venv/bin/python runs/s31_download.py        (layout as the other Well rows: data/well/<name>/, data/cnext_<name>_ckpt/)
    .venv/bin/python runs/s31_download.py --test  (doc 35 §31e phase B: the TEST files only, same pinned dataset revision,
                                                   after the prediction and arm locks; hashed into hashes_<name>_test.txt)
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/s31"
ROWS = (  # (dataset, dataset revision, checkpoint repo, checkpoint revision) -- doc 35 §31a, verified on the HF API
    ("active_matter", "dc3e9135a75b4e5a0d979086219fc97fe668f7a3",
     "polymathic-ai/UNetConvNext-active_matter", "7083c51f9762406f2a5ce0e4c3b5c74c2a83ed96"),
    ("gray_scott_reaction_diffusion", "913ef5679d5c4b4ccd6fcd52c055509dfbf190f9",
     "polymathic-ai/UNetConvNext-gray_scott_reaction_diffusion", "ae622ecbc6392e1ab807c84223f5299139f79471"),
)
SEALED = ["data/test/*", "data/test/**"]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def record(name: str, *dirs: pathlib.Path) -> None:
    lines = [f"{sha256(p)}  {p.relative_to(ROOT)}" for d in dirs for p in sorted(d.rglob("*"))
             if p.is_file() and ".cache" not in p.parts]
    (OUT / f"hashes_{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"[s31] {name}: {len(lines)} files hashed", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--test" in sys.argv:
        for name, drev, _, _ in ROWS:
            ddir = ROOT / "data/well" / name
            snapshot_download(f"polymathic-ai/{name}", repo_type="dataset", revision=drev, local_dir=ddir,
                              allow_patterns=["data/test/*"], max_workers=8)
            files = sorted(p for p in (ddir / "data/test").rglob("*") if p.is_file())
            assert files, f"{name}: no TEST files downloaded"
            (OUT / f"hashes_{name}_test.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(ROOT)}\n" for p in files))
            print(f"[s31] {name}: TEST acquired at dataset revision {drev[:10]} ({len(files)} files hashed)", flush=True)
        return
    for name, drev, ckpt, crev in ROWS:
        cdir = ROOT / f"data/cnext_{name}_ckpt"
        snapshot_download(ckpt, revision=crev, local_dir=cdir)
        ddir = ROOT / "data/well" / name
        snapshot_download(f"polymathic-ai/{name}", repo_type="dataset", revision=drev, local_dir=ddir,
                          ignore_patterns=SEALED + ["gif/*", "*.ipynb"], max_workers=8)
        assert not (ddir / "data/test").exists() or not any((ddir / "data/test").iterdir()), "TEST must stay sealed"
        record(name, cdir, ddir)
        print(f"[s31] {name}: acquired (dataset {drev[:10]}, checkpoint {crev[:10]}), TEST not downloaded", flush=True)
    print("[s31] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
