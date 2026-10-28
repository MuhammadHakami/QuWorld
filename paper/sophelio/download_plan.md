# Sophelio Dataset Download Plan

Verified 2026-08-17 against the live HF repo (`HfApi.dataset_info`, files_metadata).

## What's actually there (measured, not the PR's "133 GB")

Repo: `Sophelio/fusion-equilibrium-challenge` (dataset, CC BY 4.0, public, no auth).
Layout: `data/<config>/<shot>.parquet`, **one shot per file**, hashed train filenames
(`d3d_shot_00000a10ac.parquet` — no shot-ID leakage), exactly the layout
`experiments.py --source local` expects.

| Config | Files | Size |
|---|---|---|
| `data/diii_d_train` | 7,041 | **93.4 GB** |
| `data/diii_d_public_test` | 874 | **7.2 GB** |
| `data/mast_public_test` | 1,206 | **3.2 GB** |
| Total | 9,123 | **103.9 GB** |

Disk: `/` has ~2.1 TB free (of 3.7 TB) — no problem. Destination:
`baseline/sophelio-data/` (inside `baseline/`, already gitignored).
`hf download --local-dir` writes directly to the target (no double cache copy).

## When (GPU-idle window rule)

The download is network+disk, but decompression/verification churns page cache on the
121 GiB unified-memory Spark — do **not** run it while a training job is near the
earlyoom threshold. Rule:

- **Stage 1 — test folds (10.4 GB): any time.** Small enough to be harmless; unblocks
  submission builds without streaming and MAST TTA input-statistics work.
- **Stage 2 — train (93.4 GB): first GPU-idle window ≥ ~1 h after Aug 29** (or any
  overnight-run gap before that). At HF CDN ~50 MB/s: ~35 min ideal, budget 1–3 h.
  Run it under the periodic-run-monitoring habit: check the log mid-flight.

## How

From `baseline/sophelio-starter/` (its venv ships the `hf` CLI):

```bash
DEST=/home/mohhakami/Documents/paper/quworld/baseline/sophelio-data

# Stage 1 — both public test folds (10.4 GB)
.venv/bin/hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset \
    --include "data/mast_public_test/*" "data/diii_d_public_test/*" \
    --local-dir "$DEST" 2>&1 | tee -a "$DEST/download.log"

# Stage 2 — DIII-D train (93.4 GB), GPU-idle window, background + log
nohup .venv/bin/hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset \
    --include "data/diii_d_train/*" \
    --local-dir "$DEST" >> "$DEST/download.log" 2>&1 &
```

Interrupted? **Re-run the same command** — `hf download` resumes/skips completed files.

## Verify

```bash
for c in diii_d_train diii_d_public_test mast_public_test; do
  echo "$c: $(ls "$DEST/data/$c" | wc -l) files"; done
# expect 7041 / 874 / 1206
du -sh "$DEST"           # expect ~104 GB
# spot-check one file parses and has the full schema
.venv/bin/python -c "import pandas as pd, glob; \
  r = pd.read_parquet(sorted(glob.glob('$DEST/data/diii_d_train/*.parquet'))[0]).iloc[0]; \
  print(r['source'], len(r['efit_times']), len(r.index))"
```

Then everything runs streaming-free:

```bash
.venv/bin/python experiments.py --source local --local-data-dir "$DEST" --n-shots 50
```

(`local_score.py` hard-codes HF streaming; for fully-offline scoring use our
`my_experiments/` drivers, or point a small loader at `$DEST` — same row schema.)

## Non-goals

- No bulk re-downloads: the repo pins revisions; if organizers push a data fix,
  `hf download` re-fetches only changed files.
- Streaming (`load_dataset(..., streaming=True)`) stays the tool for ad-hoc peeks; the
  local copy is for training and submission builds.
- **Final phase (Oct 19–26)**: private-fold inputs will be a *new* config/repo path —
  grab them the hour they appear (they gate our 3 final submissions); expect ~test-fold
  size (~10 GB).
