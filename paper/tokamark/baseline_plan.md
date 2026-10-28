# TokaMark G2 baseline plan — CNN+LSTM 0.1343 repro vs. frozen-backbone + 30q adapter

Companion to `README.md` (sources, verified facts) and `dataset.py` (our
streaming loader, verified end-to-end on 2026-08-17: 3 train shots, 1,678
windows, all shapes checked, 6.32 MB moved, cold ~13 s/shot, warm ~0.01 s).
Everything below that cites the baseline code was read directly from
`github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark_baseline` (`main`) on 2026-08-17.

## 1. What "0.1343" actually is

Group-2 NRMSE, random split, CNN+LSTM (paper v3 Table 3) = **equal-weight mean
of three tasks**: task 2-1 **0.2197**, task 2-2 **0.0568**, task 2-3 **0.1263**
(mean = 0.13427). Reproducing the group headline therefore requires training
**three models**, not one. The per-task target for the headline magnetics
forecasting task is 0.2197.

Facts from their code that any repro must copy:

| setting | value | source |
|---|---|---|
| train/val stride | **0.005 s** (overridden; NOT the task's 0.001 s) | `run_training.py` (`config_task["stride_window"] = 0.005` for tasks 1-x/2-x/3-1/3-2) |
| eval stride | 0.001 s (task YAML, no override) + `test_mode=True` (sparse window filter) | `run_evaluation.py` |
| shuffle | buffer 2048, shot-level iterable stream | `run_training.py` |
| batch / workers | 512 / 8 | `src/config/config_model.yaml` + task override |
| optimizer | Adam lr 1e-4, weight decay 1e-4 | config + `src/trainer.py` |
| stopping | validate every 100 batches, patience 10, `max_steps` 1e7 (early stop governs) | `run_training.py`, config |
| model (lstm) | per-modality Conv1D/2D/3D encoders → latent D=16 → seq2seq LSTM backbone → conv decoders | `src/multi_conv_lstm_model.py` |
| normalization | z-score with `random_split_signals_stats.yaml` (train-split stats) | `initialize_MAST_dataset(use_std_scaling=True)` |
| filtering | `remove_outliers=True`, `remove_bad_efit_rating=True` | same |
| seed | default 23, one seed reported | `run_training.py` |

Windows/epoch at train stride 0.005 (measured on our 3 verified shots:
511/703/464 windows at 1 ms → ~102/141/93 at 5 ms, mean ≈ 112/shot):
~8,268 usable train shots × ~112 ≈ **0.93 M windows/epoch** → ~1,800 steps/epoch
at batch 512. (Paper hardware: one A100-SXM4-80GB; no GPU-hours given.)

## 2. Exact repro steps (their stack, unmodified)

CPU/network prep (no GPU, can run now):

1. Clone both repos **outside** our pyproject (keep our venv clean):
   `git clone .../tokamark && git clone .../tokamark_baseline` under
   `~/external/` (not inside quworld).
2. Env: their `environment_basic.yml` (conda) or a dedicated uv venv;
   `pip install -e ./tokamark` plus `line-profiler==5.0.2 torchinfo==1.8.0`.
   All deps have aarch64 wheels (torch already on the box). **Do not** `uv add`
   anything into quworld for this.
3. Point the store at S3 streaming instead of their CSD3 path: in
   `tokamark_baseline/src/config/config_model.yaml` set `local: false`
   (their `store_manager_settings` default then uses
   `s3://mast/tokamark/v1` anon + `simplecache`, verified anonymous).
   `subset_of_shots: null` = full split.
4. Data warm-up (CPU/network only, avoids I/O-starving the GPU run): prefetch
   task 2-1 signals for train+val shots through **our** `dataset.py` LRU cache
   — measured 2.1 MB/shot fetched, 1.7 MB/shot cached npz →
   train+val+test ≈ 10.3 k shots ≈ **22 GB fetched / ~18 GB cached**.
   This needs the cache cap raised from 5 GiB to ~20 GiB — **needs approval**,
   but note it is the task-2-1 subset only (~1% of the 2.7 TB dataset), disk-
   backed, one shot in RAM at a time (~15 MB) — not the OOM-class bulk mirror.
   With 8 parallel prefetch workers at the observed ~13 s/shot sequential
   latency → ~1.6 s/shot → **~4.5 h one-time**. (Their `simplecache` dir would
   duplicate this; symlink or let it re-pull — decision at run time.)

GPU phases (order matters; each gates the next):

5. **Anchor (≤1 GPU-h, do FIRST)**: mean predictor through their
   `run_evaluation.py` path (or our evaluator port) on task 2-1 test split.
   Expected group-2 mean-predictor NRMSE 0.9106 (task-level value from their
   artifacts). If we don't match ±0.01, the pipeline is miswired — stop, fix,
   never train on a broken evaluator.
6. **Pilot (~2–4 GPU-h)**: `run_training.py --task task_2-1 --config
   /src/config/config_model.yaml --model lstm --split random --seed 23` with
   `subset_of_shots` ≈ 10% . Calibrate: steps/s, dataloader
   starvation (their windowing is CPU-heavy at 8 workers), val-loss curve.
7. **Full task 2-1 (~15–40 GPU-h on GB10)**: same command, full shots. GB10 vs
   A100 is ~3–6× slower memory-bandwidth-bound; early stopping governs actual
   wall time; the pilot's steps/s makes this estimate concrete before we commit.
   Success = test NRMSE 0.2197 ± 0.01 (their seed-23 number, single seed).
8. **Only if the group headline is required**: tasks 2-2 (+`ReshapeLcfsTransform`)
   ~15–40 GPU-h and 2-3 (65×65 psi maps, Conv2D/3D branches, heavier decode)
   ~25–60 GPU-h. Full group repro ≈ **40–140 GPU-h for one seed**.

## 3. Our model on the same rails (frozen backbone + 30q adapter)

Same data contract as the repro — official splits CSV, official train-split
stats, official windowing (all already wired in `dataset.py`, verified):

1. Wrap `TokamarkStreamDataset` in a torch `IterableDataset` + collate that
   stacks the multi-clock window dict into per-signal tensors (the loader is
   deliberately torch-free; ~50-line adapter at train time, use their
   `cnn_collate_fn` NaN handling as the reference).
2. Architecture: keep the baseline's multi-branch conv encoders/decoders
   (identical I/O contract, so any gain is attributable to the backbone),
   replace the LSTM backbone with our **frozen pretrained backbone + 30q
   adapter** (phase-B pattern: vanilla / mlp30 / mr30ryd ablation trio); train
   adapter + decoders only.
3. Train at their stride 0.005, batch 512, Adam 1e-4, patience 10 — matched-
   budget fairness; eval with stride 0.001 `test_mode=True` + their
   `evaluator.py` aggregation (window→shot→signal→task equal-weight means).
4. Cost: frozen backbone cuts backward cost; data pipeline dominates →
   **~10–30 GPU-h/seed**, 3 seeds + matched control (adapter-less or MLP-
   parameter-matched, as in phase B) → **~60–120 GPU-h** for the full ablation
   on task 2-1.
5. Stats: bootstrap over test shots (their aggregation is an equal-weight mean
   over shot rows, so shot-level resampling is the natural paired test), 3
   seeds, report random + temporal splits (temporal is where CNN+LSTM degrades
   2.2× → 0.2928 — our best shot at a headline win).

## 4. Decision point: reproduce theirs vs. cite theirs

**Recommendation: Option B — cite their numbers, validate the rails with the
mean predictor, spend GPU on our model; spot-check CNN+LSTM on task 2-1 only.**

| | Option A: full repro | Option B: cite + spot-check |
|---|---|---|
| GPU cost before our contribution | 40–140 h (3 tasks, 1 seed) | ≤1 h anchor (+15–40 h optional spot-check) |
| Evidence quality | strongest ("we reproduced 0.1343") | standard practice (official splits/stats/evaluator + published numbers) |
| Risk | burns the GB10 budget on someone else's model; stride/config drift can silently miss 0.2197 | reviewer may ask for a repro of at least one task |

Decision rule (pre-registered):
- Gate 1 (mandatory, ≤1 GPU-h): mean predictor through the official evaluator
  matches 0.9106 ± 0.01 → the rails are trustworthy. If not, debug; nothing
  else proceeds.
- Gate 2: run OUR pilot (step 6 rails, our model, 10% shots). If the adapter
  ablation shows signal, commit the ~60–120 GPU-h to ours; queue the CNN+LSTM
  task-2-1 spot-check (single seed, 0.2197 ± 0.01) only when the GPU is
  otherwise idle.
- We compare against their published per-task numbers on the **identical**
  test windows; the group-2 headline (0.1343) is cited, and we report our own
  group-2 mean only if we run all three tasks (2-2/2-3 support in `dataset.py`
  = add the two task YAMLs + the LCFS reshape transform; ~1 day CPU work, no
  new data risk — psi adds ~3.8 MB/shot).

Total budget, Option B: **~65–165 GB10 GPU-hours** end-to-end (ours 3 seeds +
control + anchors + optional spot-check), vs ~180–280 for Option A + ours.

## 5. Known gaps / blockers

- `dataset.py` implements task 2-1 only (2-2/2-3: two YAMLs + LCFS reshape
  port pending; stats YAML already covers their signals).
- Cache cap 5 GiB holds ~3,000 shots (~1/3 of train split); full-split training
  without churn needs ~18 GiB — approval needed before raising.
- The tokamark/tokamark_baseline clones + `pip install -e` live outside the
  quworld venv (no `uv add` performed; nothing missing inside our venv for
  `dataset.py` itself — zarr 3.3.0 / s3fs / fsspec / numcodecs / pyyaml all
  present and verified).
- S3 latency from this site is variable (sub-second to minutes per object on
  bad days — README §6); prefetch before GPU runs, never stream cold during
  training.
- No baseline training time is published — every GPU-hour figure above is an
  estimate to be recalibrated from the step-6 pilot before committing seeds.
