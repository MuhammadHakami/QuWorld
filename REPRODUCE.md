# REPRODUCE.md — every claimed number → the command that regenerates it

Companion to `paper/manuscript/main.tex` (WM-PAI 2026 submission). Rule of
engagement ([11-master-plan.md](11-master-plan.md)): **no notebook, no claim** —
every number in the manuscript maps to a row below. Pending manuscript macros
(`\TRLquantum`, `\TDpvalue`, ...) list the command that *will* produce them.

## Environment

- Single NVIDIA DGX Spark GB10 workstation, 121 GiB unified memory, aarch64.
- Python via [uv](https://docs.astral.sh/uv/): every command below is
  `uv run ...` from the repo root (`pyproject.toml` + `uv.lock` pin the env).
- Exception: the TORAX-Bench generator runs in a separate TORAX v1.4.3 venv
  (JAX, CPU, float64) — marked `<torax-python>` below.
- Paper build: `~/.local/bin/tectonic` (v0.15.0).
- Data under `data/` (not in git): The Well `turbulent_radiative_layer_2D`
  at `data/trl2d/`, official checkpoint
  [`polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D`](https://huggingface.co/polymathic-ai/UNetConvNext-turbulent_radiative_layer_2D)
  at `data/cnext_ckpt/`, MHD_64 at `data/mhd64/`.

## Claim → command index

| # | Claimed number (manuscript location) | Value | Command / artifact |
|---|---|---|---|
| 1 | Reproduced CNextU-net anchor, one-step test mean-VRMSE over 873 official windows (abstract, §2, §4, §5, Tab. campaign) | **0.1934** | `uv run paper/phase_b/eval_official.py` → `runs/eval_official_cnext.csv` (per-window) |
| 2 | Pilot sprint table (Tab. pilot): rounds 1–2 rows | 0.419/1.143 (none), 0.425/1.213 (mlp), 0.404/0.876 (hea12), 0.432/0.913 (ryd16), fixed-2500 rows | `uv run paper/train.py <variant> 1337 300` (wall-clock) / `uv run paper/train.py <variant> 1337 900 2500` (fixed steps); logs `runs/round1_*.log`, `runs/round2*_*.log`; ledger rows in `results.tsv` |
| 3 | Persistence floor 0.583 / 1.002 (Tab. pilot) | 0.583, 1.002 | `notebooks/02_baselines_evaluation.ipynb` (executed; rebuilt by `uv run paper/build_notebooks.py`) |
| 4 | 3-seed sprint study (Tab. pilot bottom block) | none 1.111±0.174 · mlp30 1.123±0.211 · mr30hea 1.063±0.152 · mr30ryd **0.955±0.102** | `bash paper/run_round3.sh` (3 seeds × 4 variants, 5-min budget); logs `runs/round3_<variant>_s<seed>.log`; aggregate = mean±std of `val_vrmse_8` over seeds 1337/1338/1339 from `results.tsv` |
| 5 | Analog qubit-scaling trend 12q 1.168 → 16q 0.913 → 30q 0.955±0.102 (§5, H1c) | as listed | ryd12: `runs/round2_ryd12.log`; ryd16: `runs/round1_ryd16.log`; 30q: row 4 above — all in `results.tsv` |
| 6 | Wave-1.5 sprint arena (Tab. wave15, Tab. ladder): quantum+TD **0.845±0.051** vs MLP 0.953±0.135 (3/3 seeds) vs none 0.957±0.255; k=1 column; TD-gain interaction −23%/−10%/≈0% | all 8 cells | `bash paper/run_ag_wave15.sh` (fixed 2,000 steps × {none, mlp30, mr30ryd, cg30} × {k=1, k=4} × seeds {1337,1338,1339}); logs `runs/ag15_<variant>_k<k>_s<seed>.log`; rows tagged `wave15 fixed2000` in `results.tsv`; aggregation notebook `notebooks/15_adaptive_geometry.ipynb` |
| 7 | Phase-B pilot validation ordering 0.18977 < 0.18985 < 0.19009 (§Results, campaign) | as listed | `bash paper/phase_b/run_pilot_detached.sh`; logs `runs/phase_b/pilot_{mr30ryd,vanilla,mlp30}.log` |
| 8 | Campaign round 1 macros `\TRLvanilla` `\TRLquantum` `\TRLmlp` `\TRLpvalue` `\TRLwinrate` | pending | `bash paper/phase_b/run_overnight.sh` (2 h × {vanilla, mr30ryd, mlp30} × 3 seeds, ~20k steps, + mr30hea s1337); then per checkpoint `uv run paper/phase_b/eval_official.py runs/phase_b/<variant>_s<seed>.pt runs/phase_b/test_<...>.csv`; then `uv run paper/phase_b/paired_stats.py runs/phase_b/test_<...>.csv runs/eval_official_cnext.csv` → `runs/phase_b/paired_stats_report.txt` |
| 9 | Campaign round 2 (TD, k=4) macros `\TDvanilla` `\TDquantum` `\TDmlp` `\TDpvalue` `\TDwinrate` | pending | `bash paper/phase_b/run_round2_td.sh 10000` → checkpoints, test CSVs, and `runs/phase_b2/paired_stats_report.txt` |
| 10 | TORAX-Bench v1: 1,000 discharges, 121 frames, 12-dim actuator space, 44 MB, frozen 800/100/100 splits (§Artifacts, Tab. ladder) | dataset | `<torax-python> paper/torax_bench/generate_v1.py 1000 data/torax_bench_v1` (seed 20260816, resumable); splits `paper/torax_bench/splits.json`; card `paper/torax_bench/CARD.md`; manifest `data/torax_bench_v1/manifest.json` |
| 11 | TORAX-Bench baselines: persistence@8 **0.0923**, climatology 0.5546 (§Artifacts, Tab. ladder) | as listed | `python paper/torax_bench/baselines.py data/torax_bench_v1 paper/torax_bench/splits.json` (numpy-only, CPU) |
| 12 | MHD_64 ladder row (anchor 0.1633 published; "pipeline ready") | pending | anchor: `WELL_DATASET=MHD_64 uv run paper/phase_b/eval_official.py data/cnext_mhd_ckpt runs/eval_official_cnext_mhd.csv`; loader equivalence: `uv run python -c "import sys; sys.path.insert(0,'paper/phase_b'); import data_ram_mhd; data_ram_mhd.verify()"`; full commands in `paper/phase_b/README_mhd.md` |
| 13 | Quantum-core parameter counts, matched-params claim (±0.3 % sprints; 0.0007 % Phase B) | — | printed at startup by `paper/train.py` (via `paper/model.py`) and `paper/phase_b/finetune.py` (via `paper/phase_b/cnext_hybrid.py`, incl. bit-identical-at-init assert) |
| 14 | Simulator cross-check to 1e-6 (§Method) | — | `paper/qlayer.py` self-test block (`uv run python paper/qlayer.py`) |

## Published / external numbers (not ours — cited, not regenerated)

| Number | Source |
|---|---|
| CNextU-net TRL2D one-step 0.1956; rollout 0.54 (6:12) / 1.01 (13:30); MHD_64 0.1633; TFNO rayleigh_benard 0.6566 / planetswe 0.0853 | The Well (arXiv:2412.00568), Tab. 4/5 |
| WaveLiT 0.1308 / 0.1346 (median variant 0.1167) | arXiv:2605.25949 (protocol caveats in manuscript §2) |
| Walrus long-horizon behavior | arXiv:2511.15684 |
| Physics-IQ top score ≈ 63.4 (Cosmos-class) | Physics-IQ leaderboard (arXiv:2501.09038) |
| TokaMark G2 CNN+LSTM NRMSE 0.1343 (random split) | TokaMark benchmark release |
| Sophelio Ch1 d3d_S 0.9932 / Ch2 G_ratio ≈ 0.654 | Sophelio challenge leaderboard (Codabench) |
| Pelzer 1HP U-Net MAE 0.046 °C | Pelzer & Schulte heat-plume papers |
| HQ-LP-FNO lost to matched control | arXiv:2604.04828 |

## Layout

- **`results.tsv`** — append-only run ledger: `commit · val_vrmse_1 ·
  val_vrmse_8 · memory_gb · status · description`. Every sprint row in the
  manuscript exists here with the git commit that produced it.
- **`runs/`** — raw logs + status files: `round{1,2,3}_*` (sprints),
  `ag15_*` (wave 1.5), `ag_status` (wave-1.5 runner heartbeat).
- **`runs/phase_b/`** — round-1 campaign: `ft_<variant>_s<seed>.log`
  (training), `<variant>_s<seed>.pt` (best-on-valid checkpoint; `*_latest.pt`
  resumable state), `test_*.{log,csv}` (single-shot test evals),
  `overnight_status` (heartbeat), `paired_stats_report.txt` (final stats).
- **`runs/phase_b2/`** — round-2 TD campaign, same layout (`td_*` prefixes).
- **`runs/eval_official_cnext.csv`** — the frozen anchor: per-window VRMSE of
  the official checkpoint (mean 0.1934); the baseline argument of every
  `paired_stats.py` call.
- **Monitoring:** `uv run paper/check_runs.py <prefix>` (e.g. `phase_b/ft_`,
  `ag15_`) — crash/NaN/stall triage for any queue.

## Notebooks (one per challenge; executed = evidence)

| Notebook | Contents |
|---|---|
| `notebooks/01_task_analysis.ipynb` | TRL2D benchmark anatomy |
| `notebooks/02_baselines_evaluation.ipynb` | all baseline numbers incl. persistence floor 0.583/1.002 |
| `notebooks/03_our_solution.ipynb` | architecture, live hypothesis verdicts |
| `notebooks/04_competitors.ipynb` | published-competitor audit (`competitors_data.json`) |
| `notebooks/05_trl2d_challenge.ipynb` | Phase A: anchor repro + campaign tracking |
| `notebooks/15_adaptive_geometry.ipynb` | wave-1/1.5 ledger, fair-pairing verdicts (Tab. wave15) |
| `notebooks/16_self_improvement.ipynb` | structure bake-off / self-improvement loop |
| `notebooks/autoresearch.ipynb` | living report across rounds |

Notebooks 01–03 are rebuilt+re-executed by `uv run paper/build_notebooks.py`.
Numbers 06–14 (MHD, world-model adapter, physics judges, rayleigh_benard,
dequantization, TokaMark, TORAX adapters, geothermal, Sophelio) are created as
their phases complete — see [11-master-plan.md](11-master-plan.md); their
ladder rows stay "pending" in the manuscript until the notebook exists.

## Paper

```bash
cd paper/manuscript && ~/.local/bin/tectonic main.tex   # → main.pdf
```

Placeholder macros live at the top of `main.tex` (block "RESULTS PLACEHOLDER
MACROS"); each renders as a red `[...]` marker until filled from
`runs/phase_b/paired_stats_report.txt` (round 1) and
`runs/phase_b2/paired_stats_report.txt` (round 2). A grep gate before
submission: `grep -n "pending{" paper/manuscript/main.tex` must return only
macro *definitions*, and the compiled PDF must contain no red markers.
