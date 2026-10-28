"""Per-seed results for every scored row, read from the result files themselves (never from the ladder prose).

Each loader returns {arm: {seed: value}} with arms "trunk", "mlp", "q". Rows are declared once in ROWS with
their metric direction, the published reference, and whether they are PRIMARY (one per benchmark, chosen a
priori as the final-protocol / strongest-trunk row) or SECONDARY (sensitivity: older protocol, weaker trunk,
single seed, or a configuration selected on the same data it is scored on).
"""
from __future__ import annotations

import csv
import glob
import json
import pathlib
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEEDS = (1337, 1338, 1339)


def _j(p):
    return json.load(open(ROOT / p))


def _arena(alias_map: dict[str, str], pattern: str) -> dict:
    out = {}
    for arm, alias in alias_map.items():
        vals = {}
        for f in glob.glob(str(ROOT / pattern.format(alias=alias))):
            m = re.search(r"val_vrmse_8:\s+([0-9.]+)", open(f).read())
            if m:
                vals[int(re.search(r"_s(\d+)\.log$", f).group(1))] = float(m.group(1))
        out[arm] = vals
    return out


def trl2d_arena():
    return _arena({"trunk": "none", "mlp": "mlp30", "q": "mr30ryd"}, "runs/ag15_{alias}_k4_s*.log")


def trl2d_waveg_shared():
    shared = _arena({"q": "q30-link=shared", "mlp": "q30-link=shared-mlp"}, "runs/waveG/{alias}_s[0-9]*.log")
    return {"trunk": trl2d_arena()["trunk"], **shared}


def _mhd_steps(v: str, s: int) -> int | None:
    import torch
    f = ROOT / f"runs/phase_b_mhd/{v}_s{s}_latest.pt"
    return int(torch.load(f, map_location="cpu", weights_only=False)["step"]) if f.exists() else None


def mhd(length_matched: bool = True):
    """MHD test VRMSE per seed. 4 of the 9 runs stopped at ~7.5k of the 20k-step budget (restart attempts,
    one INCOMPLETE), and their best checkpoint is their last, so arms of one seed can differ only by training
    length. With `length_matched`, a seed is kept only if all three arms trained within 5% of the same steps."""
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    out = {arm: {} for arm in arms}
    for s in SEEDS:
        steps = [_mhd_steps(v, s) for v in arms.values()]
        if length_matched and (None in steps or max(steps) > 1.05 * min(steps)):
            continue
        for arm, v in arms.items():
            f = ROOT / f"runs/phase_b_mhd/test_{v}_s{s}.csv"
            if f.exists():
                out[arm][s] = st.mean(float(r["vrmse"]) for r in csv.DictReader(open(f)))
    return out


def well(ds: str):
    recs = _j(f"runs/well2d/{ds}/cnext/csv_audit.json")["records"]
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    return {arm: {int(r["name"].split("_s")[1]): r["mean_vrmse"] for r in recs if r["name"].startswith(v + "_s")}
            for arm, v in arms.items()}


def torax_fixed():
    arms = {"trunk": "none", "mlp": "mlp30", "q": "mr30ryd"}
    return {arm: {s: _j(f"runs/torax/{v}_k4_s{s}.json")["test_vrmse_8"] for s in SEEDS} for arm, v in arms.items()}


def pelzer_fixed():
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    return {arm: {s: _j(f"runs/pelzer_unet/{v}_s{s}.json")["test"]["mae_c"] for s in SEEDS} for arm, v in arms.items()}


def sophelio_fixed():
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    return {arm: {s: _j(f"runs/sophelio/{v}_s{s}.json")["official"]["S"] for s in SEEDS} for arm, v in arms.items()}


def hpo(bench: str, sign: float = 1.0):
    """Tuned rows: every arm selected by the same two-seed VALID objective (study 'ms2'). If the trunk's
    ms2 selection has not landed yet, the round-1 trunk is used and the row is flagged by `hpo_protocol`."""
    out = {}
    for arm, key in (("trunk", "none"), ("mlp", "mlp"), ("q", "q")):
        f = ROOT / f"runs/hpo/{bench}/{key}_ms2_final.json"
        f = f if f.exists() else ROOT / f"runs/hpo/{bench}/{key}_final.json"
        out[arm] = {r["seed"]: sign * r["metric"] for r in json.load(open(f))["per_seed"]} if f.exists() else {}
    return out


def hpo_protocol(bench: str) -> str:
    return "same (ms2)" if (ROOT / f"runs/hpo/{bench}/none_ms2_final.json").exists() else "MIXED: trunk round-1"


def tokamind(seeds=SEEDS):
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    out = {}
    for arm, v in arms.items():
        out[arm] = {}
        for s in seeds:
            f = ROOT / f"runs/tokamark/results/tokamind_task_1-1_{v}_s{s}.json"
            if f.exists():
                out[arm][s] = list(_j(f.relative_to(ROOT))["task_metrics"].values())[0]["task_1-1"]["NRMSE_mean"]
    return out


def replication(bench: str, sign: float = 1.0):
    """New replication units (31-powered-replication-prereg.md), keyed by (trunk seed, seed)."""
    out = {"trunk": {}, "mlp": {}, "q": {}}
    for f in sorted((ROOT / "runs" / "replication" / bench).glob("*.json")):
        d = json.load(open(f)); arm = {"none": "trunk", "mlp": "mlp", "q": "q"}[d["arm"]]
        out[arm][(d.get("trunk_seed"), d["seed"])] = sign * d["metric"]
    return out


REPLICATION_SEEDS_TOKAMIND = (1340, 1341, 1342, 1343)


def tokamark_cnn():
    arms = {"trunk": "vanilla", "mlp": "mlp30", "q": "mr30ryd"}
    out = {}
    for arm, v in arms.items():
        rows = list(csv.DictReader(open(ROOT / f"runs/tokamark/results/random/cnn_{v}/task_1-1/seed_23/task_1-1/task_metrics.csv")))
        out[arm] = {23: float(r["NRMSE_mean"]) for r in rows if r["feature_name"] == "task_1-1"}
    return out


def tdc(endpoint: str):
    """TDC reports mean±sd over its 5 official seeds only; per-seed values are not exposed, so these rows
    enter the meta-analysis as summary statistics (see judge.py)."""
    out = {}
    for arm, v in (("trunk", "vanilla"), ("mlp", "mlp30"), ("q", "mr30ryd")):
        f = ROOT / f"runs/tdc/{endpoint}_{v}.json"
        if f.exists():
            d = json.load(open(f)); out[arm] = (d["test_mean"], d["test_sd"], len(d["seeds"]))
    return out


# name, loader, lower_is_better, published reference, reference name, tier, note
ROWS = [
    ("TRL2D arena VRMSE@8", trl2d_arena, True, None, "internal arena (no external SoTA)", "primary",
     "validation split of the arena; fixed 2k steps, TD k=4"),
    ("MHD_64 VRMSE", mhd, True, 0.1633, "CNextU-net (The Well paper)", "primary",
     "official test, 960 windows; LENGTH-MATCHED seeds only (1337: all 20k steps, 1339: all ~7.5k; s1338 dropped, MLP truncated at 7.5k vs 20k); adapter gates |α|≈4e-5 → inert"),
    ("MHD_64 VRMSE, all seeds (unequal training length)", lambda: mhd(False), True, 0.1633, "CNextU-net", "secondary",
     "INVALID comparison kept for audit: s1338 MLP trained 7.5k vs 20k steps"),
    ("rayleigh_benard VRMSE", lambda: well("rayleigh_benard"), True, 0.0415, "PDE-FM (arXiv:2511.21861; The Well TFNO 0.6566)", "primary",
     "official test; official CNextU-net checkpoint reproduced at 0.6679 (published 0.6699)"),
    ("shear_flow VRMSE", lambda: well("shear_flow"), True, 0.0345, "PDE-FM (arXiv:2511.21861; The Well CNextU-net 0.8080)", "primary",
     "official test; our evaluator scores the official checkpoint 0.9946 → published anchor NOT reproduced"),
    ("TORAX-Bench VRMSE@8 (tuned)", lambda: hpo("torax"), True, 0.129, "persistence (matched protocol)", "primary",
     "Optuna, two-seed VALID objective, 16 trials/arm"),
    ("Pelzer 1HP MAE °C (tuned)", lambda: hpo("pelzer"), True, 0.046, "Pelzer & Schulte U-Net", "primary",
     "Optuna, two-seed VALID objective, 12 trials/arm"),
    ("Sophelio S (tuned)", lambda: hpo("sophelio", -1.0), False, 0.85, "award-eligibility bar (not a SoTA)", "primary",
     "Optuna, two-seed VALID objective, 10 trials/arm"),
    ("TokaMark 1-1 NRMSE (TokaMind trunk)", tokamind, True, 0.150, "TokaMind FT-Base (SoTA)", "primary",
     "their warm-start protocol, 1,020 test shots"),
    ("TORAX-Bench VRMSE@8 (fixed)", torax_fixed, True, 0.129, "persistence", "secondary", "fixed 1k-step protocol"),
    ("Pelzer 1HP MAE °C (fixed)", pelzer_fixed, True, 0.046, "Pelzer & Schulte U-Net", "secondary", "fixed protocol"),
    ("Sophelio S (fixed)", sophelio_fixed, False, 0.85, "award bar", "secondary", "starter-kit protocol"),
    ("TokaMark 1-1 NRMSE (CNN trunk)", tokamark_cnn, True, 0.2091, "their CNN baseline", "secondary",
     "their fixed seed 23: n=1"),
    ("TRL2D arena, promoted q30-link=shared", trl2d_waveg_shared, True, None, "internal", "secondary",
     "configuration selected on this arena → selection-biased"),
    # pre-registered powered replication: NEW units only (31-powered-replication-prereg.md)
    ("REP Pelzer 1HP MAE °C", lambda: replication("pelzer"), True, 0.046, "Pelzer & Schulte U-Net", "replication",
     "3 trunks x 4 adapter seeds, ms2 configs frozen"),
    ("REP Sophelio S", lambda: replication("sophelio", -1.0), False, 0.85, "award bar", "replication",
     "8 new seeds, ms2 configs frozen"),
    ("REP TokaMark 1-1 NRMSE (TokaMind trunk)", lambda: tokamind(REPLICATION_SEEDS_TOKAMIND), True, 0.150,
     "TokaMind FT-Base (SoTA)", "replication", "4 new seeds, their warm-start protocol"),
]
