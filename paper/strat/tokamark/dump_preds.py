"""§37 TokaMark: one data pass, many members -> per-window EFIT predictions on a fixed VALID sample or TRAIN subsample.

TEST is never loaded: the MAST dataset is built only from the shot ids in runs/strat/tokamark/strata.json (VALID sample)
or shot_meta.json["train_sub"] (TRAIN), and those ids are checked against the split CSV's train/val columns.
TokaMind's own eval pipeline is used unchanged (eval-phase windowing with test_mode=True, their tokenizer, their
forward_decode_native), so per-window errors are the benchmark's. Members that share a tokenizer (codec set) share one
data pass: group "scratch" (3 TokaMind-Scratch runs) or "ft" (FT-Base runs). Config snapshots / logs are redirected to
runs/strat/tokamark/cfg/ so no member run directory is modified.

  flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_tok 16 .venv-tokamark/bin/python \
      paper/strat/tokamark/dump_preds.py --group scratch --split valid [--max-shots N] [--members a,b]
  -> runs/strat/tokamark/preds_<group>_<split>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from paper.gpucap import cap_from_env  # noqa: E402

cap_from_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "paper" / "tokamark"))
import tokamind_hybrid as TH  # noqa: E402  (adds scripts_mast + paper/ to sys.path; wrap_heads, _mutate_config)
import mast_utils.config.loader as LOADER  # noqa: E402
from mast_utils.config.finalize import compute_paths  # noqa: E402
from mast_utils import (  # noqa: E402
    load_experiment_config, load_task_definition, build_signals_by_role_from_task_definition, extract_signal_stats,
    init_run_context, build_window_data, build_model_and_optional_warmstart, resolve_eval_embeddings)
from mast_utils.benchmark_imports import get_task_metadata  # noqa: E402
from mast_utils.tokamark_split import resolve_split_assets, apply_signal_stats_override  # noqa: E402
from mast_utils.benchmark_eval import _reduce_mask  # noqa: E402
from mmt.utils import validate_config, sdpa_math_only_ctx  # noqa: E402
from mmt.checkpoints import load_best_weights  # noqa: E402
from mmt.data import build_decoders  # noqa: E402
from mmt.eval.forward import forward_decode_native  # noqa: E402
from tokamark.data import initialize_MAST_dataset  # noqa: E402

RUNS = ROOT / "runs" / "tokamark" / "tokamind"
OUT = ROOT / "runs" / "strat" / "tokamark"
ZARR = ROOT / "data" / "tokamark" / "zarr"
TASK = "task_1-1"
N_LOADER = int(os.environ.get("STRAT_LOADER_WORKERS", "6"))

GROUPS = {
    "scratch": [("scr_vanilla", "ft-task_1-1-scratch-vanilla_s1337_scratch", "vanilla"),
                ("scr_mlp30", "ft-task_1-1-scratch-mlp30_s1337_scratch", "mlp30"),
                ("scr_mr30ryd", "ft-task_1-1-scratch-mr30ryd_s1337_scratch", "mr30ryd")],
    "ft": [(f"ft_{v}_s{s}", f"ft-task_1-1-ws-tokamind-base-v2-{v}_s{s}", v)
           for v in ("vanilla", "mlp30", "mr30ryd") for s in (1337, 1338, 1339)]
          + [("ft_q3d_s1337", "ft-task_1-1-ws-tokamind-base-v2-q30-geo=3d_s1337", "q30-geo=3d"),
             ("ft_q3dmlp_s1337", "ft-task_1-1-ws-tokamind-base-v2-q30-geo=3d-mlp_s1337", "q30-geo=3d-mlp")],
}
GROUPS["ftvan"] = [m for m in GROUPS["ft"] if m[2] == "vanilla"]                 # the FT-Base members of E4
GROUPS["ftneck"] = [m for m in GROUPS["ft"] if m[2] in ("mlp30", "mr30ryd")]
GROUPS["ftq"] = [m for m in GROUPS["ft"] if m[2].startswith("q30")]
# §37 short continuation fine-tunes of the Scratch vanilla run (paper/strat/tokamark/ft_short.py); absolute run dirs
GROUPS["ftcont"] = [(f"ftc_{m}", str(OUT / "ft" / f"ft-task_1-1-ws-ft-task_1-1-scratch-vanilla_s1337_scratch-strat_cont_{m}"),
                     "vanilla") for m in ("smoke", "eq", "w")]
# checkpoint (weight-space) averages; the first source must be a vanilla run (its architecture hosts the soup)
GROUPS["soupscr"] = [("soup_selfcheck_scr", GROUPS["scratch"][0][1], "soup:scr_vanilla"),
                     ("soup_scr3", GROUPS["scratch"][0][1], "soup:scr_vanilla+scr_mlp30+scr_mr30ryd"),
                     ("soup_scr_van_mlp", GROUPS["scratch"][0][1], "soup:scr_vanilla+scr_mlp30")]
GROUPS["soupft"] = [("soup_ftvan3", GROUPS["ftvan"][0][1], "soup:ft_vanilla_s1337+ft_vanilla_s1338+ft_vanilla_s1339"),
                    ("soup_ft9", GROUPS["ftvan"][0][1], "soup:" + "+".join(m[0] for m in GROUPS["ftvan"] + GROUPS["ftneck"]))]


def _redirect_snapshots(tag: str) -> None:
    """Their eval config loader writes <member run>/eval/eval.yaml; write it under runs/strat/tokamark/cfg/ instead."""
    def finalize(merged, *, phase, configs_root_path, tasks_overrides_dir):
        merged["paths"] = compute_paths(merged=merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir)
        merged["eval_id"] = merged["paths"]["eval_id"]
        out_dir = OUT / "cfg" / tag / pathlib.Path(merged["model_source"]["run_dir"]).name
        merged["paths"]["run_dir"] = str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        import yaml
        (out_dir / f"{merged['eval_id']}.yaml").write_text(yaml.safe_dump(merged, sort_keys=False))
    LOADER.finalize_and_save_config = finalize


def allowed_shots(split: str) -> list[int]:
    sa = resolve_split_assets("random")
    df = pd.read_csv(sa["data_splits_file_path"], usecols=["shot_id", "train", "val"])   # test column never read
    col = "train" if split == "train" else "val"
    ok = set(int(s) for s in df.loc[df[col].astype(bool), "shot_id"])
    if split == "valid":
        shots = sorted(int(s) for s in json.loads((OUT / "strata.json").read_text())["valid_sample_shots"])
    elif split == "validsub":                   # TTA probe subset: the 30 lowest shot ids of every stratum (fixed rule)
        st = json.loads((OUT / "strata.json").read_text())["strata_shots"]
        shots = sorted(int(s) for v in st.values() for s in sorted(v)[:30])
    else:
        shots = sorted(int(s) for s in json.loads((OUT / "shot_meta.json").read_text())["train_sub"])
    bad = [s for s in shots if s not in ok]
    if bad:
        raise SystemExit(f"refusing: {len(bad)} shots not in the {split} split (e.g. {bad[:3]})")
    return shots


def soup_model(variant: str, cfg, load_member):
    """Vanilla-architecture model whose weights are the uniform average of the listed members' weights. Wrapped heads
    contribute their inner head (modality_heads.<m>.head.* -> modality_heads.<m>.*); their bottlenecks are dropped."""
    srcs = [next(m for g in GROUPS.values() for m in g if m[0] == t) for t in variant[5:].split("+")]
    if srcs[0][2] != "vanilla":
        raise ValueError(f"soup host must be a vanilla run: {variant}")
    base, _, _ = load_member(srcs[0][1], "vanilla", cfg)
    sds = []
    for tag, rd, v in srcs:
        m, _, _ = load_member(rd, v, member_cfg(RUNS / rd))
        sd = {}
        for k, t in m.state_dict().items():
            parts = k.split(".")
            if parts[0] == "modality_heads" and len(parts) > 2 and parts[2] == "bottleneck":
                continue
            if parts[0] == "modality_heads" and len(parts) > 2 and parts[2] == "head":
                k = ".".join(parts[:2] + parts[3:])
            sd[k] = t.detach().clone()
        sds.append(sd)
        del m
    ref = base.state_dict()
    avg = {k: (torch.stack([sd[k].float() for sd in sds]).mean(0).to(ref[k].dtype) if ref[k].is_floating_point()
               else sds[0][k]) for k in ref}
    base.load_state_dict(avg, strict=True)
    return base.eval()


def member_cfg(run_dir: pathlib.Path):
    cfg = load_experiment_config(task=TASK, phase="eval", model=str(run_dir))
    TH._mutate_config(cfg, seed=1337, subset=None, workers=N_LOADER)
    cfg.raw.setdefault("loader", {})["num_workers"] = N_LOADER
    cfg.raw["data"].setdefault("cache", {})["enable"] = False
    validate_config(cfg=cfg)
    return cfg


def run_pass(group: str, split: str, members_filter: str | None, n_shots: int | None, out: pathlib.Path,
             drop_input: str | None = None) -> float:
    """One data pass over a fixed shot list with every member of `group`; returns windows/s."""
    members = [m for m in GROUPS[group] if members_filter is None or m[0] in members_filter.split(",")]
    shots = allowed_shots(split)
    if n_shots is not None and n_shots < len(shots):          # evenly spaced over the sorted list (spans campaigns)
        shots = [shots[i] for i in np.linspace(0, len(shots) - 1, n_shots).round().astype(int)]
    _redirect_snapshots(f"{group}_{split}")

    cfgs = [member_cfg(RUNS / rd) for _, rd, _ in members]
    cfg0 = cfgs[0]
    if drop_input:                                  # test-time input dropout (TTA probe): their eval collate drop hook
        cfg0.raw.setdefault("eval", {}).setdefault("drop", {})["inputs"] = [drop_input]
    device, _ = init_run_context(cfg_mmt=cfg0, phase="eval")
    amp = cfg0.eval.get("amp", {}).get("enable", True)
    cfg_task = load_task_definition(task_key=TASK)
    sa = resolve_split_assets(cfg0.model_source["data_split"])
    meta = dict(get_task_metadata(config_task=cfg_task, verbose=False))
    apply_signal_stats_override(dict_task_metadata=meta, stats_metadata_file_path=sa["stats_metadata_file_path"])
    ds = initialize_MAST_dataset(config_task=cfg_task, shots_list=shots, local_flag=True, use_std_scaling=True,
                                 stats_metadata_file_path=sa["stats_metadata_file_path"], use_nan_filling=False,
                                 remove_outliers=True, outlier_metadata_file=sa["outlier_metadata_file"],
                                 store_manager_settings={"base_local_zarr_path": str(ZARR)}, verbose=False)
    stats = extract_signal_stats(dict_metadata=meta)
    by_role = build_signals_by_role_from_task_definition(cfg_task=cfg_task, dict_metadata=meta)
    specs, codecs = resolve_eval_embeddings(cfg_mmt=cfg0, signals_by_role=by_role, dict_task_metadata=meta,
                                            train_run_dir=RUNS / members[0][1])
    loader = build_window_data(cfg_mmt=cfg0, mast_datasets={"valid": ds}, dict_task_metadata=meta, cfg_task=cfg_task,
                               signal_specs=specs, codecs=codecs, phase="eval")["valid"]["loader"]
    out_specs = list(specs.specs_for_role("output"))
    id_to_name = {s.signal_id: s.name for s in out_specs}
    dec_id = build_decoders(registry=specs, codecs=codecs, role="output")
    decoders = {s.name: dec_id[s.signal_id] for s in out_specs if s.signal_id in dec_id}
    out_stats = {s.name: stats[s.name] for s in out_specs if s.name in stats}

    def load_member(rd: str, variant: str, cfg):
        m = build_model_and_optional_warmstart(cfg_mmt=cfg, signal_specs=specs, device=device, skip_warmstart=True)
        m = TH.wrap_heads(m, variant)
        ep, bv, _ = load_best_weights(run_dir=str(RUNS / rd), model=m, map_location=str(device))
        return m.eval(), ep, bv

    models = []
    for (tag, rd, variant), cfg in zip(members, cfgs):
        if variant.startswith("soup:"):
            m = soup_model(variant, cfg, load_member)
            print(f"[dump] built {tag} = uniform weight average of {variant[5:]}", flush=True)
        else:
            m, ep, bv = load_member(rd, variant, cfg)
            print(f"[dump] loaded {tag} ({variant}) epoch={ep} best_val={bv}", flush=True)
        models.append(m)

    acc: dict[str, dict[str, list]] = {n: {"sid": [], "win": [], "true": [], "pred": []} for n in out_stats}
    n_win, t0, mismatch = 0, time.time(), 0.0
    with torch.no_grad(), sdpa_math_only_ctx():
        for batch in loader:
            per_member = [forward_decode_native(batch=batch, model=m, device=device, stats=out_stats, decoders=decoders,
                                                id_to_name=id_to_name, amp_enabled=amp) for m in models]
            y_true, _, y_mask, sids, wins = per_member[0]
            n_win += len(sids)
            for name in out_stats:
                if name not in y_true or name not in y_mask:
                    continue
                mb = _reduce_mask(mask=y_mask[name])
                if not mb.any():
                    continue
                idx = np.nonzero(mb)[0]
                a = acc[name]
                a["sid"].append(np.asarray(sids)[idx].astype(np.int64))
                a["win"].append(np.asarray(wins)[idx].astype(np.int64))
                a["true"].append(np.asarray(y_true[name])[idx].reshape(len(idx), -1).astype(np.float32))
                a["pred"].append(np.stack([np.asarray(pm[1][name])[idx].reshape(len(idx), -1)
                                           for pm in per_member]).astype(np.float32))
                for pm in per_member[1:]:
                    mismatch = max(mismatch, float(np.nanmax(np.abs(np.asarray(pm[0][name])[idx].reshape(len(idx), -1)
                                                                    - a["true"][-1]), initial=0.0)))
            if n_win // 20000 != (n_win - len(sids)) // 20000:
                print(f"[dump] {n_win} windows {time.time() - t0:.0f}s peak_gpu={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                      flush=True)
    save = {"members": np.array([m[0] for m in members]), "features": np.array(list(out_stats)),
            "shots": np.array(shots), "std": np.array([out_stats[n]["std"] for n in out_stats], dtype=np.float64)}
    for name, a in acc.items():
        if a["sid"]:
            save[f"{name}|sid"] = np.concatenate(a["sid"])
            save[f"{name}|win"] = np.concatenate(a["win"])
            save[f"{name}|true"] = np.concatenate(a["true"])
            save[f"{name}|pred"] = np.concatenate(a["pred"], axis=1)
    np.savez_compressed(out, **save)
    rate = n_win / max(time.time() - t0, 1e-6)
    print(f"[dump] done {group}/{split}: {len(shots)} shots, {n_win} windows in {time.time() - t0:.0f}s ({rate:.0f}/s), "
          f"truth mismatch across members {mismatch:.3g}, peak_gpu={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB -> {out}",
          flush=True)
    del models
    torch.cuda.empty_cache()
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default="scratch:valid,ft:valid,scratch:train,ft:train",
                    help="comma list of group:split; a pass whose output exists is skipped")
    ap.add_argument("--members", default=None, help="comma list of member tags (default: whole group)")
    ap.add_argument("--max-shots", type=int, default=None, help="smoke: at most N shots per pass")
    ap.add_argument("--train-shots", type=int, default=200, help="TRAIN subsample size (evenly spaced)")
    ap.add_argument("--budget", type=float, default=1650.0, help="do not start a pass that would overrun (s)")
    ap.add_argument("--out", default=None, help="single-pass output override (smoke)")
    ap.add_argument("--drop-inputs", default=None, help="comma list: one extra pass per dropped input signal (TTA)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    t_start, rate = time.time(), None
    for ps in args.passes.split(","):
        group, split = ps.split(":")
        out = pathlib.Path(args.out) if args.out else OUT / f"preds_{group}_{split}.npz"
        if out.exists() and not args.out:
            print(f"[dump] skip {ps}: {out} exists", flush=True)
            continue
        n = args.max_shots if split == "valid" else min(args.train_shots, args.max_shots or 10**9)
        est_windows = 280 * (n or len(allowed_shots(split)))
        if rate is not None and time.time() - t_start + 1.3 * est_windows / rate > args.budget:
            print(f"[dump] stop before {ps}: est {est_windows / rate:.0f}s would exceed the budget", flush=True)
            break
        if args.drop_inputs:
            for di in args.drop_inputs.split(","):
                o = OUT / f"preds_{group}_{split}_drop-{di}.npz"
                if not o.exists():
                    rate = run_pass(group, split, args.members, n, o, drop_input=di)
            continue
        rate = run_pass(group, split, args.members, n, out)
    print(f"[dump] all passes finished/stopped after {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
