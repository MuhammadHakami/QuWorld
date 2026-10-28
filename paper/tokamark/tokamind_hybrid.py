"""TokaMind (TokaMark SoTA, arXiv 2602.15084) as the classical trunk + our register on its CLS latent.

SoTA-first rule: their pipeline, protocol and metric are untouched. We import their entry points
(baseline/tokamind/scripts_mast/run_finetune.py, run_eval.py), patch two functions in those namespaces and call
their main():
  * load_experiment_config -> after their merge, mutate cfg.raw in memory: seed, local zarr store, cache workers,
    optional subset_of_shots (smoke). Their config snapshot is written by them; ours is recorded in the results JSON.
  * build_model_and_optional_warmstart -> after their warm-start from UKAEA-IBM-STFC/tokamind-base-v2, wrap every
    modality head as head(Bottleneck(h_cls)) (zero-init gate => identical to TokaMind at init). The register lives in
    `modality_heads`, so it is trainable in both of their stages (ft_heads 5 ep + ft_full 15 ep), saved by
    get_modality_heads_state_dict and loaded strictly by load_best_weights at eval (same wrap before loading).
Variants: vanilla | mlp30 | mr30ryd (cores parameter-matched via _count, as in every other benchmark).

Usage (repo root, .venv-tokamark):
  python paper/tokamark/tokamind_hybrid.py finetune --task task_1-1 --variant mr30ryd --seed 1337 [--subset 8] [--workers 8]
  python paper/tokamark/tokamind_hybrid.py eval     --task task_1-1 --variant mr30ryd --seed 1337 [--subset 8]
Outputs: their run dir under runs/tokamark/tokamind/<run_id>/ (symlinked REPO_ROOT/runs) and
         runs/tokamark/results/tokamind_<task>_<variant>_s<seed>[_sub<N>].json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKAMIND = ROOT / "baseline" / "tokamind"
SCRIPTS = TOKAMIND / "scripts_mast"
MODEL_PATH = ROOT / "data" / "tokamark" / "tokamind-base-v2"
ZARR = ROOT / "data" / "tokamark" / "zarr"
RESULTS = ROOT / "runs" / "tokamark" / "results"
for p in (str(SCRIPTS), str(ROOT / "paper")):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Bottleneck, MLPCore, _count  # noqa: E402
from qlayer import MultiRegister  # noqa: E402

Q_LAYERS = 4
CORE_OUT = 30


class HeadWithBottleneck(nn.Module):
    """modality head applied to the gated CLS latent: head(h + alpha * up(core(h)))."""

    def __init__(self, head: nn.Module, bottleneck: Bottleneck):
        super().__init__()
        self.bottleneck = bottleneck
        self.head = head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottleneck(h))


def build_core(variant: str, ch: int) -> nn.Module:
    if variant == "mlp30":
        return MLPCore(ch, CORE_OUT, _count(MultiRegister(ch, 10, 3, Q_LAYERS, "hea")))
    if variant == "mr30ryd":
        return MultiRegister(ch, 10, 3, Q_LAYERS, "rydberg")
    if variant.startswith("q"):                                  # wave-G grammar, incl. doc 35 §23 q30-geo=3d[-mlp]
        from model import build_config_core, parse_config
        return build_config_core(parse_config(variant), ch, 4)[0]
    raise ValueError(f"unknown variant {variant}")


def wrap_heads(model: nn.Module, variant: str) -> nn.Module:
    """Wrap each modality head with its own zero-init gated bottleneck on h_cls (d_model)."""
    if variant == "vanilla":
        return model
    dev = next(model.parameters()).device
    d_model = model.backbone.encoder.layers[0].self_attn.embed_dim
    for mod in list(model.modality_heads.keys()):
        core = build_core(variant, d_model)
        model.modality_heads[mod] = HeadWithBottleneck(model.modality_heads[mod], Bottleneck(d_model, core, CORE_OUT))
    return model.to(dev)


def _mutate_config(cfg, *, seed: int, subset: int | None, workers: int) -> None:
    raw = cfg.raw
    raw["seed"] = int(seed)
    data = raw.setdefault("data", {})
    data["local"] = True
    data["local_path"] = str(ZARR)
    if subset is not None:
        data["subset_of_shots"] = int(subset)
    cache = data.setdefault("cache", {})
    if isinstance(cache, dict):
        cache["num_workers"] = int(workers)


def _patch(module, *, variant: str, seed: int, subset: int | None, workers: int, state: dict) -> None:
    orig_load, orig_build = module.load_experiment_config, module.build_model_and_optional_warmstart

    def load_patched(*a, **k):
        cfg = orig_load(*a, **k)
        _mutate_config(cfg, seed=seed, subset=subset, workers=workers)
        state["run_dir"] = cfg.raw.get("paths", {}).get("run_dir")
        state["model_source"] = cfg.raw.get("model_source")
        return cfg

    def build_patched(*a, **k):
        model = orig_build(*a, **k)
        model = wrap_heads(model, variant)
        state["n_params"] = _count(model)
        state["n_bottleneck_params"] = sum(
            _count(m.bottleneck) for m in model.modality_heads.values() if isinstance(m, HeadWithBottleneck))
        return model

    module.load_experiment_config = load_patched
    module.build_model_and_optional_warmstart = build_patched


def _tag(args) -> str:
    return (f"{args.variant}_s{args.seed}" + ("_scratch" if getattr(args, "init", "warmstart") == "scratch" else "")
            + (f"_sub{args.subset}" if args.subset else ""))


def _result_path(args) -> pathlib.Path:
    return RESULTS / f"tokamind_{args.task}_{_tag(args)}.json"


def cmd_finetune(args) -> None:
    import run_finetune  # noqa: E402  (their entry point; sys.path has scripts_mast)
    state: dict = {}
    _patch(run_finetune, variant=args.variant, seed=args.seed, subset=args.subset, workers=args.workers, state=state)
    sys.argv = (["run_finetune.py", "--task", args.task, "--init", args.init]
                + (["--model", str(MODEL_PATH)] if args.init == "warmstart" else [])      # scratch = TokaMind-Scratch
                + ["--tag", _tag(args), "--emb_profile", args.emb_profile])
    t0 = time.time()
    run_finetune.main()
    meta = {}
    best = pathlib.Path(state["run_dir"]) / "checkpoints" / "best" / "meta.json"
    if best.exists():
        meta = json.loads(best.read_text())
    out = {"stage": "finetune", "task": args.task, "variant": args.variant, "seed": args.seed, "subset": args.subset,
           "run_dir": state["run_dir"], "trunk": str(MODEL_PATH) if args.init == "warmstart" else "scratch", "n_params": state.get("n_params"),
           "n_bottleneck_params": state.get("n_bottleneck_params"), "best_meta": meta,
           "wall_s": round(time.time() - t0, 1), "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    _result_path(args).write_text(json.dumps(out, indent=1))
    print("[tokamind-hybrid] finetune done:", json.dumps({k: out[k] for k in ("variant", "seed", "run_dir", "wall_s")}))


def _hook_predictions(members: list[str] | None) -> dict:
    """Record the decoded native-unit predictions per (shot, window) (TOKAMIND_DUMP), or replace them by the mean of
    recorded members' predictions on the same windows (TOKAMIND_ENSEMBLE), so TokaMind's own metric code scores the
    ensemble. Ground truth, masks and window selection always come from the unmodified evaluation pass."""
    import numpy as np
    import mast_utils.benchmark_eval as BE  # noqa: E402  (evaluate_benchmark_and_diagnostics looks this name up)
    orig, rec = BE.forward_decode_native, {}
    bank = [np.load(m, allow_pickle=True)["rec"].item() for m in members] if members else None
    wfile = __import__("os").environ.get("TOKAMIND_ENSEMBLE_WEIGHTS")  # doc 35 §37a: {feature: [w per member]}
    wts = json.loads(pathlib.Path(wfile).read_text()) if (wfile and bank is not None) else None
    truth = bool(__import__("os").environ.get("TOKAMIND_DUMP_TRUTH"))

    def hooked(**kw):
        y_true, y_pred, y_mask, shot_ids, win = orig(**kw)
        keys = [f"{s}|{int(w)}" for s, w in zip(shot_ids, win)]
        for name in list(y_pred):
            a = np.asarray(y_pred[name])
            t = np.asarray(y_true[name]) if truth else None
            for i, k in enumerate(keys):
                rec.setdefault(k, {})[name] = a[i].copy()
                if truth:                             # diagnostics only (TOKAMIND_DUMP_TRUTH): never fitted on
                    rec[k][name + "#true"] = t[i].copy()
            if bank is not None and wts is not None:          # TRAIN-fitted per-feature weights (sum to 1)
                w = np.asarray(wts[name], dtype=np.float64)
                assert len(w) == len(bank) and abs(w.sum() - 1) < 1e-9, (name, w)
                ens = np.stack([sum(wi * np.asarray(b[k][name], np.float64) for wi, b in zip(w, bank))
                                for k in keys]).astype(a.dtype)
                y_pred[name] = torch.as_tensor(ens) if torch.is_tensor(y_pred[name]) else ens
            elif bank is not None:
                ens = np.stack([np.mean([b[k][name] for b in bank], axis=0) for k in keys]).astype(a.dtype)
                y_pred[name] = torch.as_tensor(ens) if torch.is_tensor(y_pred[name]) else ens
        return y_true, y_pred, y_mask, shot_ids, win

    BE.forward_decode_native = hooked
    return rec


def cmd_eval(args) -> None:
    import os
    import shutil
    import numpy as np
    import run_eval  # noqa: E402
    res_path = _result_path(args)
    prev = json.loads(res_path.read_text())
    run_dir = prev["run_dir"]
    dump, members = os.environ.get("TOKAMIND_DUMP"), os.environ.get("TOKAMIND_ENSEMBLE")
    if members:                                   # the ensemble pass reuses a member's run_dir: protect its eval outputs
        backup = pathlib.Path(run_dir) / "eval.member_backup"
        if (pathlib.Path(run_dir) / "eval").exists() and not backup.exists():
            shutil.copytree(pathlib.Path(run_dir) / "eval", backup)
    rec = _hook_predictions(members.split(",") if members else None) if (dump or members) else None
    state: dict = {}
    _patch(run_eval, variant=args.variant, seed=args.seed, subset=args.subset, workers=args.workers, state=state)
    sys.argv = ["run_eval.py", "--task", args.task, "--model", run_dir]
    t0 = time.time()
    run_eval.main()
    if dump:
        np.savez_compressed(dump, rec=np.array(rec, dtype=object))
        print(f"[tokamind-hybrid] recorded {len(rec)} windows -> {dump}", flush=True)
    metrics = {}
    for csv in sorted(glob.glob(str(pathlib.Path(run_dir) / "eval" / "**" / "task_metrics.csv"), recursive=True)):
        import pandas as pd  # noqa: E402  (their eval already depends on pandas)
        df = pd.read_csv(csv, index_col=0)
        metrics[csv] = {str(i): {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}
                        for i, row in df.iterrows()}
    prev.update({"stage": "eval", "eval_run_dir": state.get("run_dir"), "task_metrics": metrics,
                 "eval_wall_s": round(time.time() - t0, 1)})
    if members:                                   # ensemble metrics go to their own file; the member's eval dir is restored
        out = pathlib.Path(os.environ["TOKAMIND_OUT"])
        out.write_text(json.dumps(prev | {"stage": "ensemble_eval", "members": members.split(",")}, indent=1))
        keep = out.parent / f"{out.stem}_eval"          # the ensemble's own per-shot / per-window CSVs (paired stats)
        shutil.rmtree(keep, ignore_errors=True); shutil.move(str(pathlib.Path(run_dir) / "eval"), str(keep))
        shutil.move(str(backup), str(pathlib.Path(run_dir) / "eval"))
        print("[tokamind-hybrid] ensemble eval done:", json.dumps(metrics)[:600])
        return
    if dump:                                      # recording pass: the member's stored result is left untouched
        return
    res_path.write_text(json.dumps(prev, indent=1))
    print("[tokamind-hybrid] eval done:", json.dumps(metrics)[:600])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("finetune", "eval"))
    ap.add_argument("--task", default="task_1-1")
    ap.add_argument("--variant", default="vanilla", help="vanilla | mlp30 | mr30ryd | wave-G grammar (q30-geo=3d[-mlp], doc 35 §23)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--subset", type=int, default=None, help="data.subset_of_shots (smoke); None = all shots")
    ap.add_argument("--workers", type=int, default=8, help="data.cache.num_workers (their default 64; RAM)")
    ap.add_argument("--emb_profile", default="dct3d")
    ap.add_argument("--init", default="warmstart", choices=("warmstart", "scratch"),
                    help="scratch = their TokaMind-Scratch recipe (finetune_scratch.yaml; paper best 0.140 on task 1-1)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU-only lane: CUDA not available")
    (cmd_finetune if args.stage == "finetune" else cmd_eval)(args)


if __name__ == "__main__":
    main()
