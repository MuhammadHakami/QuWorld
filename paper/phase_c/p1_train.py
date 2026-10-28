"""Phase-C P1: backbone-free adapter training on the teacher-forced chunk cache.

Data: data/phase_c/teacher/<traj>.pt from teacher_cache.py — per trajectory
    x0_pred (8, 16, 3, 60, 104) fp16   frozen backbone's chunk prediction
    gt      (8, 16, 3, 60, 104) fp16   ground-truth VAE chunk latent
Train = train_* files, valid = valid_* files (selection AND reporting set —
there is no third split in this subset; stated in p1_results.json).

Adapter (phase_c/README.md "Adapter-training handoff", mirrors the Phase-B
Bottleneck design in paper/model.py:52-79 — pool -> core -> zero-init gate):
    inp   = cat(x0_pred[j], prev)          prev = adapted chunk j-1 (GT chunk
                                            j-1 at a window start, zeros at j=0)
    feat  = conv stem (shared design, trained per arm)
    h     = attention pool over the 4680 feature tokens -> (B, D_H)
    code  = CORE(h) -> (B, 30)             <- the only part that differs by arm
    r     = FiLM(code) on feat -> conv decoder -> (16, 3, 60, 104)
    out   = x0_pred[j] + alpha * r         alpha zero-init per channel
Arms (parameter-matched AT THE CORE, exactly as cnext_hybrid.py:55-63):
    mr30ryd : paper/qlayer.py MultiRegister(D_H, 10, 3, 4, "rydberg") — 30
              simulated qubits (3 x 10q Rydberg-analog registers), fp32
    mlp30   : paper/model.py MLPCore(D_H, 30, n_params(mr30ryd core))
    lora    : LoRACore — rank-r factorised linear map D_H -> r -> 30 with r
              chosen so its parameter count matches the quantum core
Objective: TD/pushforward k=4 (finetune_td.py:75-85 generalised to every
step): windows of 4 consecutive chunks; the adapter's previous output feeds
the next step DETACHED (own-error exposure, one-step memory), loss = mean
over the 4 steps of MSE(out, gt).
Selection metric: chunk-latent rollout VRMSE on the valid trajectories —
per chunk sqrt(MSE/Var) per latent channel (eval_official.py:55-61
convention), mean over channels, steps 1..4 and windows. Identical LR grid
per arm; per (arm, seed) the best LR by that metric is kept.
Outputs: runs/phase_c/p1/<arm>_s<seed>_lr<lr>.{json,pt} (resumable units),
runs/phase_c/p1_<arm>_s<seed>.pt (best over the grid), runs/phase_c/
p1_results.json (per-arm mean+-SD over seeds, seed-paired comparisons,
per-trajectory paired Wilcoxon on seed-averaged VRMSE, vanilla baseline).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "paper"))
from model import MLPCore, _count  # noqa: E402  (paper/model.py)
from qlayer import MultiRegister  # noqa: E402  (paper/qlayer.py)

TEACHER_DIR = ROOT / "data" / "phase_c" / "teacher"
RUNS_DIR = ROOT / "runs" / "phase_c"
P1_DIR = RUNS_DIR / "p1"
STATUS_FILE = RUNS_DIR / "phase_c_status"

ARMS = ("mr30ryd", "lora", "mlp30")
SEEDS = (1337, 1338, 1339)
LR_GRID = (3e-4, 1e-3, 3e-3)
K_STEPS = 4
CORE_OUT = 30                 # 30 simulated qubits -> 30 readouts
N_QUBITS, N_REGISTERS, Q_LAYERS = 10, 3, 4
LAT_C, CHUNK, LAT_H, LAT_W = 16, 3, 60, 104
FEAT_C, N_QUERIES = 32, 2     # stem width, attention-pool queries -> D_H = 64
EPS = 1e-7


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def status(path: pathlib.Path, msg: str) -> None:
    line = f"{now()} p1 {msg}"
    print(line, flush=True)
    with path.open("a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------
# cores (parameter-matched at the core, cnext_hybrid.py:55-63 convention)
# --------------------------------------------------------------------------

class LoRACore(nn.Module):
    """Rank-r factorised linear map D_H -> r -> CORE_OUT, r set by budget."""

    def __init__(self, d_in: int, out_n: int, target_params: int):
        super().__init__()
        r = max(1, (target_params - out_n) // (d_in + out_n))
        self.down = nn.Linear(d_in, r, bias=False)
        self.up = nn.Linear(r, out_n)
        self.rank = r

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(h))


def build_core(arm: str, d_in: int) -> nn.Module:
    ref = MultiRegister(d_in, N_QUBITS, N_REGISTERS, Q_LAYERS, "rydberg")
    if arm == "mr30ryd":
        return ref
    if arm == "mlp30":
        return MLPCore(d_in, CORE_OUT, _count(ref))
    if arm == "lora":
        return LoRACore(d_in, CORE_OUT, _count(ref))
    raise ValueError(f"unknown arm {arm}")


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """N_QUERIES learned queries attend over feature tokens -> (B, Q*C)."""

    def __init__(self, ch: int, n_queries: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, ch) / math.sqrt(ch))
        self.key = nn.Linear(ch, ch)
        self.scale = 1.0 / math.sqrt(ch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:   # (B, N, C)
        att = torch.einsum("qc,bnc->bqn", self.q, self.key(tokens)) * self.scale
        att = att.softmax(dim=-1)
        return torch.einsum("bqn,bnc->bqc", att, tokens).flatten(1)


class ChunkAdapter(nn.Module):
    """x0_pred (B,16,3,60,104) [+ prev chunk] -> corrected chunk; zero-init gate."""

    def __init__(self, arm: str):
        super().__init__()
        c = FEAT_C
        self.stem = nn.Sequential(
            nn.Conv3d(2 * LAT_C, c, 3, padding=1), nn.SiLU(),
            nn.Conv3d(c, c, 3, stride=(1, 2, 2), padding=1), nn.SiLU())
        self.pool = AttentionPool(c, N_QUERIES)
        self.d_h = c * N_QUERIES
        self.core = build_core(arm, self.d_h)
        self.film = nn.Linear(CORE_OUT, 2 * c)
        self.dec = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.Conv3d(c, LAT_C, 3, padding=1))
        self.alpha = nn.Parameter(torch.zeros(LAT_C))             # zero-init gate
        self.arm = arm

    def forward(self, x_pred: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        feat = self.stem(torch.cat([x_pred, prev], dim=1))        # (B,C,3,30,52)
        h = self.pool(feat.flatten(2).transpose(1, 2))            # (B, D_H)
        code = self.core(h.float())                               # (B, 30)
        scale, shift = self.film(code).chunk(2, dim=-1)
        feat = feat * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        r = self.dec(feat)                                        # (B,16,3,60,104)
        return x_pred + self.alpha[None, :, None, None, None] * r


def param_report(model: ChunkAdapter) -> dict:
    core = _count(model.core)
    extra = {"rank": model.core.rank} if hasattr(model.core, "rank") else {}
    return {"total": _count(model), "core": core, "shared": _count(model) - core,
            "d_h": model.d_h, **extra}


# --------------------------------------------------------------------------
# data + metrics
# --------------------------------------------------------------------------

def load_split(data_dir: pathlib.Path, prefix: str, device: str):
    files = sorted(data_dir.glob(f"{prefix}_*.pt"))
    if not files:
        raise RuntimeError(f"no {prefix}_* files in {data_dir}")
    xs, gs, names = [], [], []
    for p in files:                                   # per-file loads, no mmap
        d = torch.load(p, weights_only=True)
        xs.append(d["x0_pred"].float()); gs.append(d["gt"].float()); names.append(p.stem)
    x, g = torch.stack(xs).to(device), torch.stack(gs).to(device)  # (N,8,16,3,60,104)
    if x.shape[1] < K_STEPS + 1:
        raise RuntimeError("need at least K+1 chunks per trajectory")
    return x, g, names


def window_starts(n_chunks: int) -> list[int]:
    return list(range(0, n_chunks - K_STEPS + 1))


def vrmse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """(B,16,3,60,104) -> (B,) per-channel sqrt(MSE/Var), mean over channels."""
    p, t = pred.flatten(2), true.flatten(2)
    mse = (p - t).square().mean(dim=2)
    var = (t - t.mean(dim=2, keepdim=True)).square().mean(dim=2)
    return (mse / (var + EPS)).sqrt().mean(dim=1)


def prev_chunk(g: torch.Tensor, traj: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """GT chunk start-1 (zeros for start == 0) — teacher-forced window start."""
    prev = torch.zeros_like(g[traj, 0])
    has = start > 0
    prev[has] = g[traj[has], (start[has] - 1)]
    return prev


def rollout(model: ChunkAdapter, x: torch.Tensor, g: torch.Tensor, traj: torch.Tensor,
            start: torch.Tensor, detach: bool) -> list[torch.Tensor]:
    """K adapter calls chained through the (optionally detached) own output."""
    prev, outs = prev_chunk(g, traj, start), []
    for i in range(K_STEPS):
        out = model(x[traj, start + i], prev)
        outs.append(out)
        prev = out.detach() if detach else out
    return outs


@torch.no_grad()
def evaluate(model: ChunkAdapter, x: torch.Tensor, g: torch.Tensor) -> dict:
    """Rollout VRMSE (mean over steps 1..K) + one-step VRMSE, per trajectory."""
    model.eval()
    n, n_chunks = x.shape[0], x.shape[1]
    starts = window_starts(n_chunks)
    traj = torch.arange(n, device=x.device).repeat_interleave(len(starts))
    start = torch.tensor(starts, device=x.device).repeat(n)
    outs = rollout(model, x, g, traj, start, detach=True)
    per_step = torch.stack([vrmse(o, g[traj, start + i]) for i, o in enumerate(outs)])
    roll = per_step.mean(dim=0).view(n, len(starts)).mean(dim=1)      # (n,)
    # one-step: every chunk with its GT predecessor as prev
    traj1 = torch.arange(n, device=x.device).repeat_interleave(n_chunks)
    idx1 = torch.arange(n_chunks, device=x.device).repeat(n)
    one = vrmse(model(x[traj1, idx1], prev_chunk(g, traj1, idx1)), g[traj1, idx1])
    model.train()
    return {"rollout": roll.mean().item(), "rollout_per_traj": roll.tolist(),
            "rollout_step_k": per_step[-1].mean().item(),
            "onestep": one.mean().item(),
            "onestep_per_traj": one.view(n, n_chunks).mean(dim=1).tolist()}


def vanilla_metrics(x: torch.Tensor, g: torch.Tensor) -> dict:
    """Frozen backbone alone (gate = 0): rollout == one-step by construction."""
    n, n_chunks = x.shape[0], x.shape[1]
    v = vrmse(x.flatten(0, 1), g.flatten(0, 1)).view(n, n_chunks).mean(dim=1)
    return {"rollout": v.mean().item(), "onestep": v.mean().item(),
            "rollout_per_traj": v.tolist(), "onestep_per_traj": v.tolist()}


# --------------------------------------------------------------------------
# one training unit (arm, seed, lr) — resumable by its result file
# --------------------------------------------------------------------------

def train_unit(arm: str, seed: int, lr: float, data: dict, args) -> dict:
    tag = f"{arm}_s{seed}_lr{lr:g}"
    res_path, ckpt_path = P1_DIR / f"{tag}.json", P1_DIR / f"{tag}.pt"
    if res_path.exists() and not args.overwrite:
        return json.loads(res_path.read_text())
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ChunkAdapter(arm).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    xt, gt, xv, gv = data["xt"], data["gt"], data["xv"], data["gv"]
    n, n_chunks = xt.shape[0], xt.shape[1]
    starts = window_starts(n_chunks)
    best, best_state, best_step, history = math.inf, None, 0, []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        traj = torch.randint(0, n, (args.batch,), generator=gen).to(args.device)
        start = torch.tensor([starts[i] for i in torch.randint(
            0, len(starts), (args.batch,), generator=gen)], device=args.device)
        outs = rollout(model, xt, gt, traj, start, detach=True)
        loss = torch.stack([F.mse_loss(o, gt[traj, start + i])
                            for i, o in enumerate(outs)]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, xv, gv)
            history.append({"step": step, "loss": loss.item(), **{k: ev[k] for k in
                            ("rollout", "rollout_step_k", "onestep")}})
            if ev["rollout"] < best:
                best, best_step = ev["rollout"], step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_eval = ev
        if time.time() - t0 > args.unit_budget_s:
            break
    secs = time.time() - t0
    if best_state is None:                            # budget hit before any eval
        best_eval, best_step = evaluate(model, xv, gv), step
        best = best_eval["rollout"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_eval = evaluate(model, xt, gt)
    result = {"arm": arm, "seed": seed, "lr": lr, "steps_run": step, "best_step": best_step,
              "train_seconds": secs, "ms_per_step": 1e3 * secs / step,
              "params": param_report(model),
              "valid": best_eval, "train_at_best": {k: train_eval[k] for k in ("rollout", "onestep")},
              "history": history}
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save({"model_state_dict": best_state, "arm": arm, "seed": seed, "lr": lr,
                "step": best_step, "valid": best_eval, "params": result["params"],
                "k_steps": K_STEPS, "core_out": CORE_OUT}, tmp)
    tmp.replace(ckpt_path)
    res_path.write_text(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def mean_sd(v: list[float]) -> dict:
    return {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "n": len(v)}


def paired(a: list[float], b: list[float]) -> dict:
    """Seed-paired a - b (negative = a better); t-test one-sided a < b."""
    from scipy.stats import ttest_rel
    d = [x - y for x, y in zip(a, b)]
    t = ttest_rel(a, b, alternative="less")
    return {"diff": mean_sd(d), "wins": sum(x < 0 for x in d),
            "t_p_one_sided": float(t.pvalue)}


def wilcoxon_per_traj(a: list[list[float]], b: list[list[float]]) -> dict:
    """Per-trajectory paired Wilcoxon on seed-averaged VRMSE (paired_stats.py)."""
    from scipy.stats import wilcoxon
    ma = [statistics.fmean(col) for col in zip(*a)]
    mb = [statistics.fmean(col) for col in zip(*b)]
    d = [x - y for x, y in zip(ma, mb)]
    try:
        p2, p1 = wilcoxon(ma, mb).pvalue, wilcoxon(ma, mb, alternative="less").pvalue
    except ValueError as exc:                          # all-zero differences
        p2 = p1 = float("nan"); print(f"wilcoxon skipped: {exc}")
    return {"n_traj": len(d), "mean_diff": statistics.fmean(d),
            "win_rate": sum(x < 0 for x in d) / len(d),
            "wilcoxon_two_sided_p": float(p2), "wilcoxon_one_sided_p": float(p1)}


def aggregate(results: dict[str, dict[int, dict]], vanilla: dict, names_v: list[str],
              args) -> dict:
    per_arm = {}
    for arm, by_seed in results.items():
        per_arm[arm] = {
            "per_seed": {str(s): {"lr": r["lr"], "valid_rollout": r["valid"]["rollout"],
                                  "valid_onestep": r["valid"]["onestep"],
                                  "valid_rollout_step_k": r["valid"]["rollout_step_k"],
                                  "train_rollout": r["train_at_best"]["rollout"],
                                  "best_step": r["best_step"], "ms_per_step": r["ms_per_step"]}
                         for s, r in by_seed.items()},
            "valid_rollout": mean_sd([r["valid"]["rollout"] for r in by_seed.values()]),
            "valid_onestep": mean_sd([r["valid"]["onestep"] for r in by_seed.values()]),
            "params": next(iter(by_seed.values()))["params"],
        }
    seeds, comparisons = list(args.seeds), {}
    q = results.get("mr30ryd", {})
    for other in ("lora", "mlp30"):
        o = results.get(other, {})
        if not q or not o:
            continue
        comparisons[f"mr30ryd_vs_{other}"] = {
            "seed_paired_rollout": paired([q[s]["valid"]["rollout"] for s in seeds],
                                          [o[s]["valid"]["rollout"] for s in seeds]),
            "seed_paired_onestep": paired([q[s]["valid"]["onestep"] for s in seeds],
                                          [o[s]["valid"]["onestep"] for s in seeds]),
            "per_traj_rollout": wilcoxon_per_traj(
                [q[s]["valid"]["rollout_per_traj"] for s in seeds],
                [o[s]["valid"]["rollout_per_traj"] for s in seeds]),
        }
    controls = [a for a in ("lora", "mlp30") if a in per_arm]
    h2a = bool(q) and bool(controls) and all(
        per_arm["mr30ryd"]["valid_rollout"]["mean"] < per_arm[a]["valid_rollout"]["mean"]
        for a in controls)
    return {"timestamp": now(), "protocol": {
                "objective": f"TD/pushforward k={K_STEPS}, detached chain, mean MSE over steps",
                "selection": "valid chunk-latent rollout VRMSE (mean over steps 1..k), "
                             "best LR per (arm, seed); valid is also the reporting set",
                "lr_grid": list(LR_GRID), "seeds": list(SEEDS), "steps": args.steps,
                "batch": args.batch, "eval_every": args.eval_every,
                "unit_budget_s": args.unit_budget_s, "data_dir": str(args.data),
                "valid_trajectories": names_v},
            "vanilla_backbone": {k: vanilla[k] for k in ("rollout", "onestep")},
            "arms": per_arm, "comparisons": comparisons,
            "h2a_quantum_below_both_controls_mean": h2a}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, default=TEACHER_DIR)
    ap.add_argument("--out", type=pathlib.Path, default=RUNS_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--unit-budget-s", type=float, default=600.0,
                    help="wall cap per (arm, seed, lr) unit")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--lrs", type=float, nargs="+", default=list(LR_GRID))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--status-file", type=pathlib.Path, default=STATUS_FILE)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    global P1_DIR
    P1_DIR = args.out / "p1"
    P1_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    xt, gt, names_t = load_split(args.data, "train", args.device)
    xv, gv, names_v = load_split(args.data, "valid", args.device)
    data = {"xt": xt, "gt": gt, "xv": xv, "gv": gv}
    vanilla = vanilla_metrics(xv, gv)
    status(args.status_file, f"start: train {len(names_t)} traj x {xt.shape[1]} chunks, "
                             f"valid {len(names_v)}; vanilla valid VRMSE "
                             f"{vanilla['onestep']:.4f}; arms {args.arms} seeds {args.seeds} "
                             f"lrs {args.lrs} steps {args.steps}")
    results: dict[str, dict[int, dict]] = {}
    t_all = time.time()
    for arm in args.arms:
        results[arm] = {}
        for seed in args.seeds:
            units = []
            for lr in args.lrs:
                r = train_unit(arm, seed, lr, data, args)
                units.append(r)
                status(args.status_file,
                       f"unit {arm} s{seed} lr={lr:g}: valid rollout {r['valid']['rollout']:.4f} "
                       f"onestep {r['valid']['onestep']:.4f} @step {r['best_step']} | "
                       f"{r['ms_per_step']:.0f} ms/step, {r['train_seconds']:.0f}s | "
                       f"core params {r['params']['core']}")
            best = min(units, key=lambda r: r["valid"]["rollout"])
            results[arm][seed] = best
            src = P1_DIR / f"{arm}_s{seed}_lr{best['lr']:g}.pt"
            dst = args.out / f"p1_{arm}_s{seed}.pt"
            dst.write_bytes(src.read_bytes())
            status(args.status_file, f"best {arm} s{seed}: lr={best['lr']:g} valid rollout "
                                     f"{best['valid']['rollout']:.4f} -> {dst.name}")
    summary = aggregate(results, vanilla, names_v, args)
    (args.out / "p1_results.json").write_text(json.dumps(summary, indent=1))
    arms_line = " | ".join(f"{a} {summary['arms'][a]['valid_rollout']['mean']:.4f}"
                           f"+-{summary['arms'][a]['valid_rollout']['sd']:.4f}"
                           for a in args.arms)
    status(args.status_file, f"done in {(time.time() - t_all) / 60:.1f} min: valid rollout "
                             f"VRMSE vanilla {vanilla['rollout']:.4f} | {arms_line} | "
                             f"H2a(mean) {summary['h2a_quantum_below_both_controls_mean']} "
                             f"-> p1_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
