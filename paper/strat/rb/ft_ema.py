"""§37 rayleigh_benard, strategy 3b/3c: a short joint fine-tune of the best single gated model with weight averaging.

Start: runs/alloc/ft/rayleigh_benard_mr30ryd_vrmse_gate_s1_pre.pt (trunk + exactly pre-trained gate; VALID-best single).
Recipe = the doc 37 joint stage (VRMSE loss on the gated output, AdamW wd 0.01, trunk lr 2e-5, gate lr 3e-4, effective
batch 4, grad clip 1.0) with two changes fixed before any score:
  1. the gate's BatchNorm stays in eval mode (the doc 37 joint stage normalised the gate input with batch-of-4 statistics
     in training but cumulative running statistics at evaluation: a train/eval mismatch);
  2. weight averaging: EMA (decay 0.998, every optimizer step) and SWA (uniform mean of iterates after 25% of the
     budget). The averaged weights are single models (one forward pass).
Optional TRAIN sampler `--onset q`: a fraction q of TRAIN windows is drawn from the onset (t0 <= 9), the rest uniform.
No VALID-based selection inside the run (TRAIN fits); the fixed stratified VALID sample scores start / final / EMA / SWA
once at the end, per window and per field, into runs/strat/rb/ft_<tag>.npz. Weights: runs/strat/rb/ft_<tag>_{ema,swa}.pt.

    flock runs/strat/gpu.lock timeout 1800 env QW_GPU_GB=12 runs/capped.sh strat_rb 16 uv run --no-sync python paper/strat/rb/ft_ema.py --budget 1020 [--onset 0.25]
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

import common as CM


def sample_batch(train, bsz, gen, onset_q):
    ni = torch.randint(0, train.n, (bsz,), generator=gen)
    ti = torch.randint(CM.well2d.N_HIST - 1, train.T - 1, (bsz,), generator=gen)
    if onset_q > 0:
        on = torch.rand(bsz, generator=gen) < onset_q
        ti = torch.where(on, torch.randint(CM.well2d.N_HIST - 1, 10, (bsz,), generator=gen), ti)
    xs, ys = zip(*(train.window(int(a), int(b)) for a, b in zip(ni, ti)))
    return torch.stack(xs).to(CM.DEV), torch.stack(ys).to(CM.DEV)


@torch.no_grad()
def score(model, valid, pairs, bsz=8):
    model.eval(); out = []
    for x, y in CM.well2d.batched(valid, pairs, bsz):
        out.append(CM.vrmse_fields(model(x.to(CM.DEV)), y.to(CM.DEV)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1020); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gate-lr", type=float, default=3e-4); ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2); ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--onset", type=float, default=0.0); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--member", default=CM.BEST_SINGLE); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = f"{a.member}_lr{a.lr:g}_on{a.onset:g}_s{a.seed}" + ("_smoke" if a.smoke else "")
    doc, pairs, sid = CM.load_strata()
    if a.smoke:
        pairs = pairs[:16]
    train, valid = CM.train_split(), CM.valid_split()
    model = CM.load_gated(a.member, valid)
    start_state = copy.deepcopy(model.state_dict())
    model.eval()                                            # gate BN frozen; trunk has LayerNorm only (no mode effect)
    params = [{"params": model.trunk.parameters(), "lr": a.lr}, {"params": model.net.parameters(), "lr": a.gate_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    ema = copy.deepcopy(model).eval(); swa = copy.deepcopy(model).eval(); n_swa = 0
    for p in list(ema.parameters()) + list(swa.parameters()):
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); step = 0; losses = []
    budget = 30 if a.smoke else a.budget
    while time.time() - t0 < budget:
        opt.zero_grad(set_to_none=True); tot = 0.0
        for _ in range(a.accum):
            x, y = sample_batch(train, a.micro, gen, a.onset)
            loss = FGvrmse(model(x), y).mean() / a.accum; loss.backward(); tot += float(loss)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1; losses.append(tot)
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(a.ema).add_(pm, alpha=1 - a.ema)
            if time.time() - t0 > 0.25 * budget:
                n_swa += 1
                for ps, pm in zip(swa.parameters(), model.parameters()):
                    ps.mul_((n_swa - 1) / n_swa).add_(pm, alpha=1 / n_swa)
        if step % 100 == 0 or step == 1:
            print(f"[ft] {tag} step {step} loss(avg100) {np.mean(losses[-100:]):.5f} {time.time() - t0:.0f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    t_train = time.time() - t0
    if n_swa == 0:
        swa.load_state_dict(model.state_dict())
    res = {}
    for name, m in (("final", model), ("ema", ema), ("swa", swa)):
        res[name] = score(m, valid, pairs)
    model.load_state_dict(start_state); res["start"] = score(model, valid, pairs)
    w = CM.weights_for(doc, sid[:len(pairs)])
    summ = {k: float((w / w.sum() * v.mean(1)).sum()) for k, v in res.items()}
    fn = CM.OUT / f"ft_{tag}.npz"
    np.savez_compressed(fn, sid=sid[:len(pairs)], **{k: v for k, v in res.items()})
    if not a.smoke:
        torch.save({"model": ema.state_dict()}, CM.OUT / f"ft_{tag}_ema.pt")
        torch.save({"model": swa.state_dict()}, CM.OUT / f"ft_{tag}_swa.pt")
    meta = {"tag": tag, "args": vars(a), "steps": step, "train_s": t_train, "n_swa": n_swa,
            "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "pop_weighted_valid_sample": summ}
    (CM.OUT / f"ft_{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[ft] {tag}: {step} steps in {t_train:.0f}s; pop-weighted VALID-sample VRMSE {summ}", flush=True)


def FGvrmse(pred, y):
    return CM.FG.vrmse(pred, y)


if __name__ == "__main__":
    main()
