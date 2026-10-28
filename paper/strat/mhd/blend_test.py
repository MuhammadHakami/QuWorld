"""Doc 35 §37h step 4: the VALID-picked two-backbone band blend scored ONCE on the full MHD_64 TEST split.

Members per window: Flowers (final) and the gated CNextU-net, single pass (headline) and 8-pass shg8 averages
(daggered), each with the divergence-free B projection; persistence = the last input frame. Weights (fit on TRAIN,
runs/strat/mhd/blend_final.json) act per Alfven regime x channel x radial band in Fourier space (bands of analyze.py);
the k = 0 band of density and B is taken from persistence (conserved means). Regime: physical mean B_x of the last
input frame > 0.5 -> Ma 0.7, else Ma 2 (eval_strat's feature). Per-channel and field-grouped VRMSE.

    flock runs/strat/gpu.lock env QW_GPU_GB=16 STRAT_FLOWER_CKPT=runs/phase_b_mhd/flower_s1337.pt runs/capped.sh s37h 24 \
        uv run --no-sync python paper/strat/mhd/blend_test.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_strat as ES  # noqa: E402  (cap_from_env first)
import torch  # noqa: E402
BANDS = [[0], [1], [2, 3], list(range(4, 8)), list(range(8, 16)), list(range(16, 32)), list(range(32, 56))]  # = analyze.BANDS
from mhd_divfree import DivFreeB  # noqa: E402

OUT = ES.OUT


def band_of_shell():
    b = torch.zeros(ES.NSHELL, dtype=torch.long)
    for i, shells in enumerate(BANDS):
        b[shells] = i
    return b


def weights_table(cand: dict, names: list[str]):
    """(2 regimes, V members, 7, nb) weights from the anchor-based theta (anchor gets 1 - sum theta)."""
    nb = len(BANDS); W = torch.zeros(2, len(names), 7, nb, dtype=torch.float64)
    for key, th in cand["theta"].items():
        k = json.loads(key)
        regs, c, b = ([k[0]], k[1], k[2]) if cand["group"] == "Ma_ch_band" else ([0, 1], k[0], k[1])
        for r in regs:
            W[r, 0, c, b] = 1.0 - sum(th)
            for j, t in enumerate(th):
                W[r, 1 + j, c, b] = t
    if cand["cons"]:
        p = names.index("P"); W[:, :, :4, 0] = 0.0; W[:, p, :4, 0] = 1.0
    return W


@torch.no_grad()
def main() -> None:
    sel = json.loads((OUT / os.environ.get("STRAT_BLEND_OUT", "blend_final.json")).read_text())
    picks = {"blend": sel["candidates"][sel["picked"]], "blend_shg8": sel["candidates"][sel["picked_shg8"]],
             # check arm: Flowers alone + projections through the same Fourier path; must reproduce §34 dfc 0.11906
             "check_F_dfc": {"members": ["F_raw_df", "P"], "group": "ch_band", "cons": True,
                             "theta": {f"[{c}, {b}]": [0.0] for c in range(7) for b in range(len(BANDS))}}}
    models, meta = ES.load_models()
    test = ES.WellND("MHD_64", "test")
    m = torch.tensor(test.mean, device=ES.DEV); s = torch.tensor(test.std, device=ES.DEV)
    proj = DivFreeB(torch.nn.Identity()).to(ES.DEV)
    groups = ES.pass_groups("trainlite")
    shells = ES.shell_index(); band = band_of_shell().to(ES.DEV)[shells]                     # (N^3,) band per wavenumber
    tables = {a: weights_table(c, c["members"]).to(ES.DEV) for a, c in picks.items()}
    out = {a: [] for a in picks}
    pairs = [(k, t) for k in range(test.n) for t in range(ES.NH - 1, test.T - 1)]
    for i, (k, t) in enumerate(pairs):
        x, y = (v.unsqueeze(0).to(ES.DEV) for v in test.window(k, t))
        xs = x.view(1, ES.NH, ES.C, *x.shape[-3:])
        preds = {}
        for name, model in models.items():
            sums = {}
            for g, syms in groups.items():
                acc = 0
                for e in syms:
                    acc = acc + e.inv(model(e.fwd(xs, (m, s)).reshape(1, ES.NH * ES.C, *x.shape[-3:])), (m, s))
                sums[g] = acc
            for vn, p in (("raw", sums["id"]), ("shg8", (sums["id"] + sums["comb"]) / 8)):
                preds[f"{name}_{vn}_df"] = torch.cat([p[:, :1], proj.project(p[:, 1:4]), p[:, 4:]], 1)
                preds[f"{name}_{vn}"] = p
        preds["P"] = x[:, -ES.C:]
        reg = 0 if float((x[0, -ES.C + 1] * s[1] + m[1]).mean()) > 0.5 else 1
        var = (y - y.mean((2, 3, 4), keepdim=True)).square().mean((2, 3, 4))
        for a, c in picks.items():
            W = tables[a][reg]                                                          # (V, 7, nb)
            Fh = torch.stack([torch.fft.fftn(preds[n][0], dim=(-3, -2, -1)).reshape(ES.C, -1) for n in c["members"]])
            blend = (W[:, :, band].to(Fh.dtype) * Fh).sum(0).reshape(ES.C, *x.shape[-3:])
            p = torch.fft.ifftn(blend, dim=(-3, -2, -1)).real.unsqueeze(0).float()
            mse = (p - y).square().mean((2, 3, 4))
            out[a].append((mse / (var + 1e-7)).clamp_min(1e-12).sqrt().float().cpu())
        if i % 100 == 0:
            print(f"[s37h-test] {i}/{len(pairs)}", flush=True)
    res = {"picked": sel["picked"], "picked_shg8": sel["picked_shg8"], "flower_step": meta["flower_step"], "arms": {}}
    for a, v in out.items():
        v = torch.cat(v); g = torch.stack([v[:, 0], v[:, 1:4].mean(1), v[:, 4:7].mean(1)], 1).mean(1)
        res["arms"][a] = {"per_channel": float(v.mean()), "field_grouped": float(g.mean()), "channels": v.mean(0).tolist(), "n": len(v)}
    res["check_ok"] = abs(res["arms"]["check_F_dfc"]["per_channel"] - float(os.environ.get("STRAT_CHECK_DFC", "0.11906"))) < 2e-4
    (OUT / os.environ.get("STRAT_BLEND_TEST_OUT", "blend_test.json")).write_text(json.dumps(res, indent=1))
    print("S37H_TEST " + " ".join(f"{a} {r['per_channel']:.5f} (grouped {r['field_grouped']:.5f})" for a, r in res["arms"].items()), flush=True)


if __name__ == "__main__":
    main()
