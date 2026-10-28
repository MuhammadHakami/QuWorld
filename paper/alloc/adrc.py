"""ADR-C v1 (35-analog-digital-router-configuration-losses.md §6): analog components to qubits in a learned
configuration, everything else to the classical head; per-input x, y, z and tethers trained with GSL-style losses.

    .venv/bin/python paper/alloc/adrc.py <row> --arm adrc_ryd --seed 1

arms: cls_only (all components classical) | static_lin (fixed top-30 by TRAIN energy, linear) |
      adrc_ryd | adrc_gnn (classical GNN on the same learned graph) | adrc_mlp | frozen_ryd (no config losses, fixed
      grid, no tethers) | adrc_gutz (mean-field register on the learned graph) | rand_ryd (random routing)
Qubit-routed components get c = s (W phi + A o), classical ones c = s W_cls phi; A zero-init. Exact metric.
v2 caches (doc 35 §8, band-wise POD) add a band one-hot per component and a per-band TEST report: for every band b and
window, captured_b <= span_b (gain_le_ceiling in the band's orthonormal sub-basis) <= Eb (Bessel inside range(P_b)),
and the bands add up (band_energy_split); the report records the largest violation of each inequality (fp16 basis).
--gate (blend caches, doc 36 §5): a classical per-window router between persistence and the trunk, alpha (C, B) =
alpha_fit + 1.5 tanh(net(causal band energies of p - x_t and dx, global stats)); the residual energy and every mode
coefficient are exact functions of alpha (adrc_cache.py docstring), trained jointly with the arm. Arm `none` (no
co-processor) isolates the gate.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "paper"))
import gpucap  # noqa: E402
from cores import TileRegister, n_params  # noqa: E402
from fgn import ZD, NoiseFiLM, dress_report, evaluate_prob, fair_es, zero_linear  # noqa: E402

gpucap.cap_from_env()
DEV = torch.device("cuda")
CACHE, OUT = ROOT / "runs/alloc/adrc_cache", ROOT / "runs/alloc/adrc_results"
OUT = pathlib.Path(__import__("os").environ.get("QW_OUT", OUT))                           # replays must not overwrite results
K, NG, G = 30, 10, 3            # atoms, atoms per register group, groups; set by --atoms (doc 35 §13)
NT = lambda: max(3, G)          # tethers: chain g-1 -> g (indices 0..G-2) + skip 0 -> 2 (index G-1); G = 3 is the §13 layout
DYN_TETHER = False              # doc 35 §13: tether strengths conditioned on each input
FGN = False                     # doc 35 §17: noise-conditioned configuration, energy-score training


class Data:
    def __init__(self, row, split, stats=None):
        d = torch.load(CACHE / row / f"{split}.pt", weights_only=False)
        self.a, self.f = d["a"].float().to(DEV), d["f"].float().to(DEV)
        self.E, self.D, self.g = d["E"].to(DEV), d["D"].to(DEV), d["g"].to(DEV)
        self.N, self.C, self.M = self.a.shape
        self.etot = (self.E / self.D).sum(1)
        self.Eb = d["Eb"].to(DEV) if "Eb" in d else None                    # (N, C, B) full-field band energies
        self.E0 = d["E0"].to(DEV) if "E0" in d else None                    # unblended trunk (band-gated blend caches)
        meta = torch.load(CACHE / row / "basis_meta.pt", weights_only=False)
        self.band = meta["band"].to(DEV) if "band" in meta else None        # (C, M) band of every mode
        self.nb = int(meta.get("n_bands", 0))
        self.gate = "gY" in d
        if self.gate:
            self.ay, self.fd0 = d["ay"].float().to(DEV), d["fd0"].float().to(DEV)
            self.gY, self.gYD, self.gDD = (d[k].to(DEV) for k in ("gY", "gYD", "gDD"))
            self.alpha_fit = meta["blend_alpha"].to(DEV)
            # CAUSAL normalization only: variance of the last input frame x_t (g block 3 = [mean, std] of x_t per
            # field). An earlier version divided by D = npix var(y), i.e. the TARGET's variance (leak; runs archived
            # in runs/alloc/adrc_results_leaky_gate/, doc 36 §8).
            C = self.C; vx = self.g[:, 5 * C:6 * C].double().square() + 1e-7
            Dn = vx[..., None]
            self.z = torch.cat([(self.gDD / Dn).clamp_min(1e-12).log().flatten(1),
                                (d["gX"].to(DEV) / Dn).clamp_min(1e-12).log().flatten(1), self.g.double()], 1).float()
        self.stats = stats if stats is not None else {
            "gmu": self.g.mean(0), "gsd": self.g.std(0) + 1e-6,
            **({"zmu": self.z.mean(0), "zsd": self.z.std(0) + 1e-6} if self.gate else {})}
        self.gmu, self.gsd = self.stats["gmu"], self.stats["gsd"]

    def phi(self, rows):
        f = self.f[rows]                                                   # (B, C, M, 4)
        B, C, M, _ = f.shape
        rms = f[..., [0, 3]].square().mean((1, 2, 3)).sqrt()[:, None, None, None] + 1e-12
        s = (f[..., [0, 3]].square().mean(-1, keepdim=True) + (1e-3 * rms) ** 2).sqrt()   # (B, C, M, 1)
        oh = torch.eye(C, device=DEV)[None, :, None].expand(B, C, M, C)
        rank = (torch.arange(M, device=DEV).float() / M)[None, None, :, None].expand(B, C, M, 1)
        g = ((self.g[rows] - self.gmu) / self.gsd).clamp(-10, 10)[:, None, None].expand(B, C, M, -1)
        parts = [(f / s).clamp(-20, 20), (s / (1e-3 * rms)).log() / 5, oh, rank, g]
        if self.band is not None:
            parts.append(nn.functional.one_hot(self.band, self.nb).float()[None].expand(B, C, M, self.nb))
        x = torch.cat(parts, -1)
        return x.reshape(B, C * M, -1), s.reshape(B, C * M, 1)

    def zin(self, rows):
        return ((self.z[rows] - self.stats["zmu"]) / self.stats["zsd"]).clamp(-10, 10)

    def resid(self, rows, alpha=None):
        """Mode coefficients (B, C*M) and full-field residual energy (B, C) for gate alpha (B, C, nb) or the cached anchor."""
        if alpha is None:
            return self.a[rows].reshape(len(rows), -1).double(), self.E[rows]
        ab = alpha.gather(2, self.band[None].expand(len(rows), -1, -1))
        a = (self.ay[rows] - ab * self.fd0[rows]).reshape(len(rows), -1).double(); al = alpha.double()
        return a, (self.gY[rows] - 2 * al * self.gYD[rows] + al.square() * self.gDD[rows]).sum(-1)

    def metric(self, rows, corr, alpha=None):                              # corr (B, C*M) raw coefficient corrections
        a, E = self.resid(rows, alpha); c = corr.double()
        gain = (a.square() - (a - c).square()).reshape(len(rows), self.C, self.M).sum(-1)
        Enew = (E - gain).clamp_min(0)
        return ((Enew / self.D[rows]).sqrt().mean(1), (gain / self.D[rows]).sum(1) / self.etot[rows],
                (Enew / self.D[rows]).sum(1) / self.etot[rows])


class Gate(nn.Module):
    """Classical digital router: per window, field and band, how much of the trunk's increment over persistence to keep."""

    def __init__(self, fin, alpha_fit):
        super().__init__()
        self.register_buffer("alpha_fit", alpha_fit.float())
        self.net = nn.Sequential(nn.Linear(fin, 64), nn.GELU(), nn.Linear(64, alpha_fit.numel()))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.alpha_fit + 1.5 * torch.tanh(self.net(z)).reshape(len(z), *self.alpha_fit.shape)


class GNNCore(nn.Module):
    """Classical control on the same learned graph: one round of message passing with weights V_ij (normalized)."""

    def __init__(self, F, target, h=None):
        super().__init__()
        h = h or max(2, round((target - 3) / (3 * F + 10)))
        self.inp = nn.Linear(F, h); self.msg = nn.Linear(h, h); self.out = nn.Linear(2 * h, 3)

    def forward(self, x, V):
        h = torch.tanh(self.inp(x))
        A = V / V.sum(-1, keepdim=True).clamp_min(1e-9)
        return torch.tanh(self.out(torch.cat([h, torch.tanh(self.msg(A @ h))], -1)))


class ADRC(nn.Module):
    def __init__(self, F, arm, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.arm = arm
        self.W_cls = nn.Linear(F, 1); nn.init.zeros_(self.W_cls.weight); nn.init.zeros_(self.W_cls.bias)
        self.W_q = nn.Linear(F, 1); nn.init.zeros_(self.W_q.weight); nn.init.zeros_(self.W_q.bias)
        self.scorer = nn.Sequential(nn.Linear(F, 32), nn.GELU(), nn.Linear(32, 2))         # [advantage, group key]
        HH = int(__import__("os").environ.get("QW_HYP", "32"))                                                    # doc 35 §24 H5: hypernet width
        self.hyp = nn.Sequential(nn.Linear(2 * F, HH), nn.GELU(), nn.Linear(HH, 3))        # per-atom (dx, dy, dz)
        nn.init.zeros_(self.hyp[-1].weight); nn.init.zeros_(self.hyp[-1].bias)
        if __import__("os").environ.get("QW_3D") == "rand":                                                         # §24 H1: every coordinate moves
            nn.init.normal_(self.hyp[-1].weight, std=0.01)
        self.tether_logit = nn.Parameter(torch.full((NT(),), float(__import__("os").environ.get("QW_TETHER_INIT", "-2.0"))))   # §24 H3                         # groups 0-1, 1-2, 0-2 (K <= 30)
        RO = 4 if (__import__("os").environ.get("QW_ZZ") and arm in ("adrc_ryd", "frozen_ryd")) else 3   # §24 H7 readout width
        self.tether_w = nn.Linear(RO, F); nn.init.zeros_(self.tether_w.weight); nn.init.zeros_(self.tether_w.bias)
        self.tether_hyp = nn.Linear(F, NT()) if DYN_TETHER else None                     # per-input tether logits
        if self.tether_hyp is not None:
            nn.init.zeros_(self.tether_hyp.weight); nn.init.zeros_(self.tether_hyp.bias)
        self.film = NoiseFiLM(F) if FGN else None                                          # doc 35 §17: core input
        self.hyp_xi = zero_linear(ZD, 32) if FGN else None                                  # offsets' hidden layer
        self.tether_xi = zero_linear(ZD, NT()) if FGN else None                             # tether logits
        mode = {"adrc_gutz": "gutz"}.get(arm, "ryd")
        reg = TileRegister(F, NG, int(__import__("os").environ.get("QW_LAYERS", "4")), mode); reg.log_lam.data.zero_(); reg.log_lam.requires_grad_(False)
        if __import__("os").environ.get("QW_LAM_XY"):              # doc 35 §33b B3: fixed in-plane scale (V x e^(6 lam)), not trained
            reg.log_lam.data[:2] = float(__import__("os").environ["QW_LAM_XY"])
        if __import__("os").environ.get("QW_TAU0"):                 # doc 35 §33c: interaction-free register (tau = 0, frozen)
            reg.tau.data.zero_(); reg.tau.requires_grad_(False)
        if arm == "adrc_gnn":
            self.core = GNNCore(F, n_params(reg))
        elif arm == "adrc_gnnm":                                     # doc 35 §26b: GNN width closest to the register's count
            hs = range(2, 12); tgt = n_params(reg)                   # (h = 4: 4F+51 against the register's 4F+56, every F)
            self.core = GNNCore(F, tgt, h=min(hs, key=lambda h: abs(h * (F + 8 + h) + 3 - tgt)))
        elif arm == "adrc_mlp":
            from cores import GroupMLP
            self.core = GroupMLP(F, n_params(reg))
        elif arm == "adrc_mlpw":                                     # doc 35 §17 wider control: nearest width + 1 unit
            from cores import GroupMLP
            h0 = max(1, round((n_params(reg) - 3) / (2 * F + 4)))
            self.core = GroupMLP(F, (h0 + 1) * (2 * F + 4) + 3)
        else:
            self.core = reg
        zz = bool(__import__("os").environ.get("QW_ZZ")) and arm in ("adrc_ryd", "frozen_ryd")
        self.A = nn.Linear(4 if zz else 3, 1); nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)   # §24 H7: +<ZZ> channel
        gx, gy = torch.meshgrid(torch.arange(5.0), torch.arange(float(NG // 5)), indexing="ij")
        g3 = __import__("os").environ.get("QW_3D")
        gz = (0.6 * torch.randn(NG, generator=torch.Generator().manual_seed(7)) if g3 == "rand"   # §24 H1: random 3-D lattice
              else (torch.arange(NG) % 10 >= 5).float() * 1.2 if g3 else torch.zeros(NG))          # §22: bilayer
        self.register_buffer("grid", torch.stack([gx.flatten() * 1.2, gy.flatten() * 1.2, gz], -1))

    def tethers(self, ctx=None, noise=None):
        if self.arm in ("frozen_ryd", "rand_ryd"):
            return torch.zeros(NT(), device=DEV)
        logit = self.tether_logit if ctx is None or self.tether_hyp is None else self.tether_logit + self.tether_hyp(ctx)
        if noise is not None and self.tether_xi is not None:
            logit = logit + self.tether_xi(noise)
        mode = __import__("os").environ.get("QW_TETHER_MODE", "hc")
        if mode == "det":                                   # doc 35 §33: continuous strength, identical at train and eval
            return torch.sigmoid(logit)
        if mode == "off":                                   # doc 35 §33: matched no-tether control
            return torch.zeros_like(logit)
        u = torch.rand(logit.shape, device=DEV).clamp(1e-6, 1 - 1e-6) if self.training else torch.full(logit.shape, 0.5, device=DEV)
        s = torch.sigmoid((u.log() - (1 - u).log() + logit) / 0.5)                        # hard-concrete-style gate
        return (s * 1.2 - 0.1).clamp(0, 1)

    def forward(self, x, s, route=None, noise=None):
        B, J, F = x.shape
        if self.film is not None and noise is None:
            noise = torch.zeros(B, ZD, device=DEV)                                          # the xi = 0 member
        corr = s.squeeze(-1) * self.W_cls(x).squeeze(-1)                                     # classical head, all comps
        if getattr(self, "keep_cls", 1.0) != 1.0:                                            # §28c credit: head removed
            corr = corr * self.keep_cls
        aux = {}
        if self.arm in ("cls_only",):
            return corr, aux
        if self.arm == "none":
            return corr * 0, aux
        sc = self.scorer(x)
        sc0 = sc[..., 0]
        if getattr(self, "band_mask", None) is not None:
            sc0 = sc0.masked_fill(~self.band_mask[None], -1e9)
        if self.arm == "static_lin":
            idx = route[None].expand(B, -1)
        elif self.arm == "rand_ryd":
            if getattr(self, "band_mask", None) is not None:
                pool = torch.nonzero(self.band_mask).squeeze(1)
                idx = pool[torch.randperm(len(pool), device=DEV)[:K]][None].expand(B, -1)
            else:
                idx = torch.rand(B, J, device=DEV).topk(K, 1).indices
        else:
            idx = sc0.topk(K, 1).indices
        aux["score"] = sc0
        gat = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))
        xq, sq = gat(x), gat(s)
        soft = torch.sigmoid(sc0.gather(1, idx)[:, :, None] - sc0.topk(K + 1, 1).values[:, -1:, None])  # straight-through
        mask = 1 + soft - soft.detach()
        if self.arm == "static_lin":
            cq = self.W_q(xq)
        else:
            key = gat(sc[..., 1:2]).squeeze(-1); order = key.argsort(1)
            xq, sq, mask, idx = (t.gather(1, order[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else t.gather(1, order)
                                 for t in (xq, sq, mask, idx))
            xg = xq.reshape(B * G, NG, F)
            t = self.tethers(xq.mean(1), noise); aux["tethers"] = t
            t = t if t.dim() == 2 else t[None].expand(B, -1)                                   # (B, 3)
            perm = getattr(self, "cf_perm", None)                   # doc 35 §33 eval-only counterfactual: another input's
            if perm is not None and getattr(self, "cf_tether", False):                        # tethers for this input
                t = t[perm]
            ro = []
            outs, pos_all = [], []
            for gi in range(G):
                xi = xg.reshape(B, G, NG, F)[:, gi]
                if gi > 0:                                                                   # feed-forward tethers
                    tin = t[:, gi - 1:gi] * ro[gi - 1] + (t[:, NT() - 1:NT()] * ro[0] if gi == 2 else 0)
                    xi = xi + self.tether_w(tin)[:, None]
                if self.arm == "frozen_ryd":
                    pos = self.grid[None].expand(B, -1, -1)
                else:
                    ctx = xi.mean(1, keepdim=True).expand_as(xi)
                    h = self.hyp[0](torch.cat([xi, ctx], -1))
                    if self.hyp_xi is not None:
                        h = h + self.hyp_xi(noise)[:, None]
                    pos = self.grid[None] + float(__import__("os").environ.get("QW_SHIFT", "0.45")) * torch.tanh(self.hyp[2](self.hyp[1](h)))   # §24 H2
                if getattr(self, "cf_grid", False):                  # doc 35 §33 eval-only: the fixed layout for every input
                    pos = self.grid[None].expand(B, -1, -1)
                elif perm is not None and getattr(self, "cf_pos", False):                     # another input's positions
                    pos = pos[perm]
                pos_all.append(pos)
                coords = torch.cat([pos, torch.zeros(B, NG, 1, device=DEV)], -1)
                xin = self.film(xi, noise) if self.film is not None else xi
                if isinstance(self.core, GNNCore):
                    V = torch.cdist(pos, pos).clamp(min=0.5).pow(-6) * (1 - torch.eye(NG, device=DEV))
                    o = self.core(xin, V)
                else:
                    o = self.core(xin, coords)
                ro.append(o.mean(1))
                # relay atoms: an active tether into group gi takes its last 2 atoms (their components go classical)
                if gi > 0:
                    relay = torch.ones(B, NG, device=DEV); relay[:, -2:] = 1 - t[:, gi - 1:gi]
                    o = o * relay[:, :, None]
                outs.append(o)
            o = torch.cat(outs, 1)
            aux["pos"] = torch.cat(pos_all, 1); aux["xq"] = xq
            cq = self.W_q(xq) + self.A(o)
            if getattr(self, "keep_cls", 1.0) != 1.0 or getattr(self, "keep_q", 1.0) != 1.0 or getattr(self, "trust", None):
                from handoff import trust_factor                                              # §28c: credit / handoff
                cq = self.keep_cls * self.W_q(xq) + self.keep_q * trust_factor(self.trust, o, NG) * self.A(o)
        cq = sq * cq * mask
        corr = corr.scatter_add(1, idx, cq.squeeze(-1) - corr.gather(1, idx) * (self.arm != "static_lin"))
        aux["idx"] = idx
        return corr, aux


def gsl_losses(aux):
    if "pos" not in aux:
        return torch.zeros((), device=DEV)
    pos, xq = aux["pos"], aux["xq"]
    B = pos.shape[0]
    P, X = pos.reshape(B * G, NG, 3), nn.functional.normalize(xq.reshape(B * G, NG, -1), dim=-1)
    dist = torch.cdist(P, P) + torch.eye(NG, device=DEV) * 10
    V = dist.clamp(min=0.5).pow(-6)
    Vn = V / V.sum(-1, keepdim=True).clamp_min(1e-9)
    dirichlet = (Vn * torch.cdist(X, X).square()).sum((1, 2)).mean()                          # coupled atoms carry similar signals
    barrier = -V.sum(-1).clamp_min(1e-9).log().mean()                                        # no isolated atom
    spacing = (0.5 - dist).clamp_min(0).square().sum((1, 2)).mean()                          # hardware minimum spacing
    return dirichlet + 0.01 * barrier + 10 * spacing


@torch.no_grad()
def band_report(d, model, route, gate=None):
    """Per-band decomposition of the converged corrector on split d (None for v1 caches). Shares are metric-weighted
    fractions of the window's total residual error, averaged over windows."""
    if d.band is None:
        return None
    model.eval(); bflat = d.band.flatten(); nb = d.nb
    acc = {k: [] for k in ("full", "span", "cap", "capq", "slots")}
    for i in range(0, d.N, 64):
        rows = torch.arange(i, min(i + 64, d.N), device=DEV); x, s = d.phi(rows)
        corr, aux = model(x, s, route)
        al = gate(d.zin(rows)) if gate is not None else None
        a, _ = d.resid(rows, al); c = corr.double()
        w = (1 / (d.D[rows] * d.etot[rows][:, None])).repeat_interleave(d.M, 1)
        e, g = a.square() * w, (a.square() - (a - c).square()) * w
        q = torch.zeros_like(e)
        if "idx" in aux:
            q.scatter_(1, aux["idx"], 1.0)
        per = lambda t: torch.stack([t[:, bflat == b].sum(1) for b in range(nb)], 1)
        acc["span"].append(per(e)); acc["cap"].append(per(g)); acc["capq"].append(per(g * q)); acc["slots"].append(per(q))
        Eb = d.Eb[rows] if al is None else d.gY[rows] - 2 * al.double() * d.gYD[rows] + al.double().square() * d.gDD[rows]
        acc["full"].append((Eb / d.D[rows][..., None]).sum(1) / d.etot[rows][:, None])
    model.train()
    t = {k: torch.cat(v) for k, v in acc.items()}
    mean = lambda k: [float(v) for v in t[k].mean(0)]
    return {"full": mean("full"), "span": mean("span"), "captured": mean("cap"), "captured_qubit": mean("capq"),
            "captured_classical": [x - y for x, y in zip(mean("cap"), mean("capq"))],
            "qubit_slot_share": [float(v) / K for v in t["slots"].mean(0)] if t["slots"].sum() > 0 else None,
            "max_viol_cap_le_span": float((t["cap"] - t["span"]).max()), "max_viol_span_le_full": float((t["span"] - t["full"]).max()),
            "bands_sum_to_total": float((t["full"].sum(1) - 1).abs().max()) if gate is None else None}


def run(a):
    tr = Data(a.row, "train")
    va = Data(a.row, "valid", tr.stats)
    te = va if __import__("os").environ.get("QW_SCREEN") else Data(a.row, "test", tr.stats)   # screens never load TEST
    for d_ in (tr, va, te):                                   # doc 35 §30: fixed per-field TRAIN scale for a proper ES
        d_.Dref = tr.D.double().mean(0, keepdim=True)
    ES_REF = __import__("os").environ.get("QW_ES_NORM") == "train"    # train the FGN objective with the proper scale
    assert not a.gate or tr.gate, "--gate needs a blend cache (ADRC_BLEND)"
    gate = Gate(tr.z.shape[1], tr.alpha_fit).to(DEV) if a.gate else None
    gate_off = [False]                                                                    # §28c credit: gate removed
    al = lambda d, rows: (gate.alpha_fit[None].expand(len(rows), *gate.alpha_fit.shape) if gate_off[0] else gate(d.zin(rows))) \
        if gate is not None else None
    energy = torch.load(CACHE / a.row / "basis_meta.pt", weights_only=False)["energy"].to(DEV)   # (C, M)
    route = energy.flatten().topk(K).indices
    F = tr.phi(torch.arange(2, device=DEV))[0].shape[-1]
    model = ADRC(F, a.arm, a.seed).to(DEV)
    if __import__("os").environ.get("QW_NO_CORE"):             # doc 35 §33a: same architecture, core term off from step 0
        model.keep_cls, model.keep_q, model.trust = 1.0, 0.0, None
    if a.qubit_bands:
        assert tr.band is not None, "--qubit_bands requires a band-wise cache (d.band is not None)"
        q_bands = [int(x) for x in a.qubit_bands.split(",")]
        model.band_mask = torch.isin(tr.band.flatten(), torch.tensor(q_bands, device=DEV))
    params = [p for p in model.parameters() if p.requires_grad] + (list(gate.parameters()) if gate is not None else [])
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)
    gdict = lambda: {k: v.clone() for k, v in gate.state_dict().items()} if gate is not None else None
    gen = torch.Generator().manual_seed(a.seed)

    @torch.no_grad()
    def evaluate(d):
        model.eval(); ms, cs = [], []
        for i in range(0, d.N, 64):
            rows = torch.arange(i, min(i + 64, d.N), device=DEV); x, s = d.phi(rows)
            m, c, _ = d.metric(rows, model(x, s, route)[0], al(d, rows)); ms.append(m); cs.append(c)
        model.train()
        return torch.cat(ms).mean().item(), torch.cat(cs).mean().item()

    es_scale = (tr.E / tr.D).sqrt().mean().item()                                         # doc 35 §17 loss scale
    select = (lambda: evaluate_prob(va, model, route, al, 8, 11)["es"]) if FGN else (lambda: evaluate(va)[0])
    best, state, gstate, bad, t0 = select(), {k: v.clone() for k, v in model.state_dict().items()}, gdict(), 0, time.time()
    LOSS_VRMSE = __import__("os").environ.get("QW_LOSS") == "vrmse"            # doc 35 §33 B2: train on the reported metric
    TGT_VRMSE = __import__("os").environ.get("QW_ROUTE_TGT") == "vrmse"         # doc 35 §33 B2: route by exact VRMSE decrement
    for ep in range(int(__import__("os").environ.get("QW_MAX_EP", "60"))):
        perm = torch.randperm(tr.N, generator=gen).to(DEV)
        for i in range(0, tr.N - 31, 32):
            rows = perm[i:i + 32]; x, s = tr.phi(rows)
            if FGN:                                                                               # fair energy score, 2 members
                corr, aux = model(x, s, route, torch.randn(len(rows), ZD, device=DEV)); alpha = al(tr, rows)
                corr2, _ = model(x, s, route, torch.randn(len(rows), ZD, device=DEV))
                loss = fair_es(tr, rows, [corr, corr2], alpha, 1e-12, ref=ES_REF).mean().float() / es_scale
            else:
                corr, aux = model(x, s, route); alpha = al(tr, rows)
                vr, cap, err = tr.metric(rows, corr, alpha)
                loss = vr.mean().float() if LOSS_VRMSE else (err.mean() if gate is not None else -cap.mean())  # same optimum w/o gate
            if a.arm not in ("cls_only", "static_lin", "frozen_ryd", "rand_ryd", "none"):
                loss = loss + 0.01 * gsl_losses(aux) + 0.002 * (2 * aux["tethers"]).reshape(-1, NT()).sum(-1).mean() / K    # tether atoms charged
            if "idx" in aux and a.arm != "static_lin":                                            # utilization band
                acoef = tr.resid(rows, alpha)[0].float().gather(1, aux["idx"]).square()
                share = acoef / (tr.D[rows].mean(1, keepdim=True) * tr.etot[rows][:, None])
                loss = loss + 1.0 * (1e-4 - share).clamp_min(0).mean()                           # starved-qubit penalty
            if a.route_sup and "score" in aux:                    # utilization-supervised router (doc 35 §9): learn,
                e = tr.resid(rows, alpha)[0].float().square() / tr.D[rows].repeat_interleave(tr.M, 1).float()
                if TGT_VRMSE:                                     # single-mode exact VRMSE decrement, field c: [sqrt(E_c/D_c)
                    Ec = (tr.resid(rows, alpha)[1] / tr.D[rows]).float().repeat_interleave(tr.M, 1)   # - sqrt((E_c-a_i^2)/D_c)]/C
                    e = (Ec.sqrt() - (Ec - e).clamp_min(0).sqrt()) / tr.C
                if getattr(model, "band_mask", None) is not None:
                    e = e.masked_fill(~model.band_mask[None], -1e9)
                tgt = torch.zeros_like(e).scatter_(1, e.topk(K, 1).indices, 1.0)                 # causally, where the
                loss = loss + 0.1 * nn.functional.binary_cross_entropy_with_logits(aux["score"], tgt)  # energy is (max U)
            if not loss.requires_grad:                                                            # arm none w/o gate
                break
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        v = select()
        if v < best - 1e-9:
            best, state, gstate, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, gdict(), 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(state)
    if gate is not None:
        gate.load_state_dict(gstate)
    if __import__("os").environ.get("QW_SCREEN"):                  # doc 35 §24 screen: VALID only, TEST never touched
        te = va
    tm, tc = evaluate(te)
    if __import__("os").environ.get("QW_DUMP_WINDOW"):                          # eval-only: one TEST window's output (Fig. 2)
        pth, wi = __import__("os").environ["QW_DUMP_WINDOW"].rsplit(":", 1); rows = torch.tensor([int(wi)], device=DEV)
        model.eval()
        with torch.no_grad():
            x, s = te.phi(rows); corr, aux = model(x, s, route)
            alw = al(te, rows) if gate is not None else None
        __import__("numpy").savez(pth, corr=corr.reshape(te.C, te.M).cpu().numpy(), window=int(wi), C=te.C, M=te.M,
                                  alpha=(alw[0].cpu().numpy() if alw is not None else __import__("numpy").zeros(0)),
                                  idx=(aux["idx"].cpu().numpy() if "idx" in aux else __import__("numpy").zeros(0)), test=tm)
        model.train()
    if __import__("os").environ.get("QW_DUMP_GEOM") and "ryd" in a.arm:                       # eval-only: per-input configuration
        model.eval()
        with torch.no_grad():
            rows = torch.linspace(0, te.N - 1, 8, device=DEV).long(); x, s = te.phi(rows)
            _, aux = model(x, s, route)
        dump = {"pos": aux["pos"], "idx": aux["idx"], "rows": rows}
        if "tethers" in aux:
            dump["tethers"] = aux["tethers"] if aux["tethers"].dim() == 2 else aux["tethers"][None].expand(len(rows), -1)
        if te.band is not None:
            dump["band"] = te.band.flatten()[aux["idx"]]
        __import__("numpy").savez(__import__("os").environ["QW_DUMP_GEOM"], **{k: v.detach().cpu().numpy() for k, v in dump.items()})
        model.train()
    base = te.metric(torch.arange(te.N, device=DEV), torch.zeros(te.N, te.C * te.M, device=DEV))[0].mean().item()
    res = {"row": a.row, "arm": a.arm, "seed": a.seed, "gate": bool(a.gate), "trunk_test": base, "test": tm, "test_rel_gain": 1 - tm / base,
           "test_captured": tc, "valid": best, "epochs": ep + 1, "seconds": time.time() - t0,
           "tethers": model.tethers().tolist() if a.arm not in ("cls_only", "static_lin", "none") else None,
           "params_core": n_params(model.core), "bands_test": band_report(te, model, route, gate),
           "trunk0_test": (te.E0 / te.D).sqrt().mean(1).mean().item() if te.E0 is not None else None}
    if FGN:                                               # doc 35 §17: M = 16 fixed draws, TEST once
        res["prob_valid"] = evaluate_prob(va, model, route, al, 8, 11)
        res["prob_test"] = evaluate_prob(te, model, route, al, 16, 12345)
        res["test_xi0"], res["test"] = tm, res["prob_test"]["vrmse_mean"]; res["test_rel_gain"] = 1 - res["test"] / base
        print(f"[adrc] {a.row} fgn TEST {res['prob_test']}", flush=True)
        if a.fgn_tau:                                     # doc 35 §17 (d): VALID-tuned spread scale, TEST once
            grid = {t: evaluate_prob(va, model, route, al, 8, 11, t)["es"] for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)}
            t = min(grid, key=grid.get)
            res["tau"] = {"valid_grid": grid, "tau": t, "prob_test": evaluate_prob(te, model, route, al, 16, 12345, t)}
            print(f"[adrc] {a.row} fgn tau={t}: TEST {res['tau']['prob_test']}", flush=True)
    if a.dress or a.dress_val or a.dress_rel:            # doc 35 §17 control: dressed deterministic ensembles
        assert not FGN, "--dress applies to deterministic runs"
        res["dress"] = dress_report(tr, va, te, model, route, al, sigma_from=va if a.dress_val else None, rel=a.dress_rel)
        with torch.no_grad():                                # doc 35 §30: deterministic model's proper ES (= its error)
            model.eval(); v = []
            for i in range(0, te.N, 64):
                rows = torch.arange(i, min(i + 64, te.N), device=DEV); x, s = te.phi(rows)
                v.append(fair_es(te, rows, [model(x, s, route)[0]], al(te, rows), ref=True))
            model.train(); res["det_es_proper"] = torch.cat(v).double().mean().item()
        print(f"[adrc] {a.row} dress: routed {res['dress']['routed']['test_es']:.5f} (s={res['dress']['routed']['scale']}) "
              f"all {res['dress']['all']['test_es']:.5f} (s={res['dress']['all']['scale']}) | det {tm:.5f}", flush=True)
    if a.shots:                                           # doc 35 §11: eval-only finite-shot readout, own output file
        assert isinstance(model.core, TileRegister), "--shots applies to the register arm only"
        res["test_shots"] = {}
        for n in (int(v) for v in a.shots.split(",")):
            model.core.shots = n; reps = []
            for r in range(3):
                torch.manual_seed(1000 * n + r); reps.append(evaluate(te)[0])
            res["test_shots"][str(n)] = reps
            print(f"[adrc] {a.row} shots={n}: TEST {sum(reps) / 3:.5f} (exact {tm:.5f})", flush=True)
        model.core.shots = 0
    if a.noise:                                           # doc 35 §12: eval-only calibration noise, own output file
        assert isinstance(model.core, TileRegister), "--noise applies to the register arm only"
        res["test_noise"] = {}
        for kind, attr in (("pos", "pos_sigma"), ("det", "det_sigma")):
            for lvl in (0.01, 0.05):
                setattr(model.core, attr, lvl * (1.2 if kind == "pos" else 1.0)); reps = []
                for r in range(3):
                    torch.manual_seed(7000 + int(1000 * lvl) + r); reps.append(evaluate(te)[0])
                res["test_noise"][f"{kind}{lvl}"] = reps; setattr(model.core, attr, 0)
                print(f"[adrc] {a.row} noise {kind} {lvl}: TEST {sum(reps) / 3:.5f} (exact {tm:.5f})", flush=True)
    if a.credit:                                          # doc 35 §28c: Shapley credit + handoff under hardware noise
        assert a.arm in ("adrc_ryd", "adrc_mlp"), "--credit applies to the register and its matched MLP"
        from handoff import credit_report
        model.keep_cls, model.keep_q, model.trust = 1.0, 1.0, None
        res["credit"] = credit_report(model, gate, va, te, evaluate, lambda off: gate_off.__setitem__(0, off))
        print(f"[adrc] {a.row} credit: shapley {res['credit']['shapley']} | handoff "
              f"{ {k: (round(v['always_core'], 5), round(v['snr']['test'], 5)) for k, v in res['credit']['handoff'].items()} }", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = ("_gate" if a.gate else "") + ("_rs" if a.route_sup else "") + (f"_qb{a.qubit_bands.replace(',', '')}" if a.qubit_bands else "") + ("_shots" if a.shots else "") + ("_noise" if a.noise else "") + (f"_k{K}" if K != 30 else "") + ("_dyn" if DYN_TETHER else "") + ("_fgn" if FGN else "") + ("t" if FGN and a.fgn_tau else "") + ("_dress" if a.dress else "") + ("_dressv" if a.dress_val else "") + ("_dressr" if a.dress_rel else "") + ("_credit" if a.credit else "")
    res["route_sup"] = bool(a.route_sup)
    res["atoms"], res["dyn_tether"], res["fgn"] = K, DYN_TETHER, FGN
    if K != 30 or DYN_TETHER:                              # doc 35 §13: oracle coverage U(K) and oracle VRMSE(K), any K
        Ks = (5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000); U = {k: [] for k in Ks}; V = {k: [] for k in Ks}
        with torch.no_grad():
            for i in range(0, te.N, 64):
                rows = torch.arange(i, min(i + 64, te.N), device=DEV); co, E = te.resid(rows, al(te, rows))
                Dr = te.D[rows].double(); e = co.square().reshape(len(rows), te.C, te.M) / Dr[:, :, None]
                ef = e.reshape(len(rows), -1); order = ef.argsort(1, descending=True)
                for k in Ks:
                    kk = min(k, ef.shape[1]); m = torch.zeros_like(ef).scatter_(1, order[:, :kk], 1.0)
                    U[k].append(((ef * m).sum(1) / te.etot[rows]).cpu())
                    g = (co.square() * m).reshape(len(rows), te.C, te.M).sum(-1)
                    V[k].append(((E - g).clamp_min(0) / Dr).sqrt().mean(1).cpu())
        res["oracle"] = {str(k): {"U": float(torch.cat(U[k]).mean()), "vrmse": float(torch.cat(V[k]).mean())} for k in Ks}
        res["modes_total"] = int(te.C * te.M)
    res["qubit_bands"] = a.qubit_bands
    (OUT / f"{a.row}_{a.arm}{sfx}_s{a.seed}.json").write_text(json.dumps(res))
    if __import__("os").environ.get("QW_CKPT"):                  # doc 35 §33: keep the VALID-best weights for diagnostics
        ck = pathlib.Path(__import__("os").environ["QW_CKPT"]); ck.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in __import__("os").environ.items() if k.startswith("QW_") and k not in ("QW_CKPT", "QW_OUT", "QW_GPU_GB")}
        torch.save({"model": state, "gate": gstate, "route": route.cpu(), "args": vars(a), "K": K, "dyn": DYN_TETHER, "env": env},
                   ck / f"{a.row}_{a.arm}{sfx}_s{a.seed}.pt")
    print(f"[adrc] {a.row} {a.arm}{sfx.replace('_', ' +')} s{a.seed}: TEST {base:.5f} -> {tm:.5f} ({100 * res['test_rel_gain']:+.2f}%) captured {tc:.4f} "
          f"tethers {res['tethers']} | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("row"); ap.add_argument("--arm", default="adrc_ryd")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--route_sup", action="store_true", help="router supervised toward the per-window top-K error energy")
    ap.add_argument("--qubit_bands", default=None, help="comma-separated band indices allowed for qubit routing, e.g. 1,2")
    ap.add_argument("--shots", default=None, help="eval-only finite-shot readout, e.g. 100,1000,10000 (doc 35 §11)")
    ap.add_argument("--noise", action="store_true", help="eval-only calibration noise: position jitter, detuning (doc 35 §12)")
    ap.add_argument("--atoms", type=int, default=30, choices=(5, 10, 20, 30, 50, 100, 200), help="qubit budget K (doc 35 §13; 50: §20)")
    ap.add_argument("--dyn_tether", action="store_true", help="tether strengths conditioned on each input (doc 35 §13)")
    ap.add_argument("--fgn", action="store_true", help="noise-conditioned configuration, fair energy score (doc 35 §17)")
    ap.add_argument("--dress", action="store_true", help="eval-only: VALID-tuned Gaussian dressing of the deterministic hybrid (doc 35 §17 control)")
    ap.add_argument("--fgn_tau", action="store_true", help="with --fgn: VALID-tuned spread scale about the ensemble mean (doc 35 §17 d)")
    ap.add_argument("--dress_rel", action="store_true", help="as --dress with noise scaled per window by the causal std of x_t (variant e)")
    ap.add_argument("--credit", action="store_true", help="eval-only: Shapley credit + handoff under hardware noise (doc 35 §28c)")
    ap.add_argument("--dress_val", action="store_true", help="as --dress with the per-mode sigma from VALID residuals (variant c)")
    _a = ap.parse_args()
    K, NG, G, DYN_TETHER, FGN = _a.atoms, min(10, _a.atoms), max(1, _a.atoms // 10), _a.dyn_tether, _a.fgn
    run(_a)
