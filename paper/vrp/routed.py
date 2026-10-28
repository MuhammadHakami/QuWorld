"""Routed node register for RouteFinder (doc 35 §16): the input-dependent counterpart of the pooled bottleneck.

The pooled bottleneck (paper/model.py: Bottleneck) averages node embeddings and broadcasts one correction to every
node, so it reads and writes only the DC mode over nodes (Proposition 1). Here, per instance:
  - a learned scorer routes K nodes (straight-through top-K) to the atoms;
  - atom positions are those nodes' own coordinates (a VRP instance is a 2-D point set), scaled to lattice units, plus a
    bounded learned offset in x, y, z from each node's embedding and the instance context;
  - K atoms form groups of <= 10 (blockade-decoupled registers) joined by feed-forward tethers whose strengths are set
    per instance;
  - each atom's readout (<Z>, <X>, <Y>) is written back to its own node through a zero-initialised map, so the model
    starts exactly at the pretrained checkpoint.
Control: the same routing design and tethers, trained separately, with an MLP core sized from the register (`-mlp`);
GroupMLP ignores the coordinates it is passed, so the control is graph-blind.
`-fgn` (doc 35 §18, after FGN / WeatherNext 3): a code xi ~ N(0, I_32) per instance row perturbs the configuration (FiLM of
the atoms' features, the offset network's hidden layer, the tether logits; all zero-init). Every row draws its own code,
in training and at test, so the 8 symmetry-augmented copies RouteFinder decodes per test instance get 8 configurations.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "alloc"))
from cores import GroupMLP, TileRegister, n_params  # noqa: E402
from fgn import ZD, NoiseFiLM, zero_linear  # noqa: E402


class RoutedNodeRegister(nn.Module):
    def __init__(self, H: int, K: int, core: str = "ryd", fgn: bool = False):
        super().__init__()
        self.K, self.NG = K, min(10, K); self.G = max(1, K // self.NG)
        self.scorer = nn.Linear(H, 1)
        self.hyp = nn.Sequential(nn.Linear(2 * H, 32), nn.GELU(), nn.Linear(32, 3)); nn.init.zeros_(self.hyp[-1].weight); nn.init.zeros_(self.hyp[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.2 * self.NG ** 0.5)))))   # unit square -> ~1.2 spacing
        self.tether_logit = nn.Parameter(torch.full((3,), -2.0)); self.tether_hyp = nn.Linear(H, 3)
        nn.init.zeros_(self.tether_hyp.weight); nn.init.zeros_(self.tether_hyp.bias)
        self.tether_w = nn.Linear(3, H); nn.init.zeros_(self.tether_w.weight); nn.init.zeros_(self.tether_w.bias)
        reg = TileRegister(H, self.NG, 4, "ryd"); reg.log_lam.data.zero_(); reg.log_lam.requires_grad_(False)
        self.core = reg if core == "ryd" else GroupMLP(H, n_params(reg))
        self.up = nn.Linear(3, H); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.fgn = fgn
        if fgn:                                                                         # doc 35 §18
            self.film, self.hyp_xi, self.tether_xi = NoiseFiLM(H), zero_linear(ZD, 32), zero_linear(ZD, 3)

    def forward(self, h: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
        B, N, H = h.shape
        sc = self.scorer(h).squeeze(-1)                                             # (B, N)
        idx = sc.topk(self.K, 1).indices
        kth = sc.topk(min(self.K + 1, N), 1).values[:, -1:]
        soft = torch.sigmoid(sc.gather(1, idx) - kth); mask = (1 + soft - soft.detach())[..., None]    # straight-through
        hk = h.gather(1, idx[..., None].expand(-1, -1, H)); xy = locs[..., :2].gather(1, idx[..., None].expand(-1, -1, 2))
        ctx = h.mean(1)
        noise = torch.randn(B, ZD, device=h.device) if self.fgn else None                # one code per instance row
        logit = self.tether_logit + self.tether_hyp(ctx) + (self.tether_xi(noise) if self.fgn else 0)
        t = (torch.sigmoid(logit) * 1.2 - 0.1).clamp(0, 1)                                              # (B, 3) per instance
        outs, ro = [], []
        for g in range(self.G):
            sl = slice(g * self.NG, (g + 1) * self.NG)
            f, p = hk[:, sl], xy[:, sl]
            if g > 0:
                tin = t[:, g - 1:g] * ro[g - 1] + (t[:, 2:3] * ro[0] if g == 2 else 0)
                f = f + self.tether_w(tin)[:, None]
            hid = self.hyp[0](torch.cat([f, ctx[:, None].expand_as(f)], -1))
            if self.fgn:
                hid = hid + self.hyp_xi(noise)[:, None]
            off = 0.45 * torch.tanh(self.hyp[2](self.hyp[1](hid)))                                      # (B, n, 3)
            pos = torch.cat([p * self.log_scale.exp(), torch.zeros_like(p[..., :1])], -1) + off
            coords = torch.cat([pos, torch.zeros_like(pos[..., :1])], -1)                               # (B, n, 4)
            o = self.core(self.film(f, noise) if self.fgn else f, coords)                               # (B, n, 3)
            ro.append(o.mean(1)); outs.append(o)
        corr = self.up(torch.cat(outs, 1)) * mask                                                       # (B, K, H)
        return h.scatter_add(1, idx[..., None].expand(-1, -1, H), corr)
