"""Graph structure of the hardest tiles (TRL2D test, GPU): do the top-K error tiles cluster in space and chain
across scales? Edges: same level/orientation/channel and Chebyshev distance 1 ('spatial'), or parent-child across
adjacent levels, same orientation/channel ('scale'). Compared with K random tiles.

    .venv/bin/python paper/spectral/tile_graph.py  ->  runs/spectral/trl2d_tile_graph.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper")); sys.path.insert(0, str(ROOT / "paper" / "phase_b")); sys.path.insert(0, str(ROOT / "paper" / "spectral"))
import cnext_hybrid  # noqa: E402
import well2d  # noqa: E402
from band_analysis import DEV, J  # noqa: E402
from router_capture import tiles  # noqa: E402

K = 30


def coords(H: int, W: int, C: int) -> torch.Tensor:
    """(n_tiles, 5) = (level, orientation, channel, y, x) in tiles() order."""
    out = []
    for j in range(J):
        h, w = H >> (j + 1), W >> (j + 1)
        for o in range(3):
            c, y, x = torch.meshgrid(torch.arange(C), torch.arange(h), torch.arange(w), indexing="ij")
            out.append(torch.stack([torch.full_like(c, j), torch.full_like(c, o), c, y, x], -1).reshape(-1, 5))
    return torch.cat(out).to(DEV)


def graph_stats(sel: torch.Tensor) -> tuple[float, float, float]:
    """sel (K, 5) -> (spatial edges per node, scale edges per node, fraction of nodes in the largest component)."""
    l, o, c, y, x = sel.unbind(1)
    same = (o[:, None] == o[None]) & (c[:, None] == c[None])
    spatial = same & (l[:, None] == l[None]) & ((y[:, None] - y[None]).abs() <= 1) & ((x[:, None] - x[None]).abs() <= 1)
    spatial.fill_diagonal_(False)
    child_of = same & (l[:, None] == l[None] - 1) & (y[:, None] // 2 == y[None]) & (x[:, None] // 2 == x[None])
    scale = child_of | child_of.T
    adj = spatial | scale
    comp = torch.arange(len(sel), device=DEV)
    for _ in range(len(sel)):                           # label propagation to connected components
        comp = torch.minimum(comp, torch.where(adj, comp[None].expand_as(adj), comp[:, None].expand_as(adj)).min(1).values)
    largest = torch.bincount(comp).max().item() / len(sel)
    return spatial.sum().item() / len(sel), scale.sum().item() / len(sel), largest


def main() -> None:
    data = well2d.Split("turbulent_radiative_layer_2D", "test"); C = data.C
    xyz = coords(data.H, data.W, C)
    model = cnext_hybrid.build_hybrid("vanilla").to(DEV).eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    acc = {r: [] for r in ("oracle", "trunk_change", "random")}
    with torch.no_grad():
        for x, y in well2d.batched(data, well2d.fixed_windows(data, 20), 16):
            x, y = x.to(DEV), y.to(DEV); p = model(x)
            score = {"oracle": tiles(p - y), "trunk_change": tiles(p - x[:, (well2d.N_HIST - 1) * C:])}
            score["random"] = torch.rand(score["oracle"].shape, generator=gen, device=DEV)
            for r, s in score.items():
                for b in range(s.shape[0]):
                    acc[r].append(graph_stats(xyz[s[b].topk(K).indices]))
    out = {r: dict(zip(("spatial_deg", "scale_deg", "largest_component"), torch.tensor(v).mean(0).tolist())) for r, v in acc.items()}
    out["K"] = K; out["windows"] = len(acc["oracle"])
    (ROOT / "runs" / "spectral" / "trl2d_tile_graph.json").write_text(json.dumps(out, indent=1))
    for r in ("oracle", "trunk_change", "random"):
        print(f"{r:12s} spatial edges/node {out[r]['spatial_deg']:.2f}  scale edges/node {out[r]['scale_deg']:.2f}  largest component {out[r]['largest_component']:.2f}")
    data.close()


if __name__ == "__main__":
    main()
