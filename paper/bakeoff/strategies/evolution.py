"""Arm (b): evolutionary population-of-registers with a QD archive.

pyribs CMA-MAE (GridArchive 20x20, learning_rate 0.01) + 3 x
EvolutionStrategyEmitter (ranker='imp', sigma0 = 0.05*FOV), tier-0
weight-inheritance fitness only: write the genome into the TARGET bank slot
(the bank most routed on the segment's first batch), keep every inherited
weight, score = forward VRMSE on a fixed val slice (1 fwd = 1 eval-unit).
Tier-1 elite sprints need the GPU campaign cadence and are NOT run here (a
hook point is documented in the final report).

Genome = one bank slot: (R x n x 2 bounded atom coords, router row logits
[width weights + bias]). Decode passes coords through the sealed repair
(5 um floor / FOV clamp) — the archive stores raw genomes, decode is
deterministic, so no ask/tell mismatch.

Behavior descriptors (computed decode-side, model-eval-free):
  (1) blockade-graph edge density: fraction of within-register pairs with
      r_ij < R_b (=1.0 sim units), averaged over registers;
  (2) interaction-spectrum entropy: Shannon entropy of the normalized |eig|
      of J_ij = 1/r_ij^6, averaged over registers, / log(n).

Operator portfolio (>= 5 distinct operators, applied outside the CMA ask/tell
loop with archive.add, lineage JSONL logged: id, parent, operator, fitness,
add status): per-atom Gaussian jitter, pairwise atom swap, sub-register block
crossover, Gram-spectrum mutation, lattice re-seed, register->register atom
transfer, router-row jitter. Parent sampling DGM-style:
p ~ softmax(objective/tau) / (1 + children); never discard. Novelty gate:
skip evaluation when the mutant lands in an occupied cell AND its genome
cosine-similarity to that cell's elite is > 0.99.

Kill-rule fields: elite_gain_rel (vs weight-inheritance parent),
archive coverage.
"""

import json

import numpy as np
import torch

from _bootstrap import add_paths

add_paths()

from data import vrmse  # noqa: E402
from ribs.archives import GridArchive  # noqa: E402
from ribs.emitters import EvolutionStrategyEmitter  # noqa: E402
from ribs.schedulers import Scheduler  # noqa: E402
from strategies.base import EvalMeter, Strategy  # noqa: E402
from substrate import repair_positions, set_adaptable  # noqa: E402

R_BLOCKADE = 1.0  # blockade radius in sim units (grid spacing)
FIT_CEIL = 5.0    # objective = FIT_CEIL/(FIT_CEIL+vrmse): (0,1], monotone
#                   decreasing in vrmse, strictly positive so CMA-MAE's
#                   threshold_min=0 archive always accepts the parent
OPERATORS = ("jitter", "atom_swap", "block_crossover", "gram_spectrum",
             "lattice_reseed", "atom_transfer", "router_jitter")


def descriptors(coords: torch.Tensor) -> tuple[float, float]:
    """coords (R, n, 2) repaired -> (edge_density, spectrum_entropy)."""
    r, n = coords.shape[0], coords.shape[1]
    dens, ents = [], []
    iu = torch.triu_indices(n, n, offset=1)
    for k in range(r):
        d = torch.cdist(coords[k], coords[k]).clamp(min=1e-6)
        dens.append((d[iu[0], iu[1]] < R_BLOCKADE).float().mean().item())
        j = d.pow(-6) * (1 - torch.eye(n))
        ev = torch.linalg.eigvalsh(j).abs()
        p = ev / ev.sum().clamp(min=1e-12)
        ent = -(p * (p + 1e-12).log()).sum() / np.log(n)
        ents.append(float(ent.clamp(0.0, 1.0)))
    return float(np.mean(dens)), float(np.mean(ents))


class EvolutionQDStrategy(Strategy):
    name = "evolution"
    seed_offset = 20
    groups = ("geometry", "router")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rng = np.random.default_rng(7919 * self.seed + self.seed_offset)
        self.parent_vrmse = float("nan")
        self.best_vrmse = float("nan")
        self.coverage = 0.0
        self.n_evals = 0
        self.op_counts = {op: 0 for op in OPERATORS}
        self.elites: dict = {}   # cell -> dict(genome, vrmse, children)
        self._id = 0

    # ---------------------------------------------------------- genome codec
    def _layout(self):
        kmix = self.model.mid.core
        bank = kmix.banks[self.k_target]
        r, n = len(bank.regs), bank.regs[0].n
        w = kmix.router.weight.shape[1]
        return r, n, w

    def _encode(self) -> np.ndarray:
        kmix = self.model.mid.core
        bank = kmix.banks[self.k_target]
        coords = torch.stack([reg.pos.detach() for reg in bank.regs])
        row = torch.cat([kmix.router.weight.detach()[self.k_target],
                         kmix.router.bias.detach()[self.k_target:
                                                   self.k_target + 1]])
        return torch.cat([coords.reshape(-1), row]).numpy().astype(np.float64)

    def _decode_coords(self, genome: np.ndarray) -> torch.Tensor:
        r, n, _ = self._layout()
        raw = torch.from_numpy(np.asarray(genome[:r * n * 2])).float()
        return torch.stack([repair_positions(c)
                            for c in raw.reshape(r, n, 2)])

    def _write(self, genome: np.ndarray, coords: torch.Tensor) -> None:
        r, n, w = self._layout()
        kmix = self.model.mid.core
        bank = kmix.banks[self.k_target]
        row = torch.from_numpy(np.asarray(genome[r * n * 2:])).float()
        with torch.no_grad():
            for i, reg in enumerate(bank.regs):
                reg.pos.copy_(coords[i])
            kmix.router.weight[self.k_target] = row[:w]
            kmix.router.bias[self.k_target] = row[w]

    # ------------------------------------------------------------- fitness
    def _evaluate(self, genome: np.ndarray) -> tuple[float, float,
                                                     tuple[float, float]]:
        coords = self._decode_coords(genome)
        meas = descriptors(coords)
        self._write(genome, coords)
        with torch.no_grad():
            v = vrmse(self.model(self._xv), self._yv).item()
        self.n_evals += 1
        obj = FIT_CEIL / (FIT_CEIL + v) if np.isfinite(v) and v >= 0 else 1e-9
        return obj, v, meas

    @staticmethod
    def _cell(meas) -> tuple[int, int]:
        return (min(19, int(meas[0] * 20)), min(19, int(meas[1] * 20)))

    def _record_elite(self, genome, v, meas) -> None:
        cell = self._cell(meas)
        cur = self.elites.get(cell)
        if cur is None or v < cur["vrmse"]:
            children = cur["children"] if cur else 0
            self.elites[cell] = {"genome": np.array(genome), "vrmse": v,
                                 "children": children}

    # ------------------------------------------------------------ operators
    def _pick_parent(self) -> tuple[tuple, dict]:
        cells = list(self.elites)
        objs = np.array([FIT_CEIL / (FIT_CEIL + self.elites[c]["vrmse"])
                         for c in cells])
        kids = np.array([self.elites[c]["children"] for c in cells])
        logits = objs / 0.5
        p = np.exp(logits - logits.max()) / (1.0 + kids)
        p = p / p.sum()
        cell = cells[self.rng.choice(len(cells), p=p)]
        return cell, self.elites[cell]

    def _mutate(self, genome: np.ndarray, op: str,
                parent2: np.ndarray) -> np.ndarray:
        r, n, w = self._layout()
        g = np.array(genome)
        nc = r * n * 2
        coords = g[:nc].reshape(r, n, 2)
        rng = self.rng
        if op == "jitter":
            coords += rng.normal(0.0, self.cfg.sigma0, coords.shape)
        elif op == "atom_swap":
            k = rng.integers(r)
            i, j = rng.choice(n, size=2, replace=False)
            coords[k, [i, j]] = coords[k, [j, i]]
        elif op == "block_crossover":
            k = rng.integers(r)
            coords[k] = parent2[:nc].reshape(r, n, 2)[k]
        elif op == "gram_spectrum":
            k = rng.integers(r)
            p = coords[k] - coords[k].mean(0)
            gram = p @ p.T
            vals, vecs = np.linalg.eigh(gram)
            vals = vals * (1.0 + 0.1 * rng.normal(size=vals.shape))
            top = np.clip(vals[-2:], 0.0, None)
            coords[k] = (vecs[:, -2:] * np.sqrt(top)) + coords[k].mean(0)
        elif op == "lattice_reseed":
            k = rng.integers(r)
            side = int(np.ceil(np.sqrt(n)))
            grid = np.stack(np.meshgrid(np.arange(side), np.arange(side),
                                        indexing="ij"), -1
                            ).reshape(-1, 2)[:n].astype(np.float64)
            coords[k] = grid + 0.05 * rng.normal(size=(n, 2))
        elif op == "atom_transfer":
            ka, kb = rng.choice(r, size=2, replace=(r < 2))
            i = rng.integers(n)
            coords[kb, i] = coords[ka, i]
        elif op == "router_jitter":
            g[nc:] += 0.1 * rng.normal(size=w + 1)
        else:
            raise ValueError(f"unknown operator {op!r}")
        g[:nc] = coords.reshape(-1)
        return g

    # --------------------------------------------------------------- adapt
    def adapt(self, segment, meter: EvalMeter, probe) -> None:
        cfg = self.cfg
        set_adaptable(self.model, ())
        model, kmix = self.model, self.model.mid.core
        lineage_path = None
        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)
            lineage_path = (self.outdir /
                            f"lineage_{segment.name}_s{self.seed}.jsonl")

        def log_lineage(row: dict) -> None:
            if lineage_path is not None:
                with open(lineage_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")

        # target bank = most-routed bank on the segment's first batch
        x0_batch, _ = next(segment.batches(cfg.batch, 1))
        if not meter.try_charge(fwd=1):
            return
        with torch.no_grad():
            h = model.pre(model.lift(x0_batch.to(self.device))).mean((2, 3))
            idx = kmix.router(h).argmax(dim=-1)
        self.k_target = int(torch.bincount(idx, minlength=len(kmix.banks)
                                           ).argmax())
        self.samples_seen += x0_batch.shape[0]

        self._xv, self._yv = segment.val_slice(cfg.batch)
        self._xv = self._xv.to(self.device)
        self._yv = self._yv.to(self.device)

        x0 = self._encode()
        dim = x0.size
        kw = dict(solution_dim=dim, dims=(20, 20),
                  ranges=[(0.0, 1.0), (0.0, 1.0)], seed=self.seed)
        archive = GridArchive(learning_rate=0.01, threshold_min=0.0, **kw)
        result = GridArchive(**kw)
        emitters = [EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=cfg.sigma0, ranker="imp",
            batch_size=cfg.emitter_batch, seed=self.seed + 100 + i)
            for i in range(cfg.n_emitters)]
        sched = Scheduler(archive, emitters, result_archive=result)

        if not meter.try_charge(fwd=1):
            return
        obj0, self.parent_vrmse, meas0 = self._evaluate(x0)
        archive.add(x0[None], [obj0], [meas0])
        result.add(x0[None], [obj0], [meas0])
        self._record_elite(x0, self.parent_vrmse, meas0)
        log_lineage({"id": 0, "parent": None, "operator": "inherit",
                     "vrmse": self.parent_vrmse, "measures": meas0,
                     "gen": -1})

        best = {"genome": x0, "vrmse": self.parent_vrmse}
        gen = 0
        n_ask = cfg.n_emitters * cfg.emitter_batch
        while meter.remaining >= meter.cost(fwd=n_ask):
            sols = sched.ask()
            objs, meass = [], []
            for s in sols:
                meter.try_charge(fwd=1)  # affordability pre-checked above
                obj, v, meas = self._evaluate(s)
                objs.append(obj)
                meass.append(meas)
                self._record_elite(s, v, meas)
                if v < best["vrmse"]:
                    best = {"genome": np.array(s), "vrmse": v}
                self._id += 1
                log_lineage({"id": self._id, "parent": "cma", "operator":
                             "cma-es", "vrmse": v, "measures": meas,
                             "gen": gen})
            sched.tell(objs, meass)
            self.samples_seen += int(self._xv.shape[0]) * len(sols)

            for _ in range(cfg.op_batch):
                if meter.remaining < meter.cost(fwd=1):
                    break
                pcell, parent = self._pick_parent()
                _, parent2 = (self._pick_parent() if len(self.elites) > 1
                              else (pcell, parent))
                op = OPERATORS[self.rng.integers(len(OPERATORS))]
                child = self._mutate(parent["genome"], op,
                                     parent2["genome"])
                cmeas = descriptors(self._decode_coords(child))
                ccell = self._cell(cmeas)
                occupant = self.elites.get(ccell)
                if occupant is not None:
                    a, b = child, occupant["genome"]
                    cos = float(a @ b / (np.linalg.norm(a)
                                         * np.linalg.norm(b) + 1e-12))
                    if cos > 0.99:
                        self._id += 1
                        log_lineage({"id": self._id, "parent": str(pcell),
                                     "operator": op, "vrmse": None,
                                     "rejected": "novelty_gate", "gen": gen})
                        continue
                meter.try_charge(fwd=1)
                obj, v, meas = self._evaluate(child)
                archive.add(child[None], [obj], [meas])
                result.add(child[None], [obj], [meas])
                self._record_elite(child, v, meas)
                parent["children"] += 1
                self.op_counts[op] += 1
                if v < best["vrmse"]:
                    best = {"genome": np.array(child), "vrmse": v}
                self._id += 1
                log_lineage({"id": self._id, "parent": str(pcell),
                             "operator": op, "vrmse": v, "measures": meas,
                             "gen": gen})
                self.samples_seen += int(self._xv.shape[0])
            gen += 1
            probe(meter.used, self.samples_seen)

        # install the champion (weight-inheritance parent if nothing beat it)
        self._write(best["genome"], self._decode_coords(best["genome"]))
        self.best_vrmse = best["vrmse"]
        self.coverage = float(result.stats.coverage)

    def diagnostics(self) -> dict:
        gain = ((self.parent_vrmse - self.best_vrmse)
                / max(self.parent_vrmse, 1e-9)
                if np.isfinite(self.parent_vrmse) else float("nan"))
        return {"parent_vrmse": self.parent_vrmse,
                "best_vrmse": self.best_vrmse,
                "elite_gain_rel": gain,
                "coverage": self.coverage,
                "n_tier0_evals": self.n_evals,
                "k_target": getattr(self, "k_target", None),
                "op_counts": self.op_counts}
