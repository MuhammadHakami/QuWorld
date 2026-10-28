# Fusion Equilibrium Challenge (NeurIPS 2026)

> **Forum:** https://www.codabench.org/forums/17429/
> A Codabench platform bug (since 29 September) hides the Forum tab from participant accounts. The
> forum still works at this link; announcements and Q&A are posted there.

Tokamak control needs the plasma's magnetic equilibrium — the 2-D poloidal flux map `ψ(R,Z)`,
its last-closed-flux-surface (LCFS) boundary, and a handful of scalar parameters — in real time.
Traditional reconstructions (EFIT) lean on magnetic sensors that drift, fail, and are absent on
next-step machines. **This challenge asks you to reconstruct the equilibrium from non-magnetic
diagnostics alone**: coil currents and Thomson-scattering profiles.

You get labeled data from **DIII-D** (a conventional tokamak). You are then asked to generalize
**zero-shot to MAST** (a spherical tokamak) — a machine with a different geometry, different coil
set, and **no training data at all**.

## Two challenges, two awards

There are two challenges, and each grants exactly one award — the mapping is **one to one**. They
share one dataset, one submission format, and one leaderboard: the two challenges are two sortable
**columns** on that board. Enter either or both.

| | Challenge 1 — Intra-machine (DIII-D) | Challenge 2 — Cross-machine (DIII-D → MAST) |
|---|---|---|
| **What you do** | Reconstruct the equilibrium on the machine you trained on. | Take that model to a machine it has never seen — MAST has **no training data at all**. |
| **Scored by** | `S_model` on the hidden DIII-D test set | `G_ratio = S_MAST / S_DIII-D` (entries with DIII-D composite `S ≥ 0.85`) |
| **Leaderboard column** | `Ch1: DIII-D S` (default sort) | `Ch2: G_ratio` |
| **Award** | **Award #1** | **Award #2** |

**Challenge 2 needs both machines.** `G_ratio` compares your MAST score against your DIII-D score,
so a DIII-D-only entry scores `G_ratio = 0`. You can attempt Challenge 1 on its own; Challenge 2
always includes Challenge 1.

Honorable mentions recognize the best flux map, best predicted scalars (`q95`/`βN`), best
boundary, and best flux-map consistency, each judged on the MAST blind fold. A **compute-light badge** recognizes models that train
end-to-end in under 2 h on a single GPU *or CPU* — it applies across **both** challenges and is not
a separate track or leaderboard.

## Phases

| Phase | Window | Submissions |
|---|---|---|
| **Development** — public leaderboard, iterate freely | **Jul 27 – Oct 18, 2026** | 5/day, 100 total |
| **Final** — blind private fold, decides the winners | **Oct 19 – Oct 26, 2026** | 3 total; leaderboard hidden until close |

## How it works

1. Register your team on Codabench (one contact email per team).
2. Pull the data from Hugging Face: **`Sophelio/fusion-equilibrium-challenge`**.
3. Train on DIII-D; predict on the public test fold for both machines.
4. Submit predictions (one `.npz` per machine config — see **Submission format**). The scorer runs
   automatically and updates the leaderboard.

See **Data**, **Evaluation**, **Rules**, and **Submission format** for details.

**Starter kit** — data loader, a runnable submission skeleton, a local validator, and baselines:
[`github.com/Sophelio/fusion-equilibrium-challenge-starter`](https://github.com/Sophelio/fusion-equilibrium-challenge-starter).
Start there; it produces a correctly formatted submission on the first run.