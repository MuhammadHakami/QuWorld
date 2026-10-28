"""Build and EXECUTE the three analysis notebooks (CPU-only cells — the GPU
belongs to the sprint queue).

  notebooks/01_task_analysis.ipynb       — what the benchmark is and why it's hard
  notebooks/02_baselines_evaluation.ipynb — every baseline number, ours + published
  notebooks/03_our_solution.ipynb         — architecture, mechanism, live hypothesis
                                            verdicts, and the continue-here lab cells

Idempotent: rebuilds + re-executes all three. Re-run after each round.
"""

import pathlib

import nbformat as nbf
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
NBDIR = ROOT / "notebooks"

PRELUDE = """\
import sys, pathlib, glob, re, warnings
warnings.filterwarnings('ignore')
ROOT = pathlib.Path().resolve().parent
sys.path.insert(0, str(ROOT / 'paper'))
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 90
DEVICE = 'cpu'   # GPU reserved for the sprint queue; flip when free
torch.manual_seed(0)
"""

PARSE = """\
from log_notebook import parse_log
def results_frame():
    rows = []
    for f in sorted(glob.glob(str(ROOT / 'runs' / 'round*_*.log'))):
        name = pathlib.Path(f).stem
        rnd, rest = name.split('_', 1)
        m = re.match(r'(.+?)_s(\\d+)$', rest)
        variant, seed = (m.group(1), int(m.group(2))) if m else (rest, 1337)
        d = parse_log(f)
        if d['crashed']:
            continue
        rows.append(dict(round=rnd, variant=variant, seed=seed,
                         v1=float(d['val_vrmse_1']), v8=float(d['val_vrmse_8']),
                         steps=int(d['num_steps']), params=int(d['num_params'])))
    return pd.DataFrame(rows)
"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def nb_task():
    cells = [
        md("# 01 — Task Analysis: The Well `turbulent_radiative_layer_2D`\n\n"
           "**Benchmark:** The Well (arXiv:2412.00568, NeurIPS D&B). Radiatively "
           "cooled turbulent mixing layers — cold dense gas below, hot dilute gas "
           "above; the shear interface develops chaotic mixing whose evolution "
           "depends on cooling time `t_cool` (9 regimes). Fields: density, "
           "pressure, velocity (x,y). Official metric **VRMSE** (1.0 = as good as "
           "predicting the spatial mean).\n\n"
           "**Our subset:** 3 regimes spanning the range (t_cool 0.03 / 0.56 / "
           "3.16), train+valid, downsampled 2x to 64x192 for the sprint loop."),
        code(PRELUDE),
        code("from data import load, vrmse\n"
             "train, valid, mean, std = load()\n"
             "print('train', tuple(train.shape), '(traj, T, C, H, W)')\n"
             "print('valid', tuple(valid.shape))\n"
             "print('channels: density, pressure, vx, vy (z-scored)')"),
        md("## What the fields look like\nEarly, middle and late snapshots of one "
           "trajectory — the mixing layer develops fine chaotic structure over time."),
        code("fig, ax = plt.subplots(4, 3, figsize=(9, 9))\n"
             "names = ['density', 'pressure', 'vx', 'vy']\n"
             "for c in range(4):\n"
             "    for j, t in enumerate([0, 50, 100]):\n"
             "        ax[c, j].imshow(train[0, t, c].T, cmap='RdBu_r', aspect='auto')\n"
             "        ax[c, j].set_xticks([]); ax[c, j].set_yticks([])\n"
             "        if c == 0: ax[c, j].set_title(f't={t}')\n"
             "        if j == 0: ax[c, j].set_ylabel(names[c])\n"
             "plt.suptitle('trajectory 0 (t_cool=0.03)'); plt.tight_layout(); plt.show()"),
        md("## Why it is hard: growing temporal change + chaotic mixing"),
        code("d = (train[:, 1:] - train[:, :-1]).abs().mean(dim=(0, 2, 3, 4))\n"
             "plt.figure(figsize=(7, 3))\n"
             "plt.plot(d.numpy()); plt.xlabel('timestep'); plt.ylabel('mean |x(t+1)-x(t)|')\n"
             "plt.title('per-step change grows as mixing develops'); plt.tight_layout(); plt.show()"),
        code("# energy spectrum flattens as turbulence develops fine scales\n"
             "def spec(v):\n"
             "    f = torch.fft.rfft(v, dim=-1).abs().square().mean(dim=(0, 1))\n"
             "    return f.mean(dim=0)\n"
             "s0 = spec(train[:, 5, 2:4]); s1 = spec(train[:, 95, 2:4])\n"
             "plt.figure(figsize=(6, 3.5))\n"
             "plt.loglog(s0.numpy(), label='t=5'); plt.loglog(s1.numpy(), label='t=95')\n"
             "plt.xlabel('wavenumber'); plt.ylabel('velocity power'); plt.legend()\n"
             "plt.title('spectral broadening = fine-scale turbulence'); plt.tight_layout(); plt.show()"),
        md("## Trivial floors every model must beat"),
        code("pers1, pers8 = [], []\n"
             "for t in range(0, valid.shape[1] - 9, 8):\n"
             "    x = valid[:, t]\n"
             "    pers1.append(vrmse(x, valid[:, t + 1]).item())\n"
             "    pers8.append(vrmse(x, valid[:, t + 8]).item())\n"
             "print(f'persistence  VRMSE@1: {np.mean(pers1):.3f}   VRMSE@8: {np.mean(pers8):.3f}')\n"
             "print('mean-predictor VRMSE = 1.0 by construction')"),
        md("## Published numbers on the FULL official protocol (context)\n\n"
           "| model | one-step TEST | rollout 6:12 | rollout 13:30 |\n|---|---|---|---|\n"
           "| CNextU-net ~18M | **0.1956** | 0.54 | 1.01 |\n"
           "| U-Net classic | 0.2418 | 0.66 | 1.04 |\n"
           "| FNO | 0.5001 | 1.79 | 3.54 |\n"
           "| TFNO | 0.5016 | 6.01 | >10 |\n"
           "| Walrus 1.3B (median protocol) | 0.0831 | — | 0.339 @T[1:20] |\n\n"
           "**Takeaways.** (1) Even 1.3B-param foundation models exceed VRMSE 1.0 at "
           "long horizons — rollout stability is the open flank. (2) Our sprint "
           "numbers use an easier internal protocol (1-frame input, downsampled, "
           "valid split) — comparable to each other, not to this table, until the "
           "protocol-faithful trainer (scoreboard H1d) runs."),
    ]
    return cells


def nb_baselines():
    cells = [
        md("# 02 — Baselines Evaluation\n\nEvery run in the loop so far, parsed "
           "live from `runs/*.log`. Sprint protocol: fixed seed per round, 5-min "
           "wall-clock (or fixed steps where marked `round2s`), parameter-matched "
           "~1.33M, identical data/LR/batch."),
        code(PRELUDE),
        code(PARSE + "df = results_frame()\n"
             "df.sort_values(['round','v8']).reset_index(drop=True)"),
        md("## Round 1 — 5-minute wall-clock (throughput included)"),
        code("r1 = df[df['round'] == 'round1'].set_index('variant')\n"
             "fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))\n"
             "r1['v1'].plot.bar(ax=ax[0], title='VRMSE@1 (round 1)')\n"
             "r1['v8'].plot.bar(ax=ax[1], title='VRMSE@8 rollout (round 1)', color='tab:orange')\n"
             "for a in ax: a.axhline(1.0, ls='--', c='gray', lw=.8)\n"
             "plt.tight_layout(); plt.show()\n"
             "print(r1[['v1','v8','steps']].to_string())"),
        md("**Read:** `mlp` (matched classical control) is *worse* than no "
           "bottleneck at all — the round-1 quantum win (hea12, -23% rollout) is "
           "not a 'any bottleneck helps' artifact."),
        md("## Round 2s — fixed 2,500 steps (throughput removed)"),
        code("r2s = df[df['round'] == 'round2s'].set_index('variant')\n"
             "ax = r2s['v8'].plot.bar(figsize=(5, 3), color='tab:green',\n"
             "                        title='VRMSE@8 at exactly 2500 steps')\n"
             "ax.axhline(1.0, ls='--', c='gray', lw=.8)\n"
             "plt.tight_layout(); plt.show()\n"
             "base = r2s.loc['none', 'v8']\n"
             "for v in r2s.index:\n"
             "    print(f'{v:8s} v8={r2s.loc[v,\"v8\"]:.3f}  vs none: {100*(r2s.loc[v,\"v8\"]/base-1):+.1f}%')"),
        md("**Read:** at equal gradient steps the 12q block cuts rollout error "
           "~-37% — the quantum effect is per-step real, not a speed artifact. "
           "hea16 beats baseline but loses to hea12: digital-ansatz qubit scaling "
           "is NOT monotone. (The analog ansatz scaled the other way: ryd12 1.168 "
           "-> ryd16 0.913.)"),
        md("## Run-to-run variance (honesty check)"),
        code("h1 = df.query('round==\"round1\" and variant==\"hea12\"')['v8'].iloc[0]\n"
             "h2 = df.query('round==\"round2\" and variant==\"hea12f\"')['v8'].iloc[0]\n"
             "print(f'hea12 round1 {h1:.3f} vs math-equivalent fused rerun {h2:.3f}')\n"
             "print(f'=> single-run sigma on v8 is roughly {abs(h1-h2):.2f} — single-seed')\n"
             "print('   wins are NOT bankable; round 3 runs 3 seeds per variant.')"),
        md("## Full ledger"),
        code("df.sort_values(['round', 'variant', 'seed']).reset_index(drop=True)"),
    ]
    return cells


def nb_solution():
    cells = [
        md("# 03 — Our Solution: 30 Simulated Qubits in a World-Model Setup\n\n"
           "**System:** classical trunk (sprint loop: small FNO; headline: frozen "
           "Ant Group **LingBot-World v2 14B** causal-fast, weights on disk) + a "
           "quantum bottleneck whose analog Hamiltonian **adapts to the simulation "
           "state**: pooled latent -> per-qubit detunings + trainable 2D atom "
           "geometry -> Rydberg evolution -> <Z> readout -> zero-init gated "
           "residual.\n\n**30 simulated qubits** = 3 independent 10-qubit "
           "registers (blockade-decoupled patches of one 30-atom array — "
           "hardware-honest, trainable today). Ladder: entangled 30q via MPS-chi "
           "training next; exact 30q statevector is inference-only (8.6 GB).\n\n"
           "Hypotheses under test (see 08-scoreboard.md): **H1** analog quantum "
           "blocks improve physical simulation; **H2** qubits improve large "
           "multimodal world models."),
        code(PRELUDE),
        md("## Mechanism, visibly\nThe trainable atom geometry and the blockade "
           "interaction it induces — this is the object that would compile onto "
           "neutral-atom hardware (Pulser export)."),
        code("from qlayer import QuantumBlock, MultiRegister\n"
             "qb = QuantumBlock(d_in=24, n=10, layers=4, ansatz='rydberg')\n"
             "print('one register params:', sum(p.numel() for p in qb.parameters()))\n"
             "mr = MultiRegister(24, 10, 3, 4, 'rydberg')\n"
             "print('3x10q (30 simulated qubits) params:', sum(p.numel() for p in mr.parameters()))\n"
             "fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))\n"
             "pos = qb.pos.detach().numpy()\n"
             "ax[0].scatter(pos[:, 0], pos[:, 1], s=120, c='tab:red')\n"
             "for i, p in enumerate(pos): ax[0].annotate(str(i), p, fontsize=8)\n"
             "ax[0].set_title('trainable atom geometry (register of 10)')\n"
             "eng = qb._blockade_energy().detach().numpy()\n"
             "ax[1].hist(eng, bins=40, color='tab:purple')\n"
             "ax[1].set_title('blockade energy over 1024 basis states')\n"
             "plt.tight_layout(); plt.show()"),
        md("## The block adapts to the input state\nTwo different pooled latents "
           "produce different detuning patterns, hence different analog evolutions "
           "and readouts — input-conditioned analog simulation in one figure."),
        code("h_a = torch.randn(1, 24); h_b = torch.randn(1, 24)\n"
             "za = qb(h_a).detach().numpy()[0]; zb = qb(h_b).detach().numpy()[0]\n"
             "x = np.arange(10)\n"
             "plt.figure(figsize=(7, 3))\n"
             "plt.bar(x - .2, za, .4, label='latent A'); plt.bar(x + .2, zb, .4, label='latent B')\n"
             "plt.xlabel('qubit'); plt.ylabel('<Z_i> readout'); plt.legend()\n"
             "plt.title('same circuit, different simulation states -> different physics')\n"
             "plt.tight_layout(); plt.show()"),
        md("## Live hypothesis verdicts (re-run this section after every round)"),
        code(PARSE + "df = results_frame()\n"
             "r3 = df[df['round'] == 'round3']\n"
             "if len(r3) == 0:\n"
             "    print('round 3 still running — showing rounds 1-2 evidence')\n"
             "else:\n"
             "    piv = r3.pivot_table(index='variant', columns='seed', values='v8')\n"
             "    piv['mean'] = piv.mean(axis=1); piv['std'] = piv.std(axis=1)\n"
             "    print(piv.round(3).to_string())\n"
             "    from scipy.stats import wilcoxon\n"
             "    for q in ['mr30hea', 'mr30ryd']:\n"
             "        for c in ['none', 'mlp30']:\n"
             "            a = r3[r3.variant == q].sort_values('seed')['v8'].values\n"
             "            b = r3[r3.variant == c].sort_values('seed')['v8'].values\n"
             "            n = min(len(a), len(b))\n"
             "            if n >= 2:\n"
             "                win = (a[:n] < b[:n]).all()\n"
             "                print(f'{q} vs {c}: seeds={n}, all-seeds-win={win}, '\n"
             "                      f'mean diff={np.mean(a[:n]-b[:n]):+.3f}')\n"
             "print()\n"
             "print('NOTE: n=3 paired Wilcoxon cannot reach p<0.05; the protocol')\n"
             "print('upgrade (round 4) logs per-regime VRMSE -> 9 paired points.')"),
        md("## Problems log (live)\n"
           "1. **Run variance ~0.17 v8** between math-equivalent runs -> multi-seed "
           "everything; per-regime eval to get 9 paired points per comparison.\n"
           "2. **hea qubit scaling non-monotone** at fixed steps (12q>16q) — "
           "consistent with bottleneck-capacity vs optimization tradeoff; the "
           "analog ansatz scaled positively (12->16). 30q multi-register decides.\n"
           "3. **Entangled 30q untrainable** on statevector (8.6 GB, minutes/grad) "
           "-> product-register now, MPS-chi next (CUDA-Q tensornet-mps verified).\n"
           "4. **Sprint protocol != official protocol** (1-frame vs 4-frame "
           "history, downsampled, valid vs test) -> H1d trainer closes the gap.\n"
           "5. cudaq/torch cannot share a process (LLVM clash) -> CUDA-Q referee "
           "runs subprocess-only.\n\n"
           "## Impact analysis\n"
           "- H1 pass -> first quantum-hybrid entry on a public PDE benchmark with "
           "matched controls (literature has zero; closest attempt lost to its own "
           "control).\n- H1 fail -> equally publishable negative with the "
           "strongest controls to date (2607.09905 bar).\n- H2 pass -> first "
           "quantum adapter in a frozen video world model (slot verified open as "
           "of 2026-07-26).\n- Either way: every number reproduces on one 126 GB "
           "desktop."),
        md("## Phase B (live) — beating CNextU-net from its own checkpoint\n\n"
           "**Goal (user):** one-step TEST mean-VRMSE < **0.1934** (our "
           "reproduction of the official checkpoint over 873 windows; published "
           "0.1956), paired per-window Wilcoxon p<0.05.\n\n"
           "Strategy: the quantum bottleneck wraps the U-Net **neck** of the "
           "official checkpoint with a zero-init gate — at initialization the "
           "hybrid IS the published SOTA model bit-for-bit; fine-tuning arms "
           "(vanilla continuation / +30q Rydberg / +matched MLP) start from the "
           "identical point and identical valid score (0.1865)."),
        code("import csv as _csv\n"
             "anchor = np.array([float(r['vrmse']) for r in _csv.DictReader(\n"
             "    open(ROOT / 'runs/eval_official_cnext.csv'))])\n"
             "fig, ax = plt.subplots(1, 1, figsize=(7, 3))\n"
             "ax.hist(anchor, bins=50, color='tab:blue', alpha=.8)\n"
             "ax.axvline(anchor.mean(), c='k', lw=1.5,\n"
             "           label=f'mean {anchor.mean():.4f} (published 0.1956)')\n"
             "ax.set_xlabel('per-window one-step test VRMSE'); ax.legend()\n"
             "ax.set_title('official CNextU-net: 873 test windows (our reproduction)')\n"
             "plt.tight_layout(); plt.show()\n"
             "print(f'mean {anchor.mean():.4f}  median {np.median(anchor):.4f}  '\n"
             "      f'std {anchor.std():.4f}  n={len(anchor)}')"),
        md("### Statistical power of the paired test\nFine-tuned siblings of the "
           "same checkpoint are highly correlated per-window, so the paired test "
           "detects tiny real shifts. Minimal detectable mean shift (alpha=0.05, "
           "power 0.8) under assumed sibling correlations rho:"),
        code("for rho in (0.8, 0.9, 0.95, 0.99):\n"
             "    sd = anchor.std() * np.sqrt(2 * (1 - rho))\n"
             "    print(f'rho={rho:.2f}: detectable shift ~ '\n"
             "          f'{2.8 * sd / np.sqrt(len(anchor)):.4f} VRMSE')\n"
             "print('=> even ~0.001-0.005 improvements are provable at n=873;')\n"
             "print('   the goal does not require a miracle effect size.')"),
        md("### Live fine-tune tracker (re-run this cell during any Phase-B run)"),
        code("runs_dir = ROOT / 'runs/phase_b'\n"
             "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
             "best = {}\n"
             "for f in sorted(glob.glob(str(runs_dir / '*.log'))):\n"
             "    txt = open(f).read()\n"
             "    pts = [(int(m.group(1)), float(m.group(2))) for m in\n"
             "           re.finditer(r'val_check step=(\\d+) vrmse=([\\d.]+)', txt)]\n"
             "    if pts:\n"
             "        s, v = zip(*pts)\n"
             "        name = pathlib.Path(f).stem\n"
             "        ax.plot(s, v, marker='o', label=name)\n"
             "        best[name] = min(v)\n"
             "if best:\n"
             "    ax.axhline(0.1865, ls='--', c='gray', lw=.8, label='checkpoint start (valid)')\n"
             "    ax.set_xlabel('fine-tune step'); ax.set_ylabel('valid VRMSE (stride-2)')\n"
             "    ax.set_title('Phase-B fine-tuning from the official checkpoint'); ax.legend()\n"
             "    plt.tight_layout(); plt.show()\n"
             "    for k, v in sorted(best.items(), key=lambda x: x[1]):\n"
             "        print(f'{k:24s} best valid {v:.4f}')\n"
             "else:\n"
             "    print('no val_check lines yet — runs starting')"),
        md("## Continue here (lab bench)\nFlip `RUN_QUICK=True` when the GPU is "
           "free to train any variant for a short budget and append to the "
           "ledger; then re-run the verdict cell above."),
        code("RUN_QUICK = False\n"
             "VARIANT, SEED, BUDGET_S = 'mr30ryd', 1340, 120\n"
             "if RUN_QUICK:\n"
             "    import subprocess\n"
             "    log = ROOT / 'runs' / f'lab_{VARIANT}_s{SEED}.log'\n"
             "    subprocess.run(['uv', 'run', 'paper/train.py', VARIANT, str(SEED),\n"
             "                    str(BUDGET_S)], cwd=ROOT, stdout=open(log, 'w'),\n"
             "                   stderr=subprocess.STDOUT)\n"
             "    print(open(log).read()[-400:])\n"
             "else:\n"
             "    print('GPU busy with sprint queue — skipped (set RUN_QUICK=True later)')"),
    ]
    return cells


def nb_competitors():
    cells = [
        md("# 04 — Competitors: Latest Physical World Models vs Ours\n\n"
           "Recency-biased landscape (sweep of arXiv/OpenReview/HF/lab pages; see "
           "`notebooks/competitors_data.json` for the sourced rows, one URL per "
           "row). Motivation frame: large-qubit fault-tolerant machines are "
           "decades out — the question is whether a *hybrid* qubit+world-model "
           "gives better analog-like physical simulation **today** than world "
           "models alone, with a native export path to analog hardware."),
        code(PRELUDE),
        code("import json\n"
             "data = json.load(open(ROOT / 'notebooks/competitors_data.json'))\n"
             "comp = pd.DataFrame(data['rows'])\n"
             "print(f\"sweep date: {data['sweep_date']}, rows: {len(comp)}\")\n"
             "comp.sort_values('date', ascending=False).reset_index(drop=True)"),
        md("## The Well TRL2D — the directly comparable arena"),
        code("well = comp[comp['benchmark'].str.contains('turbulent_radiative', case=False)]\n"
             "well = well.copy()\n"
             "well['score_f'] = pd.to_numeric(well['score'], errors='coerce')\n"
             "well = well.dropna(subset=['score_f']).sort_values('score_f')\n"
             "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
             "colors = ['tab:red' if 'ours' in m.lower() else 'tab:blue' for m in well['model']]\n"
             "ax.barh(well['model'], well['score_f'], color=colors)\n"
             "ax.axvline(0.1934, ls='--', c='k', lw=1, label='our reproduced anchor 0.1934')\n"
             "ax.set_xlabel('one-step test VRMSE (lower better; protocol per row)')\n"
             "ax.legend(); plt.tight_layout(); plt.show()\n"
             "print(well[['model','params','metric','score']].to_string(index=False))"),
        md("## Mechanism comparison — how each model 'does physics'"),
        code("mech = comp.dropna(subset=['physics_mechanism']) if 'physics_mechanism' in comp else comp\n"
             "for _, r in mech.iterrows():\n"
             "    if r.get('physics_mechanism'):\n"
             "        print(f\"{r['model']:22s} | {str(r.get('params','?')):>8s} | {r['physics_mechanism'][:90]}\")\n"
             "print()\n"
             "print('OURS: input-conditioned Rydberg-Hamiltonian bottleneck (~500-2k quantum')\n"
             "print('params) inside frozen SOTA backbones; physics via learned analog dynamics')\n"
             "print('+ pre-registered invariant/rollout evaluation; Pulser-exportable.')"),
        md("## Where we sit\n"
           "- **Scale axis**: competitors push parameters (1.3B-64B); our delta is "
           "a ~1.5M-param adapter on a frozen backbone — the efficiency end of the "
           "frontier, reproducible on one 126 GB desktop.\n"
           "- **Mechanism axis**: published physics mechanisms are architectural "
           "priors, data scale, or losses; none couples to an analog-hardware-native "
           "dynamical system. That slot (audit C5/C1) is ours.\n"
           "- **Evaluation axis**: rollout collapse past ~13 steps remains "
           "universal (incl. 1.3B Walrus) — long-horizon stability is the open "
           "flank our quantum arms attack.\n"
           "- **Risk**: the field moves ~1 relevant paper/month (audit 09); "
           "re-run this sweep before submission."),
    ]
    return cells


def nb_trl2d():
    cells = [
        md("# 05 — Challenge: The Well `turbulent_radiative_layer_2D`\n\n"
           "**Leaderboard #1 (Phase A of the master plan).** Beat the strongest "
           "official baseline — CNextU-net, published 0.1956 one-step test "
           "mean-VRMSE, **our reproduction 0.1934** over 873 windows — via "
           "checkpoint surgery: the 30-simulated-qubit Rydberg bottleneck wrapped "
           "around the official checkpoint's neck (bit-identical at init), "
           "fine-tuned under identical budgets as vanilla-continuation and "
           "matched-MLP controls. Significance: paired per-window Wilcoxon, "
           "3 seeds. Kill rule: controls match quantum -> negative result "
           "published as-is.\n\nEvery figure below recomputes from `runs/` on "
           "re-execution — this notebook IS the verification artifact."),
        code(PRELUDE),
        md("## Arms and their fine-tuning trajectories (live)"),
        code("logs = sorted(glob.glob(str(ROOT / 'runs/phase_b/pilot_*.log')) +\n"
             "              glob.glob(str(ROOT / 'runs/phase_b/ft_*.log')))\n"
             "fig, ax = plt.subplots(figsize=(9, 4))\n"
             "for f in logs:\n"
             "    pts = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(\n"
             "        r'val_check step=(\\d+) vrmse=([\\d.]+)', open(f).read())]\n"
             "    if pts:\n"
             "        s, v = zip(*pts)\n"
             "        ax.plot(s, v, marker='o', ms=3, label=pathlib.Path(f).stem)\n"
             "ax.axhline(0.18985, ls='--', c='gray', lw=1, label='control best (3k steps)')\n"
             "ax.set_xlabel('fine-tune step'); ax.set_ylabel('valid VRMSE (stride-2)')\n"
             "ax.set_title('checkpoint-surgery arms, identical start 0.1902')\n"
             "ax.legend(fontsize=7); plt.tight_layout(); plt.show()"),
        md("## Campaign ledger (overnight arms appear here as they run)"),
        code("rows = []\n"
             "for f in logs:\n"
             "    txt = open(f).read()\n"
             "    tr = re.search(r'target_reached steps=(\\d+) best=([\\d.]+)', txt)\n"
             "    checks = re.findall(r'val_check step=(\\d+) vrmse=([\\d.]+)', txt)\n"
             "    rows.append(dict(arm=pathlib.Path(f).stem,\n"
             "                     status='DONE' if tr else ('running' if checks else 'queued'),\n"
             "                     steps=int(checks[-1][0]) if checks else 0,\n"
             "                     best=min((float(v) for _, v in checks), default=None)))\n"
             "pd.DataFrame(rows).sort_values('best', na_position='last').reset_index(drop=True)"),
        md("## Final verdict (populates when single-shot test evals exist)"),
        code("import subprocess\n"
             "tests = sorted(glob.glob(str(ROOT / 'runs/phase_b/test_*.csv')))\n"
             "if not tests:\n"
             "    print('no test-set evaluations yet — test set is touched exactly once,')\n"
             "    print('after each arm completes (single-shot discipline, see 08-scoreboard).')\n"
             "else:\n"
             "    for t in tests:\n"
             "        print('===', pathlib.Path(t).name, '===')\n"
             "        r = subprocess.run(['uv', 'run', 'paper/phase_b/paired_stats.py', t,\n"
             "                            str(ROOT / 'runs/eval_official_cnext.csv')],\n"
             "                           capture_output=True, text=True, cwd=ROOT)\n"
             "        print(r.stdout)"),
        md("## Threat ledger for this leaderboard\n"
           "- **WaveLiT-L 0.1308 / -B 0.1346** (arXiv:2605.25949): *claimed* "
           "Well-consistent protocol, workshop venue, 2D-only — protocol "
           "verification is a Phase-F task; our claim rests on the official "
           "baseline + paired significance either way.\n"
           "- 'Physics Transformer' 0.15: non-comparable (their FNO 0.29 vs "
           "official 0.5001).\n"
           "- Walrus 0.0831 / Poseidon-L 0.1323: median-VRMSE protocol, "
           "different metric — context, not competitors on this board."),
    ]
    return cells


def nb_adaptive():
    cells = [
        md("# 15 — Adaptive Geometry (Qubits-on-the-Go): E-series ledger\n\n"
           "**Thesis (user):** neutral atoms can restructure their register per "
           "simulation — so geometry becomes learnable, input-conditioned, and "
           "continually adapted. Design: 13-adaptive-geometry.md + "
           "14-structure-bakeoff.md. This notebook recomputes every E-series "
           "verdict from `runs/ag*_*.log` on re-execution."),
        code(PRELUDE),
        code("def parse_ag(pattern):\n"
             "    rows = []\n"
             "    for f in sorted(glob.glob(str(ROOT / 'runs' / pattern))):\n"
             "        txt = open(f).read()\n"
             "        g = lambda k: re.search(rf'^{k}:\\s+(\\S+)', txt, re.M)\n"
             "        if not g('val_vrmse_1'):\n"
             "            continue\n"
             "        m = re.match(r'ag(\\d+)_(.+?)_k(\\d+)_s(\\d+)', pathlib.Path(f).stem)\n"
             "        rows.append(dict(wave=m.group(1), variant=m.group(2),\n"
             "                         k=int(m.group(3)), seed=int(m.group(4)),\n"
             "                         v1=float(g('val_vrmse_1').group(1)),\n"
             "                         v8=float(g('val_vrmse_8').group(1)),\n"
             "                         steps=int(g('num_steps').group(1))))\n"
             "    return pd.DataFrame(rows)\n"
             "ag = parse_ag('ag*_*.log')\n"
             "ag.sort_values(['wave','k','v8']).reset_index(drop=True)"),
        md("## Fair pairings (matched steps, matched objective) — the E1/E3 verdicts"),
        code("w1 = ag[ag.wave == '1']\n"
             "pairs = [('cg30','mr30ryd',1,'E1: conditioned vs static'),\n"
             "         ('cg30','mr30ryd',4,'E1 under TD'),\n"
             "         ('cg30-3d','cg30',1,'E3: 3D vs 2D')]\n"
             "for a, b, k, label in pairs:\n"
             "    ra = w1.query('variant==@a and k==@k'); rb = w1.query('variant==@b and k==@k')\n"
             "    if len(ra) and len(rb):\n"
             "        va, vb = ra.v8.iloc[0], rb.v8.iloc[0]\n"
             "        sa, sb = ra.steps.iloc[0], rb.steps.iloc[0]\n"
             "        fair = abs(sa-sb)/max(sa,sb) < 0.1\n"
             "        print(f'{label:28s} {a} {va:.3f} vs {b} {vb:.3f} '\n"
             "              f'({100*(va/vb-1):+.1f}%) steps {sa}/{sb} fair={fair}')"),
        md("## Wave 1.5 — fixed-step deconfounded grid (populates live)"),
        code("w15 = ag[ag.wave == '15']\n"
             "if len(w15) == 0:\n"
             "    print('wave 1.5 running — re-execute when WAVE15_DONE')\n"
             "else:\n"
             "    piv = w15.pivot_table(index='variant', columns=['k','seed'], values='v8')\n"
             "    print(piv.round(3).to_string())\n"
             "    if len(w15) >= 12:\n"
             "        m = w15.groupby(['variant','k']).v8.agg(['mean','std','count'])\n"
             "        print(); print(m.round(3).to_string())"),
        md("## Standing verdicts (wave-1, seed 1337, honest caveats)\n"
           "- **E1 supported 2/2 fair pairings**: conditioned > static at k=1 "
           "(1.108 vs 1.116) and k=4 (0.912 vs 1.038, −12%).\n"
           "- **E3 strong**: 3D conditioned 0.888 vs 2D 1.108 (−20%) — best k=1 "
           "rollout in program; 3D = simulation-only on Pasqal today.\n"
           "- **E4 double-edged**: TD objective un-starves the baseline (−13%) "
           "AND the classical control (0.885, wave-best wall-clock) — the "
           "quantum-specific claim must survive wave-1.5's fixed-step stats.\n"
           "- **kmix30 pathology**: soft mixture evaluates all banks (4× cost, "
           "averaged dynamics) — hard Gumbel routing required before the "
           "bake-off uses it.\n"
           "- Single-seed wave-1 numbers are directional; wave-1.5 (3 seeds, "
           "fixed 2,000 steps) is the statistical referee."),
    ]
    return cells


def nb_selfimprove():
    cells = [
        md("# 16 — The Self-Improvement System: One Quantum Layer, Every Problem\n\n"
           "**Foundation rule (user, 2026-08-17):** the SAME 3×10-qubit adaptive "
           "layer enters every challenge; what changes per problem is the "
           "*register configuration* — chosen by the router, shaped by the "
           "conditioned geometry, refined by TD test-time adaptation, and "
           "renewed by evolutionary bank refresh (14-structure-bakeoff.md). "
           "This notebook: the model, samples of every dataset on disk, the "
           "register visibly reconfiguring per problem, a live evolution demo, "
           "and the real results ledger by configuration."),
        code(PRELUDE),
        md("## The model"),
        code("import matplotlib.patches as mp\n"
             "fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis('off')\n"
             "def box(x, y, w, h, label, ec='0.3', lw=1.5, fs=9):\n"
             "    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',\n"
             "                 fill=False, edgecolor=ec, linewidth=lw))\n"
             "    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fs)\n"
             "def arrow(x1, y1, x2, y2, label='', c='0.3', ls='-'):\n"
             "    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color=c, ls=ls))\n"
             "    if label: ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, ha='center', fontsize=8, color=c)\n"
             "box(0.01, 0.30, 0.17, 0.44, 'FROZEN BACKBONE\\nCNextU-net / LingBot 14B\\n/ Cosmos /任 encoder')\n"
             "box(0.24, 0.42, 0.09, 0.20, 'pool → h')\n"
             "box(0.40, 0.24, 0.26, 0.56, '3×10q RYDBERG REGISTERS\\nδᵢ(h) detunings · rᵢ geometry\\nΩ drive · blockade C₆/r⁶', ec='teal', lw=2)\n"
             "box(0.73, 0.42, 0.12, 0.20, '⟨Zᵢ⟩ → W_out\\nα⊙residual (α₀=0)')\n"
             "arrow(0.18, 0.52, 0.24, 0.52, 'latent tap')\n"
             "arrow(0.33, 0.55, 0.40, 0.60, 'δ(h)', c='teal')\n"
             "arrow(0.33, 0.49, 0.40, 0.40, 'Δr(h)', c='teal')\n"
             "arrow(0.66, 0.52, 0.73, 0.52, 'measure')\n"
             "arrow(0.79, 0.42, 0.79, 0.16, ''); arrow(0.79, 0.16, 0.10, 0.16, ''); arrow(0.10, 0.16, 0.10, 0.30, 'gated residual')\n"
             "arrow(0.53, 0.24, 0.53, 0.10, 'TD / rollout loss ← self-improvement', c='0.4', ls='--')\n"
             "arrow(0.53, 0.80, 0.53, 0.92, 'Pulser export → Pasqal/QuEra', c='teal', ls='--')\n"
             "ax.text(0.53, 0.02, 'router picks register bank per problem · evolution refreshes the bank · e-prop/SPSA adapt on hardware',\n"
             "        ha='center', fontsize=8.5)\n"
             "ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.tight_layout(); plt.show()"),
        md("## Every dataset on disk, one sample each\n(TokaMark and Sophelio: "
           "access verified, data intentionally not yet local — marked pending.)"),
        code("import h5py, glob as g\n"
             "fig, axs = plt.subplots(1, 4, figsize=(13, 3.1))\n"
             "# TRL2D (sprint cache)\n"
             "tr = torch.load(ROOT/'data/cache.pt', weights_only=True, mmap=True)['train']\n"
             "axs[0].imshow(tr[0, 60, 0].T, cmap='RdBu_r', aspect='auto')\n"
             "axs[0].set_title(f'TRL2D density {tuple(tr.shape[-2:])}', fontsize=9)\n"
             "# MHD_64 (one valid file, mid-slice)\n"
             "mf = sorted(g.glob(str(ROOT/'data/mhd64/data/valid/*.hdf5')))[0]\n"
             "with h5py.File(mf) as h:\n"
             "    key = [k for k in h['t0_fields']][0]\n"
             "    mhd = h[f't0_fields/{key}'][0, 50]\n"
             "axs[1].imshow(mhd[:, :, 32], cmap='magma')\n"
             "axs[1].set_title(f'MHD_64 {key} 64³ slice', fontsize=9)\n"
             "# TORAX pilot (Te heatmap time×radius)\n"
             "tx = np.load(ROOT/'data/torax_pilot/discharge_00000.npz')\n"
             "axs[2].imshow(tx['T_e'].T, aspect='auto', cmap='inferno', origin='lower')\n"
             "axs[2].set_title(f\"TORAX T_e {tx['T_e'].shape} (t×ρ)\", fontsize=9)\n"
             "# Pelzer prepared (first .pt found)\n"
             "pf = sorted(g.glob(str(ROOT/'data/geothermal/darus-3653/*.pt')))\n"
             "if pf:\n"
             "    p = torch.load(pf[0], weights_only=True)\n"
             "    t = p if torch.is_tensor(p) else next(v for v in (p.values() if isinstance(p, dict) else p) if torch.is_tensor(v))\n"
             "    img = t.squeeze()[-1] if t.dim() > 2 else t.squeeze()\n"
             "    axs[3].imshow(img.T if img.shape[0] < img.shape[-1] else img, cmap='coolwarm', aspect='auto')\n"
             "    axs[3].set_title(f'Pelzer geothermal {tuple(t.shape)}', fontsize=9)\n"
             "else:\n"
             "    axs[3].text(0.5, 0.5, 'Pelzer: fetch in progress', ha='center'); axs[3].set_title('geothermal', fontsize=9)\n"
             "for a in axs: a.set_xticks([]); a.set_yticks([])\n"
             "plt.suptitle('four physics worlds, one quantum layer', y=1.02)\n"
             "plt.tight_layout(); plt.show()\n"
             "print('pending data: TokaMark (MAST S3 verified), Sophelio (133GB, dev phase to Oct 18)')"),
        md("## The register reconfigures per problem\nEach dataset's pooled "
           "latent drives the geometry hypernetwork and detunings — same block, "
           "different physical program per problem. (Hypernet seeded for "
           "visibility — the *mechanism*; trained-configuration results below.)"),
        code("from qlayer import ConditionedGeometryBlock\n"
             "torch.manual_seed(7)\n"
             "blk = ConditionedGeometryBlock(d_in=24, n=10, layers=4, dims=2)\n"
             "with torch.no_grad():\n"
             "    blk.geo[-1].weight.normal_(0, 0.25)  # visibility: untrained hypernet is zero-init\n"
             "proj = torch.randn(24, 512, generator=torch.Generator().manual_seed(3))\n"
             "def latent_of(x):\n"
             "    v = torch.nn.functional.adaptive_avg_pool2d(x[None, None].float(), (16, 32)).flatten()\n"
             "    return (proj @ v[:512]) / 10\n"
             "samples = {'TRL2D': tr[0, 60, 0], 'MHD': torch.tensor(mhd[:, :, 32]),\n"
             "           'TORAX': torch.tensor(tx['T_e']), 'GEO': (img if pf else tr[1, 30, 1])}\n"
             "fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))\n"
             "for j, (name, s) in enumerate(samples.items()):\n"
             "    h = latent_of(s)\n"
             "    with torch.no_grad():\n"
             "        shift = blk.MAX_SHIFT * torch.tanh(blk.geo(h[None])).reshape(10, 2)\n"
             "        pos = (blk.pos0 + shift).numpy()\n"
             "        det = np.pi * torch.tanh(blk.cond(h[None]).reshape(4, 10)).numpy()\n"
             "    d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))\n"
             "    for a in range(10):\n"
             "        for b in range(a + 1, 10):\n"
             "            if d[a, b] < 1.15:\n"
             "                axs[0, j].plot(*zip(pos[a], pos[b]), c='teal', alpha=0.45, lw=1)\n"
             "    axs[0, j].scatter(pos[:, 0], pos[:, 1], s=90, c='teal', zorder=3)\n"
             "    axs[0, j].set_title(f'{name}: register geometry', fontsize=9)\n"
             "    axs[0, j].set_aspect('equal'); axs[0, j].set_xticks([]); axs[0, j].set_yticks([])\n"
             "    im = axs[1, j].imshow(det, cmap='PuOr', vmin=-np.pi, vmax=np.pi, aspect='auto')\n"
             "    axs[1, j].set_title('detunings δ (layer × qubit)', fontsize=9)\n"
             "    axs[1, j].set_xticks([]); axs[1, j].set_yticks([])\n"
             "plt.suptitle('one block — four different quantum programs (edges = blockade-coupled pairs)', y=1.0)\n"
             "plt.tight_layout(); plt.show()"),
        md("## Live self-improvement: evolving the register (CPU micro-demo)\n"
           "Quality-diversity evolution (pyribs CMA-ME) over 8-qubit geometries; "
           "fitness = cross-problem response diversity (variance of ⟨Z⟩ readouts "
           "across the four datasets' latents) — the bake-off's evolutionary arm "
           "in miniature, running live on the Grace cores."),
        code("from qlayer import QuantumBlock\n"
             "from ribs.archives import GridArchive\n"
             "from ribs.emitters import EvolutionStrategyEmitter\n"
             "from ribs.schedulers import Scheduler\n"
             "torch.manual_seed(0)\n"
             "qb = QuantumBlock(d_in=24, n=8, layers=3, ansatz='rydberg')\n"
             "with torch.no_grad():\n"
             "    qb.cond.weight.normal_(0, 0.3)  # conditioning is zero-init by design;\n"
             "    # seed it so inputs actually differentiate the dynamics for this demo\n"
             "H = torch.stack([latent_of(s) for s in samples.values()])\n"
             "def fitness(flat):\n"
             "    with torch.no_grad():\n"
             "        qb.pos.data = torch.tensor(flat.reshape(8, 2), dtype=torch.float32)\n"
             "        z = qb(H)\n"
             "    return z.var(dim=0).mean().item()\n"
             "def descriptors(flat):\n"
             "    p = flat.reshape(8, 2)\n"
             "    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))\n"
             "    iu = np.triu_indices(8, 1)\n"
             "    return [d[iu].mean(), (d[iu] < 1.15).mean()]\n"
             "x0 = qb.pos.detach().numpy().flatten()\n"
             "archive = GridArchive(solution_dim=16, dims=[12, 12],\n"
             "                      ranges=[(0.4, 3.0), (0.0, 1.0)])\n"
             "emitters = [EvolutionStrategyEmitter(archive, x0=x0, sigma0=0.15, batch_size=12)]\n"
             "sched = Scheduler(archive, emitters)\n"
             "best_hist, best_geom = [], x0\n"
             "for gen in range(18):\n"
             "    sols = sched.ask()\n"
             "    objs = [fitness(s) for s in sols]\n"
             "    descs = [descriptors(s) for s in sols]\n"
             "    sched.tell(objs, descs)\n"
             "    stats = archive.stats\n"
             "    best_hist.append(stats.obj_max)\n"
             "    be = archive.best_elite\n"
             "    if be is not None:\n"
             "        best_geom = be['solution'] if isinstance(be, dict) else be.solution\n"
             "fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))\n"
             "axs[0].plot(best_hist, marker='o', ms=3, c='teal')\n"
             "axs[0].set_title('best fitness per generation'); axs[0].set_xlabel('generation')\n"
             "for pts, ttl, ax in [(x0.reshape(8, 2), 'initial register', axs[1]),\n"
             "                     (np.asarray(best_geom).reshape(8, 2), 'evolved register', axs[2])]:\n"
             "    ax.scatter(pts[:, 0], pts[:, 1], s=90, c='teal')\n"
             "    ax.set_title(ttl); ax.set_aspect('equal')\n"
             "plt.tight_layout(); plt.show()\n"
             "print(f'fitness {best_hist[0]:.4f} → {best_hist[-1]:.4f} '\n"
             "      f'({100*(best_hist[-1]/max(best_hist[0],1e-9)-1):+.0f}%), archive fills {archive.stats.num_elites} niches')"),
        md("## Real results by qubit configuration (from results.tsv — every "
           "number a completed run)"),
        code("import csv as _csv\n"
             "rows = list(_csv.DictReader(open(ROOT/'results.tsv'), delimiter='\\t'))\n"
             "import collections\n"
             "groups = collections.defaultdict(list)\n"
             "for r in rows:\n"
             "    d = r['description']\n"
             "    if 'wave15' in d:\n"
             "        v, k = d.split()[2], d.split()[3]\n"
             "        groups[f'{v} {k}'].append(float(r['val_vrmse_8']))\n"
             "fig, ax = plt.subplots(figsize=(9, 3.6))\n"
             "names = sorted(groups, key=lambda n: np.mean(groups[n]))\n"
             "means = [np.mean(groups[n]) for n in names]\n"
             "stds = [np.std(groups[n]) for n in names]\n"
             "colors = ['#0f8f80' if any(q in n for q in ('mr30', 'cg30')) else '0.5' for n in names]\n"
             "ax.bar(names, means, yerr=stds, capsize=3, color=colors)\n"
             "ax.set_ylabel('rollout VRMSE@8 (fixed 2000 steps, 3 seeds)')\n"
             "ax.set_title('configurations compared — teal = quantum registers')\n"
             "plt.xticks(rotation=20); plt.tight_layout(); plt.show()\n"
             "print('headline: mr30ryd k4 = 0.845±0.051 beats matched control 0.953 on 3/3 seeds;')\n"
             "print('TD objective is what unlocks the gap (quantum −23% vs control −10%).')"),
        md("## Where the self-improvement bake-off goes next\n"
           "Four strategies adapt the SAME K-register bank on the same continual "
           "stream (TRL2D regime shift → TORAX transfer), equal eval-unit "
           "budgets, kill rules pre-registered: **gradient** (log-coupling "
           "recipe) · **evolution** (this notebook's demo at full scale, "
           "tier-0 weight-inheritance fitness) · **e-prop** (eligibility traces "
           "from measured ⟨ZᵢZⱼ⟩ correlators × TD error — needs no backprop → "
           "hardware-in-the-loop trainable) · **SPSA** (hardware-standard "
           "control). Every strategy must beat an RLS readout-only baseline. "
           "Verdicts land in this notebook as the bake-off runs."),
    ]
    return cells


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python"}
    path = NBDIR / name
    nbf.write(nb, path)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(NBDIR)}})
    client.execute()
    nbf.write(nb, path)
    n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"executed {name}: {len(nb.cells)} cells, {n_out} with outputs")


if __name__ == "__main__":
    import sys as _sys

    NBDIR.mkdir(exist_ok=True)
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    todo = {
        "01_task_analysis.ipynb": nb_task,
        "02_baselines_evaluation.ipynb": nb_baselines,
        "03_our_solution.ipynb": nb_solution,
        "04_competitors.ipynb": nb_competitors,
        "05_trl2d_challenge.ipynb": nb_trl2d,
        "15_adaptive_geometry.ipynb": nb_adaptive,
        "16_self_improvement.ipynb": nb_selfimprove,
    }
    for name, fn in todo.items():
        if only and only not in name:
            continue
        if fn is nb_competitors and not (NBDIR / "competitors_data.json").exists():
            print(f"skip {name}: competitors_data.json not yet written")
            continue
        build(name, fn())
