"""Build and EXECUTE notebooks/19_models_and_interpretation.ipynb (user request Sep 24 2026): our model (backbone, neck,
head, register) with a proper diagram, where each benchmark stands against its SoTA, and model interpretation per
benchmark (what the trunk gets wrong, where the atoms go, what the trained register learned, and whether the quantum
core adds anything over its matched classical twin).

Run:  .venv/bin/python paper/build_interpret_notebook.py      (GPU cells; interpretation models from paper/alloc/interp.py)
"""
import pathlib

import nbformat as nbf
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "19_models_and_interpretation.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

PRELUDE = r"""
import sys, pathlib, json, re, warnings
warnings.filterwarnings('ignore')
ROOT = pathlib.Path().resolve().parent
for p in ('paper', 'paper/alloc', 'paper/phase_b', 'paper/geothermal'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
pd.set_option('display.precision', 4); pd.set_option('display.max_colwidth', 110); pd.set_option('display.width', 200)
plt.rcParams['figure.dpi'] = 100
import nbviz, interp, analyze
print('GPU:', torch.cuda.get_device_name(0))
"""

ROWS = {
    "turbulent_radiative_layer_2D": dict(
        title="4.1 The Well TRL2D: turbulent radiative layer (headline row)",
        model="**Backbone/neck/head:** released CNextU-net (encoder 42→336 ch; neck 672 ch at 8×24; decoder + 1×1 head, "
              "4 fields). **Quantum:** pooled register in the neck (audit and TD fine-tuning), routed co-processor after the head (frozen study).",
        take="**Interpretation.** Nothing the trunk gets wrong at fine scale is predictable from causal inputs: every tile strategy, "
             "every core and a 64× larger MLP capture zero, even with oracle placement (ceiling 44%). The pipeline is not the limit: "
             "fed the residual itself, the same co-processor captures 99.7% of the ceiling (VRMSE 0.194 → 0.150). The pooled insertion "
             "captures a sliver of its DC ceiling (+0.06%). The one register-specific effect on this row is training-mediated: under "
             "temporal-difference fine-tuning the register lowers long-horizon rollout error (0.953 vs 1.030 trunk, 1.032 MLP), and the "
             "band analysis below shows it acts only at wavenumber 0, as the pooling proposition requires."),
    "rayleigh_benard": dict(
        title="4.2 rayleigh_benard: convection (fission-proxy thermal hydraulics)",
        model="**Backbone/neck/head:** released CNextU-net (512×128, buoyancy, pressure, velocity). **Quantum:** pooled register in "
              "the neck (audit) and routed co-processor on the residual (study).",
        take="**Interpretation.** The released trunk's error is large and systematic at the convection onset (initial transient "
             "VRMSE ≈ 7). 30 routed atoms cut mean VRMSE by ~36% (median −4%, 94% of windows improve) against 6–7% for the pooled "
             "insertion. This is the one row where a nonlinear core matters (+~10% over the linear corrector), but the register and "
             "the matched MLP are equal; the mean-field register is within noise."),
    "shear_flow": dict(
        title="4.3 shear_flow: 2-D shear-layer turbulence",
        model="**Backbone/neck/head:** released CNextU-net (256×512). Flag: the released checkpoint scores 0.9946 on our evaluator vs "
              "0.8080 published, so its error is partly a calibration mismatch. **Quantum:** pooled + routed.",
        take="**Interpretation.** The cleanest utilization gradient of the study: random (U 0.0003) → pooled (0.039) → static "
             "(0.215) → per-input routing (0.30–0.34), and the captured share tracks U at ~95% efficiency, so the gain follows the "
             "allocation almost exactly (+26% vs +3.5–7%). All three cores coincide to four digits: the correction is essentially "
             "linear once the atoms are in the right place. With 300 routed atoms, 68% of the error energy is removed."),
    "MHD_64": dict(
        title="4.4 MHD_64: 3-D magnetohydrodynamic turbulence (fusion)",
        model="**Backbone/neck/head:** released 3-D CNextU-net (64³, 7 fields). Flag: released checkpoint scores 0.5447 on our "
              "evaluator vs 0.1633 published. **Quantum:** pooled + routed, 3-D Haar tiles (262k per window).",
        take="**Interpretation.** Ceilings are small because the error is spread over 262k tiles (DC 0.2%, 30 routed tiles 2.3%), "
             "but what is allocated is captured at ~96%: gain +0.9% with 30 atoms, rising with K (captured 0.9% → 7.9% from 10 "
             "to 300 atoms). Cores identical."),
    "realpde_unet": dict(
        title="4.5 RealPDEBench airfoil, released U-Net (real PIV measurements)",
        model="**Backbone/neck/head:** RealPDEBench's released U-Net (20→20 frames, 64×128, u,v). **Quantum:** band-limited "
              "Fourier corrector (earlier, 1.2%) and routed Haar co-processor (study).",
        take="**Interpretation.** The U-Net's remaining error is mostly shared across architectures (a measurement-noise floor) and "
             "diffuse in space-time; the Haar allocation of 30 tiles reaches only 1.7% of it and captures ~0. The DC part is "
             "predictable (+0.12%), and the earlier low-mode Fourier corrector with a much larger block (+1.2%, part of our SoTA "
             "ensemble 0.00931) remains the better allocation here: the right basis matters as much as the right tiles."),
    "realpde_fno": dict(
        title="4.6 RealPDEBench airfoil, released FNO",
        model="**Backbone/neck/head:** RealPDEBench's released FNO. **Quantum:** routed co-processor (study); earlier low-mode "
              "Fourier corrector (−10%).",
        take="**Interpretation.** The FNO's sim-to-real bias is systematic and low-frequency: every causal strategy captures ~93% "
             "of its ceiling (+3.9% with 30 Haar tiles; the 140-mode Fourier corrector reaches −10%). Register, MLP and linear "
             "cores are identical; the gain belongs to the allocation."),
    "pelzer": dict(
        title="4.7 Pelzer 1HP heat plume (geothermal)",
        model="**Backbone/neck/head:** 1HP_NN U-Net (inputs g,k,s,i → temperature, 256×16), trained by us on 700 maps. "
              "**Quantum:** pooled register at the deepest encoder (audit) and routed co-processor (study).",
        take="**Interpretation.** A data-limited trunk leaves a systematic plume-edge error in fixed places: static allocation "
             "(+9.2%) is as good as per-input routing, and oracle placement would give +15%. Inside the joint model the register's "
             "core carries a larger share of the correction energy than the MLP's core does, yet the end result is no better; the "
             "planar register (no scale axis) is slightly better than the 3-D one."),
}


def row_cells(bench: str, spec: dict):
    return [
        md(f"### {spec['title']}\n\n{spec['model']}"),
        code(f"B = {bench!r}\nalloc = interp.allocation(B)\n"
             "print('utilization ceilings (share of the trunk error energy reachable by each rule):')\n"
             "display(pd.Series({k: v for k, v in alloc['ceil'].items() if k == 'dc' or k.endswith('@30') or k.startswith('oracle')}).round(4).to_frame('U').T)\n"
             "nbviz.allocation_plot(alloc, B); plt.show()"),
        code("display(nbviz.row_table(B).round(4))\n"
             "fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); nbviz.ksweep(B, ax[0])\n"
             "run, m, st, d = interp.load(B, 'ryd'); x, y = interp.correction_vs_residual(run, m, st)\n"
             "nbviz.scatter_corr(x, y, B, ax[1]); plt.tight_layout(); plt.show()\n"
             "display(nbviz.controls(B).round(4))"),
        code("print('best strategy (VALID):', st)\n"
             "print('learned analog controls:', interp.register_physics(m))\n"
             "print('readout usage (share of |A| per observable):', interp.readout_usage(m))\n"
             "print('register path split:', {k: round(v, 4) for k, v in interp.path_split(run, m, st).items()})\n"
             "print('entanglement of the trained register on test groups:', {k: round(v, 3) for k, v in interp.entanglement(run, m, st).items()})\n"
             "run2, m2, _, _ = interp.load(B, 'mlp')\n"
             "print('matched-MLP path split:', {k: round(v, 4) for k, v in interp.path_split(run2, m2, st).items()})"),
        md(spec["take"]),
    ]


def build():
    cells = [
        md("# 19 · QUOTA: the model and its interpretation on every benchmark\n\n"
           "**QUOTA** (*Qubit Utilization thrOugh Trained Allocation*) is a hybrid physics world model: a pretrained classical "
           "trunk does the part it does well (the *digital* part), and small Rydberg registers work only on the part it gets "
           "wrong (the *analog* part), at places chosen per input. This notebook shows\n\n"
           "1. **the model**: backbone, neck and head, the two quantum insertion points, and the register itself;\n"
           "2. **the theory** it is built on (machine-checked in Lean 4);\n"
           "3. **where every benchmark stands** against its state of the art;\n"
           "4. **per-benchmark interpretation**: where the trunk's error lives, where the atoms go, what the trained register "
           "learned, and whether it beats its matched classical twin;\n"
           "5. the benchmarks outside the allocation study (Sophelio, TokaMark, TORAX, TDC);\n"
           "6. the analog-vs-digital probe; 7. the synthesis.\n\n"
           "All numbers come from result files in `runs/`; interpretation models are seed-1 refits at each row's validation-best "
           "strategy (`paper/alloc/interp.py`). Everything is classically simulated (≤10 atoms per register); nothing here is a "
           "quantum-advantage claim."),
        code(PRELUDE),
        md("## 1. The model: backbone, neck, head\n\n"
           "Every trunk is split into three slots. The **backbone** encodes the input history; the **neck** is the "
           "deepest, lowest-resolution representation; the **head** decodes to the predicted fields. QUOTA inserts a register at "
           "two places:\n\n"
           "* **in the neck (the audit insertion):** mean-pool the neck features, run a 3×10-atom register on the pooled vector, "
           "and broadcast a zero-initialized gated correction back. It starts exactly at the released model, but it can only read "
           "and write the spatial DC mode (Lean `pool_blind`).\n"
           "* **after the head (the routed co-processor):** write the trunk's residual in an orthonormal Haar basis, let an "
           "allocation rule pick K tiles per input, place one atom per tile (x, y from position, z from scale), and write a "
           "correction back to exactly those tiles. The trunk is frozen; the corrected error is exact from coefficients.\n\n"
           "The matched classical control uses the identical boxes with the register replaced by a parameter-matched MLP."),
        code("nbviz.architecture(); plt.show()"),
        md("### 1.1 Backbone, neck and head of every benchmark"),
        code("display(nbviz.backbone_table())"),
        md("### 1.2 Live: the released CNextU-net split into its slots, and the register in its neck\n\n"
           "Parameter counts per slot for each released Well checkpoint, then the TRL2D hybrid: the neck is wrapped with the "
           "gated bottleneck; with α = 0 the hybrid reproduces the released model exactly (checked on the GPU)."),
        code("from cnext_hybrid import load_official, build_hybrid\n"
             "rows = {}\n"
             "for name, d in [('TRL2D', 'data/cnext_ckpt'), ('rayleigh_benard', 'data/cnext_rb_ckpt'), ('shear_flow', 'data/cnext_shear_flow_ckpt'), ('MHD_64 (3-D)', 'data/cnext_mhd_ckpt')]:\n"
             "    mdl, cfg = load_official(str(ROOT / d)); rows[name] = {**nbviz.cnext_parts(mdl), 'total': nbviz.count(mdl)}\n"
             "display(pd.DataFrame(rows).T)\n"
             "base = build_hybrid('vanilla').cuda().eval(); hyb = build_hybrid('mr30ryd').cuda().eval(); mlp = build_hybrid('mlp30').cuda().eval()\n"
             "bq, bm = hyb.neck.bottleneck, mlp.neck.bottleneck\n"
             "print(f'neck insertion: register core {nbviz.count(bq.core):,} params (+ up-projection {nbviz.count(bq.up):,}, gate {bq.alpha.numel()}); '\n"
             "      f'matched MLP core {nbviz.count(bm.core):,}')\n"
             "x = torch.randn(1, 16, 128, 384, device='cuda')\n"
             "with torch.no_grad(): print('zero-init identity, max |hybrid - released| =', float((hyb(x) - base(x)).abs().max()))"),
        md("### 1.3 The register close-up\n\n"
           "Each register is 10 atoms under $H/\\hbar = \\sum_i \\tfrac{\\Omega_\\ell}{2}X_i - \\sum_i \\delta_{\\ell,i} n_i + "
           "\\sum_{i<j} C_6 |r_i-r_j|^{-6} n_i n_j$, simulated exactly as $L=4$ layers of per-atom $R_z(\\delta)R_x(\\Omega)$ "
           "followed by the interaction phase. Detunings encode the tile features; positions come from the allocation, so the "
           "interaction graph changes with every input. Below: one trained register on a real heat-plume test input."),
        code("run, m, st, d = interp.load('pelzer', 'ryd')\npos, V, det, o = interp.one_group(run, m, st)\n"
             "nbviz.register_closeup(pos, V, det, o); plt.show()\n"
             "from cores import make_tile_core, n_params\nF = run.fz.F\n"
             "display(pd.Series({k: n_params(make_tile_core(k, F)) for k in ['ryd', 'gutz', 'ryd2d', 'mlp', 'mlpX4', 'hyb', 'mlp2']}, name=f'core parameters (F={F} features)').to_frame().T)"),
        md("## 2. The theory (machine-checked in Lean 4, standard axioms only)\n\n"
           "* `pool_blind`, `broadcast_orthogonal`: a pooled insertion reads and writes only the DC mode, whatever its core.\n"
           "* `residual_after_band_removal`, **`gain_le_ceiling`**: a correction supported on a component set removes at most the error "
           "energy there, so the captured share never exceeds the utilization ceiling U.\n"
           "* **`ceiling_mono`**: nested allocations have nested ceilings; **`pooled_gain_le_dc`**: a broadcast correction removes at most "
           "$(\\sum_i e_i)^2/n$, however many atoms produced it.\n"
           "* `swap_improves`: under a budget, top-K by gain is optimal.\n\n"
           "These prove **where** atoms must work. They do not prove that atoms do the job better than a classical core; that is "
           "tested empirically below. The cell lists the theorems and checks two of them numerically."),
        code("for f in sorted((ROOT / 'formal' / 'QuotaScaling').glob('*.lean')):\n"
             "    print(f'{f.name:22s}', ', '.join(re.findall(r'^theorem (\\w+)', f.read_text(), re.M)))\n"
             "g = torch.Generator().manual_seed(0); e = torch.randn(10000, 64, generator=g)\n"
             "c = e.mean(1, keepdim=True) * torch.rand(10000, 1, generator=g) * 2\n"
             "print('pooled_gain_le_dc holds on 10k random fields:', bool(((e**2).sum(1) - ((e - c)**2).sum(1) <= e.sum(1)**2 / 64 + 1e-4).all()))\n"
             "Q = torch.linalg.qr(torch.randn(64, 64, generator=g))[0]; a = torch.randn(10000, 8, generator=g)\n"
             "coef = e @ Q; corr = torch.zeros_like(coef); corr[:, :8] = a\n"
             "print('gain_le_ceiling holds:', bool(((e**2).sum(1) - ((coef - corr)**2).sum(1) <= (coef[:, :8]**2).sum(1) + 1e-4).all()))"),
        md("## 3. Where every benchmark stands against its state of the art"),
        code("display(nbviz.sota_table())"),
        md("The matched-control audit (pooled register in the neck, fine-tuned, three seeds; `runs/judge/scorecard.json`):"),
        code("display(nbviz.audit_rows().round(4))"),
        md("## 4. Per-benchmark interpretation (allocation study, frozen trunks)\n\n"
           "For each row: the utilization ceilings of each allocation rule; where the 30 atoms go on one test window and at which "
           "scales; the strategy × core table (test gain, captured share, efficiency = captured / U); qubits vs utilization; "
           "the trained register's corrections against the true residual; the controls; and a look inside the trained register "
           "(learned blockade scale and drives, which observables it uses, how much of the correction it carries, and how "
           "entangled it is on real inputs)."),
    ]
    for b, spec in ROWS.items():
        cells += row_cells(b, spec)
        if b == "turbulent_radiative_layer_2D":
            cells += [
                md("**TRL2D spectral evidence (fine-tuned audit arms):** where the released checkpoint's error lives by wavenumber shell, "
                   "and the register's effect per shell."),
                code("d = json.loads((ROOT / 'runs/spectral/trl2d_bands.json').read_text())\n"
                     "E = np.array(d['arms']['official']['E_err_k']); k = np.arange(len(E))\n"
                     "shells = {'0 (DC)': k == 0, '1-4': (k >= 1) & (k <= 4), '5-16': (k >= 5) & (k <= 16), '17-32': (k >= 17) & (k <= 32), '>=33': k >= 33}\n"
                     "def share(arm): E = np.array(d['arms'][arm]['E_err_k']); return {s: E[msk].sum() for s, msk in shells.items()}\n"
                     "base = {s: np.mean([share(f'vanilla_s{x}')[s] for x in (1337, 1338, 1339)]) for s in shells}\n"
                     "tab = pd.DataFrame({'error share (released)': {s: E[msk].sum() / E.sum() for s, msk in shells.items()},\n"
                     "    'register vs vanilla %': {s: 100 * (np.mean([share(f'mr30ryd_s{x}')[s] for x in (1337, 1338, 1339)]) / base[s] - 1) for s in shells},\n"
                     "    'MLP vs vanilla %': {s: 100 * (np.mean([share(f'mlp30_s{x}')[s] for x in (1337, 1338, 1339)]) / base[s] - 1) for s in shells}})\n"
                     "display(tab.round(3))\n"
                     "print('TD fine-tuning, TRL2D rollout VRMSE@13:30 (3 seeds): register 0.9528 | trunk 1.0301 | matched MLP 1.0322 | no-interaction 1.0156 | mean-field 1.0495')"),
            ]
    cells += [
        md("## 4b. Persistence, the band gate and the analog-digital router (docs 35 §8, 36)\n\n"
           "**Evaluation error first.** On the Well rows the benchmark's mean VRMSE is heavy-tailed. Persistence "
           "($\\hat y_{t+1}=x_t$) is computed on the full official TEST split, per window, next to every trunk CSV. "
           "**Digital part (classical).** A per-window, per-field, per-band gate decides how much of the trunk's increment over "
           "persistence to keep: $\\alpha=1$ is the trunk and $\\alpha=0$ is persistence. The gate is exact from cached band "
           "statistics, and Lean proves that the per-band oracle gate is no worse than either anchor (`BandGate.gate_opt_le`). "
           "**Analog part (qubits).** The residual after the gate is written in band-wise POD modes. A learned router sends 30 "
           "of them per window to the register in a learned x, y, z / tether configuration, and the classical head takes the rest. "
           "Per band, captured ≤ span ≤ full holds for every window (`gain_le_ceiling`, `band_span_le_energy`)."),
        code("P = {r: json.loads((ROOT / 'runs/persistence' / f'{r}_summary.json').read_text()) for r in\n"
             "     ('rayleigh_benard', 'shear_flow', 'turbulent_radiative_layer_2D', 'MHD_64') if (ROOT / 'runs/persistence' / f'{r}_summary.json').exists()}\n"
             "rows = []\n"
             "for r, d in P.items():\n"
             "    for k, v in d['trunks'].items():\n"
             "        if 'trunk_mean' in v: rows.append({'row': r, 'trunk': k.split('/')[-1].replace('_test.csv', '').replace('test_', ''), 'persistence': d['persistence_mean'],\n"
             "                                           'trunk mean': v['trunk_mean'], 'trunk median': v['trunk_median'], 'trunk wins windows': v['trunk_beats_persistence_share'],\n"
             "                                           'oracle min(trunk, persistence)': v['oracle_min_mean']})\n"
             "display(pd.DataFrame(rows).round(4))"),
        md("The gated stacks (full TEST where marked; mean ± sample SD over 3 seeds). `none` = gate only; every arm shares the "
           "identical gate, so arm differences are the co-processor's. Auto-generated from `runs/alloc/adrc_summary.json`."),
        code("S = json.loads((ROOT / 'runs/alloc/adrc_summary.json').read_text())\n"
             "rows = []\n"
             "for r in S:\n"
             "    for arm, a in r['arms'].items():\n"
             "        rows.append({'cache': r['row'], 'gate': r['gate'], 'arm': arm, 'test': a['mean'], 'sd': a['sd'], 'anchor': r['anchor_test'],\n"
             "                     'trunk0': r['trunk0_test'], 'persistence': r['persistence_test']})\n"
             "T = pd.DataFrame(rows); display(T[T.cache.str.contains('gate|fb4|ft_')].round(4))"),
        code("fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))\n"
             "for ax, key in zip(axes, [k for k in ('shear_flow__gate_mr30ryd_s1337', 'rayleigh_benard__gate_mr30ryd_s1337', 'pelzer__fb4')]):\n"
             "    r = next((x for x in S if x['row'] == key), None)\n"
             "    if r is None or 'adrc_ryd' not in r['arms'] or not r['arms']['adrc_ryd']['bands']: ax.set_title(key + ' (pending)'); continue\n"
             "    b = r['arms']['adrc_ryd']['bands']; x = np.arange(len(b['full']))\n"
             "    ax.bar(x - 0.3, b['full'], 0.2, label='full$_b$'); ax.bar(x - 0.1, b['span'], 0.2, label='span$_b$')\n"
             "    ax.bar(x + 0.1, b['captured_qubit'], 0.2, label='captured by qubits'); ax.bar(x + 0.3, b['captured_classical'], 0.2, label='captured classically')\n"
             "    ax2 = ax.twinx(); ax2.plot(x, b['qubit_slot_share'], 'k.-'); ax2.set_ylabel('qubit slot share')\n"
             "    ax.set_xticks(x, [f'b{i}' for i in x]); ax.set_title(key, fontsize=9)\n"
             "axes[0].legend(fontsize=7); plt.tight_layout(); plt.show()"),
        md("**Reading.** The gate is the digital decision: which part of each field, band by band, the classical world model or "
           "persistence carries. The register only sees the analog remainder. Its share per band shows where the router "
           "found predictable analog structure. A quantum-specific claim still needs the register to beat the classical GNN "
           "on the same learned graph (doc 35 §6); otherwise the gain belongs to routing, configuration and the gate."),
    ]
    cells += [
        md("## 5. Benchmarks outside the allocation study\n\n"
           "### 5.1 Fusion Equilibrium Challenge (Sophelio, DIII-D)\n"
           "**Backbone:** residual MLP over coil currents at 14 time lags and Thomson features. **Neck:** latent coupling or a "
           "spectral-routed coupling that writes only PCA components k ≥ 8 of the flux (register or matched MLP). **Head:** PCA-64 "
           "flux decoder + scalar heads. Interpretation: routing the coupling into the fine flux band lifts VALID S by +0.0012 to "
           "+0.0015 with **either** core; the ensemble selected on VALID (with MLP-coupled members) scores TEST 0.9910, 0.0022 below "
           "the leader, and the register-coupled selection is lower (0.9905)."),
        code("s = (ROOT / 'runs/sophelio_sota/status').read_text()\n"
             "for key in ('CONFIRM_DONE', 'SOPHELIO_TEST_DONE'):\n"
             "    line = [l for l in s.splitlines() if l.startswith(key)][-1]\n"
             "    for part in re.findall(r'\\[(?:confirm|test)\\][^\\[]+', line): print(part.strip()[:160])"),
        md("### 5.2 TokaMark 1-1 (MAST), TORAX-Bench and the heat plume in the audit\n"
           "**TokaMark:** TokaMind-Base-v2 transformer backbone; the register wraps each modality head's CLS input (a vector "
           "latent, so no pooling limit applies); register 0.1501 vs MLP 0.1504 vs trunk 0.1509; our ensemble 0.1454 is below the "
           "0.140 leader. **TORAX-Bench:** a 1-D spectral net with the register in its `mid` slot; the tuned MLP adapter is best "
           "(0.0141) and the register worst (0.0168). Both rows are in the audit table of §3."),
        md("### 5.3 TDC ADMET: the 'digital' control (22 endpoints)\n"
           "**Backbone:** CheMeleon D-MPNN foundation model; **neck:** register (or MLP) after mean aggregation of atom "
           "embeddings; **head:** property FFN, then a VALID-selected stack with MapLight(+GNN) CatBoost. Molecules are discrete "
           "graphs; the thesis predicts no register benefit here, and there is none."),
        code("rows = []\n"
             "for f in sorted((ROOT / 'runs/tdc_chemeleon').glob('*_mr30ryd.json')):\n"
             "    ep = f.stem[:-len('_mr30ryd')]; q = json.loads(f.read_text()); mm = ROOT / f'runs/tdc_chemeleon/{ep}_mlp30.json'\n"
             "    if not mm.exists(): continue\n"
             "    m_ = json.loads(mm.read_text()); sgn = 1 if q['higher_is_better'] else -1\n"
             "    rows.append({'endpoint': ep, 'metric': q['metric'], 'register': q['test_mean'], 'matched MLP': m_['test_mean'], 'register better by': sgn * (q['test_mean'] - m_['test_mean'])})\n"
             "t = pd.DataFrame(rows).set_index('endpoint'); print(f\"register better on {(t['register better by'] > 0).sum()} / {len(t)} endpoints\")\n"
             "t['register better by'].sort_values().plot.barh(figsize=(6, 5), color=np.where(t['register better by'].sort_values() > 0, 'C2', 'C3'))\n"
             "plt.axvline(0, color='k', lw=0.7); plt.title('TDC: register minus matched MLP (metric units, + = register better)', fontsize=8); plt.show()\n"
             "print((ROOT / 'runs/tdc_ensemble/stack_table.txt').read_text()[:2200])"),
        md("## 6. Analog vs digital: does the register's inductive bias suit analog maps?\n\n"
           "Controlled probe (registered before running): groups of 10 atoms, oscillatory targets $\\sin(\\omega a)\\cos(\\omega b)$ "
           "(analog) and a sign-product target (digital); test $R^2$, three seeds. Rule: an analog edge needs the register to beat "
           "its matched MLP on 3/3 seeds at ≥3 of 5 frequencies. Starred arms (linear-angle register, sine-MLP) are exploratory."),
        code("P = nbviz.probe_frame(); display(P)\n"
             "ax = P.loc[[i for i in P.index if i.startswith('analog')]].rename(index=lambda s: float(s.split('=')[1])).sort_index().plot(marker='o', figsize=(6, 3.4))\n"
             "ax.set(xscale='log', xlabel='frequency ω', ylabel='test R²', title='analog targets'); plt.show()"),
        md("**Interpretation.** The register's function class is smooth: it cannot draw thresholds (digital $R^2$ 0.07 vs 0.72–0.95 "
           "for MLPs), which supports sending digital structure to classical cores. But it is not better on smooth oscillatory maps: "
           "it wins only at ω = 2, trails at ω ≤ 1 and ω = 4, and every core fails at ω ≥ 8. The registered analog-edge rule is not met."),
        md("## 7. Synthesis"),
        code("R = analyze.load(); fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
             "for i, b in enumerate(nbviz.NAMES):\n"
             "    for s in analyze.STRATS:\n"
             "        v = R[(b, s, 'ryd', 30)]\n"
             "        if v: ax[0].scatter(analyze.mean([x['test_U'] for x in v]), max(3e-5, analyze.mean([x['test_captured'] for x in v])), color=f'C{i}', s=18)\n"
             "    ax[0].scatter([], [], color=f'C{i}', label=nbviz.NAMES[b])\n"
             "ax[0].plot([1e-4, 1], [1e-4, 1], 'k--', lw=0.8); ax[0].set(xscale='log', yscale='log', xlabel='utilization ceiling U', ylabel='captured share', title='captured <= U (Lean); rows differ by predictability')\n"
             "ax[0].legend(fontsize=6); nbviz.forest(ax[1]); plt.tight_layout(); plt.show()\n"
             "print(open(ROOT / 'runs/alloc/summary.md').read().split('# H_U')[1][:1500])"),
        md("**What the model does, benchmark by benchmark, in one sentence each.** A frozen SoTA trunk leaves an error that is "
           "either *predictable* (systematic bias: convection onset, sim-to-real FNO bias, a data-limited plume model, miscalibrated "
           "released checkpoints) or *not* (fine-scale turbulence of a well-trained trunk). Where it is predictable, placing the atoms "
           "on the right tiles turns a fixed budget of 30 atoms into gains of 1–36%, and more atoms keep paying; the pooled neck "
           "insertion cannot, whatever its size, exactly as the Lean bounds say. Where it is not, no allocation and no core helps. "
           "In every row the Rydberg register does this job as well as, and no better than, a parameter-matched classical core; "
           "its one register-specific effect remains the training-mediated long-horizon gain on TRL2D. **Utilization, not qubit "
           "count, decides the hybrid gain.**"),
    ]
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    print(f"wrote {NB}")


if __name__ == "__main__":
    build()
