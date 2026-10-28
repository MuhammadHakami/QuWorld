---
title: Novelty / duplication audit + reviewer-concern survey (WM-PAI freeze, Sep 6 2026)
date: 2026-09-06
project: quworld
tags: [novelty, audit, reviewers, related-work, iclr]
status: applied to main4p.tex FREEZE v3 (before-deadline patches); ICLR-version items open
---

Source: 47-agent workflow (claim-by-claim sweep, 30 OpenReview forums mined, neighbor deep-reads, 2x adversarial verification per hit, AC synthesis). Headline: nothing scooped; two overstated claims and one wrong neighbor characterization were corrected in the paper.

# NOVELTY + REVIEWER-READINESS SYNTHESIS — main4p.tex (2026-09-06)

File audited: /home/mohhakami/Documents/paper/quworld/paper/manuscript/main4p.tex (612 lines; compiled main4p.pdf = 11 pages, body ends at bottom of p.8, references p.9–11). Bib: /home/mohhakami/Documents/paper/quworld/paper/manuscript/references.bib. All arXiv metadata below was fetched live from export.arxiv.org on 2026-09-06 (raw dump: /tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad/arxiv_meta.txt).

Headline: nothing is scooped. Two claims are stated too strongly relative to prior art that the manuscript does not cite (C2's "each ingredient exists in isolation… the composite… does not" is contradicted by ReCon; C6's division-of-labor framing is prior art in the Coveney QIML line), and one neighbor characterization is factually wrong (Lumbreras). Everything else is cite-and-differentiate.

---

## 1. NOVELTY VERDICT PER CLAIM

| Claim | Verdict | Deciding evidence |
|---|---|---|
| C1 Checkpoint surgery vs published number (l.187–195) | novel-with-caveat | Sweep found zero co-occurrence of The Well / CNextU-net / any released SciML checkpoint with a quantum layer. But "controlled insertion of a PQC into a released pretrained backbone with a parameter-matched bottleneck control" now exists: Levy et al. 2608.22475 (Aug 23 2026; ImageNet ResNet-50 layer3, 4 seeds, honest null). LEGO 2601.21780 = frozen backbone + VQC head. Composite (official benchmark checkpoint + published-number target + paired per-window Wilcoxon over 873 windows + kill rules) is unoccupied. Paper already disclaims architecture; keep. Cite Levy. |
| C2 Input-conditioned analog Rydberg ansatz, trained geometry (l.196–205) | adjacent-prior-art-MUST-CITE and MUST-REWORD | (a) ReCon 2408.13389 (already cited as silver2024recon) co-trains 2D positions + per-site couplings h_i + global drive in ONE model — so "each ingredient exists in isolation" is false as written. (b) ResQ 2506.21537 encodes inputs as per-atom detunings and ran on Aquila. (c) de Keijzer et al., npj QI 2025 (2504.06702) introduce positions-as-ansatz-parameters and argue position gradients FAIL under the divergent r^-6 term — uncited; our l.244 clamp at 0.5 R_b is the rebuttal but is never framed as one. (d) Djellabi 2509.09421 and Sotirov 2606.11000 put inputs into detunings and positions (fixed maps, no gradients). What survives: learned detuning map from a trunk latent + autodiff-trained geometry + 10-atom registers + embedding in a predictive world model. Also: ReCon/ResQ ran on real Aquila hardware; we are simulation-only — do not claim realizability novelty over them. |
| C3 Pre-registered, matched, kill-rule protocol (l.206–215) | novel-with-caveat | Zero hits for pre-registration/kill rules in QML. But (i) matched-control attribution nulls with multiplicity correction are contemporaneous: Chen 2608.06846 (2x2 factorial, BH), Gillani 2608.18155, Herbort 2607.04915 (climate regression, Optuna-tuned null), Levy 2608.22475; (ii) "quantum module stabilizes long-horizon turbulence rollouts" is PRIOR art with an advantage claim: Wang et al. Science Advances 12(16) 2026 (2507.19861, v1 Jul 2025) and Xue et al. 2607.21688 — against over-parameterized / unmatched baselines, no seeds, no pre-registration. (iii) Internal: the "significance-tested… on a public PDE benchmark" conjunction is delivered by the TRL2D campaign (§4.1), NOT by the ladder (§4.2 is an internal arena with no formal test). Scope the sentence to the campaign. |
| C4 Fixed-parameter qubit ladder, flat for n≥5 (l.216–220, §4.2) | adjacent-prior-art-must-cite | Saturation-in-qubit-count is already reported: Vyskubov 2604.06007 (IJCNN 2026; MNIST saturates ~Q=5, params grow 2QL, no classical arm), Zabihi 2606.04679 (PINNs on Burgers/Allen-Cahn, 3–7 qubits, non-monotone, single runs, unmatched), LEGO (16–20q insensitivity + Theorem 1: error independent of qubit count once classical block is expressive — this IS our "n-independent perturbation" alternative, with a published theory), Pandey 2607.07978 (co-scaled QRC, flat, 8 paired seeds, Wilcoxon — NOT the cited 2607.09905). What is new: fixed TOTAL parameters, a 0-qubit arm, matched-MLP control, pre-registered interior arms, PDE rollout metric. The finding is not novel; the control design is. Say so. |
| C5 Inference-inert, interactions needed in training (§4.3) | novel-with-caveat | Closest: De Lorenzis 2509.06873 ("entanglement onset improves accuracy yet MPS-simulable" — but there entanglement is load-bearing at inference; opposite to ours), Chen 2603.09789 (Drop-Prior engineers classical inference by design), Keller 2608.20076 (faithful shadow of a trained circuit), Quantum-Train ICLR 2025 (2410.09846), Klement 2601.15046 (qPINN faster convergence = training-dynamics reading, no inertness test). The dissociation (surrogates reproduce to 2e-5 AND pre-registered V_ij=0 ablation loses the effect) is unoccupied. Caveat already stated (Gutzwiller confound). §4.3 currently cites nothing — reviewers will read that as unawareness. |
| C6 Division-of-labor framing (abstract l.83–86, intro l.160–183) | duplicated at framing level; novel as instantiation | Wang et al. Sci Adv 2026 abstract: quantum model "guides the representation of small-scale interactions… fine-scale dynamics" while "a classical autoregressive predictor" generates fields. Wang et al. 2606.13422 literally uses "hybrid quantum-classical division of labour" and "a statistical module within a classical scientific workflow". LEGO formalizes frozen-classical-block + trainable quantum module. Ours differs: in-forward-pass layer, per-input reconfiguration, analog Rydberg, no-advantage stance. "Our motivating conjecture" must become "our instantiation of" a stance others articulate, with citations. |

---

## 2. REVIEWER-CONCERN MATRIX (from 30 OpenReview forums on hybrid QML)

| # | Recurring concern (freq.) | Status in main4p.tex | Fix (exact text + location + offset) |
|---|---|---|---|
| 1 | Attribution: is the quantum part doing anything; do quantum params even move (13/30) | ADDRESSED. Matched MLP + product-state controls (l.211, 255–257); kill rule fired and reported (l.366–368); dequantization suite (l.553–556); geometry moves −0.007 R_b coherently and detuning modes carry 91% variance (Fig. 1 caption l.296–304). | None needed. This is the paper's strongest axis; keep it visible. |
| 2 | Unmatched/missing classical baselines; cherry-picked data (20/30) | ADDRESSED for controls (0.4% param match l.256; tuning-matched l.325–326; official checkpoint anchor). GAP: the arena is "a 3-regime TRL2D subset" (l.449) with no stated selection rule. | Add 6 words at l.449: "(a 3-regime TRL2D subset chosen before any arena run," … Only if the subset was in fact fixed before runs; otherwise leave. Page-neutral. |
| 3 | Hardware realism: noise, shots, device constraints (20/30) | PARTIAL. 128-shot readout in suite (l.554); device-constraint mapping deferred (l.200–201). Missing: any decoherence/control-noise statement; training under noise. | Add to Limitations after "quantum-advantage claim \citep{...}." (l.594): "No noise model beyond 128-shot readout at inference is applied; inference-inertness makes the deployed model noise-agnostic, but training under decoherence is untested." (+1.5 lines). Offset: cut F below. |
| 4 | No advantage pathway / "why quantum" / quantum-inspired (20/30) | ADDRESSED: utilitarian stance stated three times (l.117–118, 591–597). Reviewers explicitly read "5q = 30q" as "no advantage" (qtDNN RVpTy) — paper already concedes (l.474–480). | None. Do not add an advantage pathway; it would contradict the framing. |
| 5 | Gains within seed noise; few seeds; no significance (11/30) | ADDRESSED honestly (l.379–388, Table 1 caption). n=3 seeds is a known workshop limitation and is disclosed (l.587). | None before deadline. ICLR: 5+ seeds, SESOI pre-specified for the "indistinguishable" verdict (Chen 2608.06846 makes exactly this self-critique). |
| 6 | Qubit scaling missing/non-monotone/from 2–3 points (10/30) | ADDRESSED: 5-point ladder, honest "bounds, not resolves" (l.466–480). GAP: not positioned against the four prior saturation reports. | Patch P6 below. |
| 7 | Barren plateaus / trainability not analysed (14/30) | NOT ADDRESSED. "Barren" appears nowhere in the body; the design choice "points above 10 qubits add registers of 10" (Table 2 caption) is never motivated. | Add to Table 2 caption after "not larger registers." (l.504): "(bounding register width at 10 keeps trainability fixed across the ladder \citep{cerezo2025barren})". (+0.5 line; cerezo already in bib.) |
| 8 | Toy datasets (16/30) | ADDRESSED: The Well official benchmark + MHD_64. | None. |
| 9 | Compute cost / wall-clock / simulation overhead not reported (14/30) | NOT ADDRESSED. No timing anywhere; budgets are in steps (l.449–450). Logs in runs/ contain no step-time tokens (checked). | Add one honest sentence to Limitations after "state access." (l.599): "Budgets are matched in optimizer steps, not wall-clock; the exact statevector propagator makes the quantum arm the slowest per step, and per-step cost is reported in the archival version." (+1.5 lines). Do NOT invent a slowdown factor. Offset: cut E. |
| 10 | Mechanism asserted post hoc (11/30) | ADDRESSED: entropy measured, ablation run, confound named (§4.3). GAP: no citations at all in §4.3. | Patch P7. |
| 11 | Ansatz/reproducibility under-specified (12/30) | ADDRESSED: Instantiation paragraph (l.237–260) gives L, gate order, inits, readout, parameter counts, CUDA-Q cross-check. Code withheld for anonymity (unavoidable). | None. |
| 12 | Incremental "A+B" (12/30) | PARTIAL. C1 disclaims architecture (good). C2's unhedged "the composite… does not" is the exposure. | Patch P4. |
| 13 | Overclaiming / framing mismatch (9/30) | Three soft spots: "Contributions, ordered by novelty:" (l.185); abstract "no practical application fits on today's qubit counts" lacks the intro's "whole" (l.80 vs l.162); title "Fidelity versus Qubit Count" for a flat curve (acceptable — it is the measurement). | l.185 -> "Contributions:"; l.80–81 -> "no practical application fits whole on today's qubit counts". Page-neutral. |
| 14 | Unfriendly to general ML readership (9/30) | Dense but self-contained (Rydberg Hamiltonian defined, VRMSE defined, TD defined). | None. |
| 15 | What accepted papers did right: isolate the block, seeds/variance, quantum-only-in-training, hardware/noisy case, transparent limitations | 4 of 5 present. Missing: noisy/hardware case (see #3). | Covered by #3 sentence. |

---

## 3. MUST-ADD CITATIONS (verified BibTeX) + the sentence that cites each

Priority tier A (before deadline): wang2026qiml, wang2026qprior, xue2026qcml, qi2026lego, dekeijzer2025consensus, delorenzis2025qelm, vyskubov2026scaling, zabihi2026hqpinn, klement2026qpinn, chen2026dropprior, levy2026sidecar. Tier B (ICLR): pandey2026qrcdimension, chen2026qeat, herbort2026cloud, gillani2026audit, keller2026shadow.

```bibtex
@article{wang2026qiml,
  title   = {Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos with Practical Quantum Advantage},
  author  = {Wang, Maida and Xue, Xiao and Gao, Mingyang and Coveney, Peter V.},
  journal = {Science Advances},
  volume  = {12},
  number  = {16},
  year    = {2026},
  doi     = {10.1126/sciadv.aec5049}
}

@article{wang2026qprior,
  title   = {Practical Quantum Advantage before Fault Tolerance via Quantum-Informed Machine Learning},
  author  = {Wang, Maida and Xue, Xiao and Chung, Minh and Coveney, Peter V.},
  journal = {arXiv preprint arXiv:2606.13422},
  year    = {2026}
}

@article{xue2026qcml,
  title   = {Explainable Quantum-Compressed Machine Learning for Complex Fluid Flows},
  author  = {Xue, Xiao and Wang, Maida and Gao, Mingyang and Chung, Minh and Coveney, Peter V.},
  journal = {arXiv preprint arXiv:2607.21688},
  year    = {2026}
}

@article{qi2026lego,
  title   = {Quantum {LEGO} Learning: A Modular Design Principle for Hybrid Artificial Intelligence},
  author  = {Qi, Jun and Yang, Chao-Han Huck and Chen, Pin-Yu and Hsieh, Min-Hsiu and Zenil, Hector and Tegner, Jesper},
  journal = {arXiv preprint arXiv:2601.21780},
  year    = {2026}
}

@article{dekeijzer2025consensus,
  title   = {Consensus-Based Qubit Configuration Optimization for Variational Algorithms on Neutral Atom Quantum Systems},
  author  = {de Keijzer, Robert and Visser, Luke and Tse, Oliver and Kokkelmans, Servaas},
  journal = {npj Quantum Information},
  year    = {2025},
  doi     = {10.1038/s41534-025-01132-8}
}

@article{delorenzis2025qelm,
  title   = {Entanglement and Classical Simulability in Quantum Extreme Learning Machines},
  author  = {De Lorenzis, A. and Casado, M. P. and Lo Gullo, N. and Lux, T. and Plastina, F. and Riera, A.},
  journal = {arXiv preprint arXiv:2509.06873},
  year    = {2025}
}

@article{vyskubov2026scaling,
  title   = {Scaling Laws for Hybrid Quantum Neural Networks: Depth, Width, and Quantum-Centric Diagnostics},
  author  = {Vyskubov, Danil and Vyskubov, Kirill and Innan, Nouhaila and Shafique, Muhammad},
  journal = {arXiv preprint arXiv:2604.06007},
  year    = {2026}
}

@article{zabihi2026hqpinn,
  title   = {Hybrid Quantum-Classical Physics-Informed Neural Networks for Solving Nonlinear {PDE}s: When and Where Hybridization Is Effective?},
  author  = {Zabihi, Kaveh and Montazeri, Hamid and Suiker, Akke S. J.},
  journal = {arXiv preprint arXiv:2606.04679},
  year    = {2026}
}

@article{klement2026qpinn,
  title   = {Quantum-Enhanced Convergence of Physics-Informed Neural Networks},
  author  = {Klement, Nils and Eyring, Veronika and Schwabe, Mierk},
  journal = {arXiv preprint arXiv:2601.15046},
  year    = {2026}
}

@article{chen2026dropprior,
  title   = {A Hybrid Quantum Circuit Born Machine Framework for Financial Volatility Forecasting: Quantum-Assisted Training and Classical Inference},
  author  = {Chen, Yixiong},
  journal = {arXiv preprint arXiv:2603.09789},
  year    = {2026}
}

@article{levy2026sidecar,
  title   = {Quantum-Inspired Hybrid Neural Networks for Neural Decoding: A Controlled Ablation Study of Learnable Quantum Sidecar Integration},
  author  = {Legziel Levy, Diana and Finkelstein, Menachem and Chin, Peter and Vaadia, Eilon and Cohen, Sarel},
  journal = {arXiv preprint arXiv:2608.22475},
  year    = {2026}
}

@article{pandey2026qrcdimension,
  title   = {A Quantum Reservoir Architecture for Chaotic Forecasting and a Test of Whether Its High Dimension Helps},
  author  = {Pandey, Tushar},
  journal = {arXiv preprint arXiv:2607.07978},
  year    = {2026}
}

@article{chen2026qeat,
  title   = {Investigating Quantum-Embedded Transformers on Classical Datasets for Cross-Modality Classification},
  author  = {Chen, Hao-Yuan},
  journal = {arXiv preprint arXiv:2608.06846},
  year    = {2026}
}

@article{herbort2026cloud,
  title   = {How Hard Is Quantum Advantage? A Cloud Microphysics Stress Test for Variational Quantum Models},
  author  = {Herbort, Felix and Sarauer, Ellen and Ohl de Mello, Daniel and Christiansen, Paul and Hien, Steffen and Br{\"u}gmann, Cedric and Jaksch, Dieter and Eyring, Veronika and Kiffner, Martin and Schwabe, Mierk},
  journal = {arXiv preprint arXiv:2607.04915},
  year    = {2026}
}

@article{gillani2026audit,
  title   = {How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection},
  author  = {Gillani, Syeda Anshrah and Baig, Mirza Samad Ahmed and Shah, Shahid Munir and Ali, Asher and Siddiqui, Hamzah},
  journal = {arXiv preprint arXiv:2608.18155},
  year    = {2026}
}

@article{keller2026shadow,
  title   = {Shadow Models of a Quantum Model for Cloud Cover and the Influence of Finite Sampling Noise},
  author  = {Keller, Hedwig and Schwabe, Mierk and Eyring, Veronika},
  journal = {arXiv preprint arXiv:2608.20076},
  year    = {2026}
}
```

Verification notes: 2509.06873 has NO journal-ref on arXiv (the "Phys. Rev. Applied 2026" attribution in the sweep is unconfirmed — cite as preprint). 2604.06007's arXiv comment says "Accepted at IJCNN 2026" (no proceedings metadata; preprint form is safe). de Keijzer volume/pages not in the arXiv record; DOI is verified, do not guess a volume. Note that plainnat PRINTS the `note` field — never put commentary in `note`.

Sentences that cite them (exact replacement text):

P5 — intro l.149–154, replace the sentence beginning "To our knowledge, neither literature has produced" with:
> "To our knowledge, neither literature has produced a matched-control win on a public physical-simulation benchmark: quantum-informed priors report rollout gains on turbulence, but as offline-trained regularizers against over-parameterized classical priors and with an advantage claim \citep{wang2026qiml,xue2026qcml}; the closest in-model attempt, a 5-qubit mixer in a 3D FNO surrogate on proprietary laser-processing data, beat a larger dense-mixer FNO on mean-error metrics but was matched or beaten by its own parameter-matched MLP-mixer control on all but one metric \citep{papierz2026hqlpfno}; and reservoir studies show apparent quantum edges evaporating under equal-tuning classical baselines \citep{pandey2026qrcbaselines}."

P5b — intro l.170–174, replace "We therefore treat the qubit budget as a scarce resource to be \emph{allocated}:" with:
> "We therefore adopt the division-of-labor stance that quantum-informed workflows \citep{wang2026qprior} and modular frozen-block designs \citep{qi2026lego} also articulate, but instantiate it in the model rather than in the loss: the qubit budget is a scarce resource to be \emph{allocated}:"

P4 — contribution 2, l.201–205, replace "Each ingredient exists in isolation \citep{silver2024recon,chen2026resq,pulserdiff2025}; the composite, embedded as a layer of a predictive world model, does not (\S\ref{sec:system})." with:
> "ReCon co-trains positions, site couplings, and drive derivative-free at four atoms \citep{silver2024recon}, ResQ feeds inputs into per-atom detunings on fixed lattices \citep{chen2026resq}, and PulserDiff differentiates pulses and positions as tooling \citep{pulserdiff2025}; position gradients were argued ineffective under the divergent $r^{-6}$ term \citep{dekeijzer2025consensus}, which we bound by clamping distances (\S\ref{sec:system}). To our knowledge no prior work composes a learned detuning map with autodiff-trained geometry as a layer of a predictive model."

P6 — contribution 3, l.206–207, replace "Qubit-scaling claims exist in looser forms \citep{ionq2026scaling,kornjaca2024aquila}; we contribute, to our knowledge, their first … instantiation on a public PDE benchmark:" with:
> "Qubit-count sweeps exist in looser, non-parameter-matched forms \citep{ionq2026scaling,kornjaca2024aquila,vyskubov2026scaling,zabihi2026hqpinn}, and controlled quantum-sidecar insertions with matched bottleneck controls exist for classification \citep{levy2026sidecar}; we contribute, to our knowledge, the first \emph{pre-registered, tuning-and-parameter-matched, kill-ruled} instantiation on a public PDE benchmark (significance-tested in the TRL2D campaign, \S\ref{sec:campaign}):"

P6b — §4.2 l.474–476, replace "or that the register acts as an $n$-independent perturbation of TD training." with:
> "or that the register acts as an $n$-independent perturbation of TD training --- the reading a frozen-block bound predicts \citep{qi2026lego}."

P7 — §4.3, append after "inference-time quantum computation." (l.564, before \AblationVerdict):
> "Neighbors report pieces of this pattern in other settings --- faster convergence of hybrid PINNs \citep{klement2026qpinn}, quantum-assisted training with classical inference by design \citep{chen2026dropprior}, and reservoirs whose short-range entanglement is load-bearing at inference yet MPS-simulable \citep{delorenzis2025qelm} --- but none tests both inference-inertness and training-side interaction necessity."

---

## 4. NEIGHBOR CHARACTERIZATIONS THAT MUST BE REWORDED

R1 — Lumbreras (l.154–158) is factually wrong (the ENVIRONMENT is classical; the qutrit is in the MODEL; it is a proof). Replace:
> "Concurrent theory proves that certain classical environments with long-range (irrational-phase) memory admit an exact single-qutrit quantum world model while every finite classical model misaligns the optimal policy \citep{lumbreras2026irreducible}; a fully observed finite-difference PDE is Markov in its state and lies outside that construction, so our contribution is empirical and control-matched. The similarly named QSWM \citep{sun2026qswm} is quantum-inspired (no register) and, against matched controls on cellular automata, finds its edge confined to short horizons --- the opposite profile of ours."

R2 — Papierz (l.150–152): included in P5 above ("on proprietary data… matched or beaten… on all but one metric"). Their Table 4: control wins 8/10 rows, 1 tie, 1 marginal win.

R3 — references.bib l.232 `note = {IonQ; accuracy scaling with qubit count against matched classical baselines}` PRINTS in the PDF and is false (single runs, 886-param PQC vs 296-param SVC, trend reverses at 18q) and contradicts contribution 3. Delete the note. Also delete l.124 `note = {Introduces HQ-LP-FNO}` and l.165 QSWM note (harmless but prints); rename key sun2026qswm -> jiang2026qswm only if you have time to update the one \citep.

R4 — ReCon/"each ingredient exists in isolation" (l.202): replaced by P4.

R5 — C6 "Our motivating conjecture" (abstract l.83, intro l.179): abstract can stay ("Our motivating conjecture is division of labor" is acceptable if the intro credits the neighbors via P5b). Do not touch the abstract for length.

R6 — l.185 "Contributions, ordered by novelty:" -> "Contributions:" (C1 is the least novel of the four by the audit's own matrix; the ordering invites a reviewer to attack the weakest item first).

---

## 5. RANKED, TIME-BOXED PATCH LIST

### BEFORE DEADLINE (<=45 min, text-only, page-neutral)

Line accounting (body must stay at 8 pages; ~2.5 words ≈ 0.1 line at NeurIPS width). Adds total ≈ 12 lines; cuts total ≈ 12.5 lines.

CUTS (verbatim, in order of safety):
- Cut A (l.164–169, ≈4.5 lines): delete "The analogy we draw from signal analysis is that Fourier and multiresolution decompositions separate a field into sparse, effectively discrete structure and continuous spectral content \citep{mallat1989multiresolution,daubechies1992ten}, and neural operators already exploit the spectral side classically \citep{li2021fno}; we do not claim a precise correspondence between this split and quantum hardness." (decorative; its hedge exists only because of the analogy).
- Cut B (l.283–286, ≈2.5 lines): delete "Systematic comparison of configuration strategies --- population-based, reinforcement-learning-driven \citep{zoph2017neural}, and quantum-architecture-search methods \citep{du2022quantum} --- is deferred to the archival version." (verbatim duplicate of l.604–606).
- Cut C (l.273–277, ≈1.5 lines saved): replace "a smoke-scale population search (CMA-ES over register geometries, $\sim$200 evaluations per flow regime; best elite VRMSE@8 1.14--1.31 across three regimes, not competitive with gradient training) exercised an evolutionary configuration path for the classical--quantum fusion." with "a smoke-scale gradient-free search (CMA-ES, $\sim$200 evaluations per regime; best VRMSE@8 1.14--1.31) was not competitive with gradient training." (keeps the de Keijzer-relevant fact).
- Cut E (l.137–139, ≈1.5 lines): delete "; video-physics benchmarks report the same gap for generative world models \citep{motamed2025physicsiq,li2025worldmodelbench}" (tangential).
- Cut F (l.266–268, ≈1.5 lines): delete ", and coherent arrays now reach 6{,}100 atoms \citep{manetsch2025tweezer}; the same Hamiltonian family also supports annealing-style schedules \citep{ebadi2022quantum}" -> end sentence after "\citep{barredo2018synthetic}".
- Cut G (l.185): "Contributions, ordered by novelty:" -> "Contributions:".

ADDS, ranked (do in this order; stop when time runs out — each is independent):
1. [5 min] R3: delete printing bib notes (ionq2026scaling, papierz2026hqlpfno). Zero page cost. Removes a printed falsehood.
2. [5 min] R1: Lumbreras + QSWM sentence (+1.5 lines net, incl. QSWM). Add sun2026qswm cite. Offset: Cut F.
3. [10 min] P4: ReCon/ResQ/de Keijzer reword + "to our knowledge" (+2.5 lines). Add dekeijzer2025consensus bib. Offset: Cut B.
4. [8 min] P5 + P5b: Coveney line + LEGO in intro (+3 lines). Add wang2026qiml, xue2026qcml, wang2026qprior, qi2026lego bibs. Offset: Cut A.
5. [5 min] P6 + P6b: scaling neighbors + Levy in contribution 3; LEGO bound in §4.2; scope "significance-tested" to the campaign (+2 lines). Add vyskubov2026scaling, zabihi2026hqpinn, levy2026sidecar. Offset: Cut C + Cut G.
6. [5 min] P7: §4.3 neighbor sentence (+3 lines). Add klement2026qpinn, chen2026dropprior, delorenzis2025qelm. Offset: Cut E + slack from Cut A.
7. [3 min] Abstract l.80–81: "no practical application fits whole on today's qubit counts". Page-neutral.
8. [2 min] Table 2 caption (l.504) barren-plateau clause (+0.5 line, cerezo already in bib). Offset: remaining slack. Skip if page overflows.
9. [3 min] Limitations noise sentence (#3) and compute-cost sentence (#9) (+3 lines). Only if the recompile shows slack; otherwise defer to ICLR — these are the lowest-risk omissions because inference-inertness and step-matched budgets are already stated.

After edits: recompile, confirm body still ends on p.8 (`pdftotext -f 8 -l 8 main4p.pdf - | tail`, next page must start with "References"), and grep the PDF for "matched classical baselines" to confirm the ionq note is gone.

### ICLR VERSION
- Add tier-B citations (pandey2026qrcdimension next to pandey2026qrcbaselines at l.152 and in §4.2; chen2026qeat, herbort2026cloud, gillani2026audit in the C3 protocol sentence; keller2026shadow + Schreiber/Jerbi surrogate lineage in §4.3) and the audit's older must-cites (Mari 1912.08278 for quantum transfer learning; Freinberger 2601.04732 and 2512.02066 for prior Wilcoxon-tested hybrid comparisons; Quantum-Train 2410.09846).
- Gutzwiller control (interactions kept, entanglement forbidden) to separate entanglement from mean-field nonlinearity in C5.
- Pre-specify a smallest effect of interest for the "indistinguishable" verdict; 5+ seeds; per-trajectory power.
- Public pre-registration timestamp (OSF/Zenodo) — the "pre-registered" title claim is currently verifiable only via a withheld repo.
- Report per-step wall-clock for quantum vs MLP vs bare-trunk arms, and a noisy-simulator (Aquila-calibrated) training run; note ReCon/ResQ already ran on Aquila.
- Run the ladder on the official TRL2D protocol (not the arena) so C3's "significance-tested on a public benchmark" covers the scaling claim; add the frozen-backbone arm (which moves you closer to LEGO — cite it there).
- Novelty-delta table vs QuFeX / HQ-UNet / PHQFNO / HQ-LP-FNO / Levy / LEGO / QIML.
- Cite WaveLiT (sankaran2026wavelit, TRL2D 0.1308 non-official) beside "best official baseline" (l.132); cite cui2026physnative for the adjacent "physically native world models" framing.
- Bib hygiene: keys silver2024recon and chen2026resq do not match first authors (DiBrita); rename to dibrita2024recon / dibrita2025resq.

Residual risk: Semantic Scholar was unreachable during the sweep and Sep 1–6 arXiv listings were only partially covered; a same-week workshop preprint could exist. The claims as patched above are all hedged with "to our knowledge", which is the correct exposure for that risk.