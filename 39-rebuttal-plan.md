# 39 — ICLR 2027 rebuttal plan (OpenReview GNJXn14y6P; submitted = commit 88f2c7f)

Built from the adversarial audit by the GPT Codex reviewer (tmux qucodex, bridge name `sol`, messages
.bridge/msgs/20260927-011429*, 011622*, 011729*, 011810*) and Claude's checks against the submitted source. Rebuttal
stance (agreed): lead with the matched-control negative result on the quantum core and the robust classical allocation
evidence; do not rescue a quantum edge from small p-values that fail stronger controls.

| # | Severity | Issue (submitted version) | Evidence | Fix (revision text / run) | Status |
|---|---|---|---|---|---|
| 1 | major | "Strictly proper" energy score; both terms divided by D(y) (window target variance) = observation-weighted, minimised by a tilted law | paper/alloc/fgn.py field_err/field_dist | Call it variance-normalised; proper ES with fixed TRAIN scale (fgn.py ref=True); rerun FGN claim | wording applied 01:38 (main + app O); FGN result moved to appendix; §30 running |
| 2 | major | Main text "beats published SoTA on shear flow" vs app:sota "no comparison on shear/MHD"; persistence beats 0.808/0.657 | main_iclr.tex:298 vs :828 (88f2c7f) | [SUPERSEDED by round 2: published = PDE-FM/Flowers/EddyFormer; shear lower, RB numerically close, TRL2D 1% lower same protocol, MHD above, RealPDE only vs our reproduced U-Net] | applied 01:38 (revision tex, 9 pp) |
| 3 | major | Allocation headline (26%/36%) attributed to qubits; shared linear corrector gets +26.26% vs register +26.29% (shear), MLP +36.55% > register +36.25% (RB) | runs/alloc/summary.md | Abstract: gains belong to routing with any core; qubit-specific effect = ties | applied 01:38 (revision tex, 9 pp) |
| 4 | major | "Hybrid scaling law": U(K) from TEST labels (adrc.py oracle), closed form fitted post hoc; ranges [eta_prev U, U] are weak tests | fig_hybrid_scaling.py, adrc.py oracle block | Wording: descriptive coverage fit + registered bounds; add VALID-only U(K) and a narrow prospective VRMSE forecast (register §31) | wording applied 01:38; §31 registered (prospective) |
| 5 | major | Adaptive reuse of TEST across amendments (each run scored once, but later amendments follow TEST results) | doc 35 history | Label confirmatory vs exploratory claims; state limitation; no new claim from post-hoc TEST reuse | applied 01:38 (revision tex, 9 pp) |
| 6 | minor | Pooled-register proposition is about the insertion latent; "sees only the spatial mean of the error" and "ceiling" overstate for the output | main_iclr.tex:300, :325 | Scope to the latent; call 2.9% a measured DC share, not a ceiling | applied 01:38 (revision tex, 9 pp) |
| 7 | minor | Figure 1 arrow c_m = <y - yhat, phi_m> looks like an inference input (it is a training/oracle target; no leakage found in Data.phi, route_sup, alpha_fit) | adrc.py, adrc_cache.py | Redraw: dashed "training target" arrow | applied 01:38 (revision tex, 9 pp) |
| 8 | major | Table 1 RB routed row is a stratified test sample, labelled "test split", no dagger, beside full-test published 0.6566 | main_iclr.tex:248-263; app:hscale | Dagger + caption "stratified test sample"; generated rows carry the flag | applied 01:38 (revision tex, 9 pp) |
| 9 | minor | Abstract per-input 3-D/tether clause vs §22 (fixed 2-D grid ties) | §22 | "set per input; a registered control shows routing, not geometry, carries the gain" | applied 01:38 (revision tex, 9 pp) |
| 10 | minor | Pasqal mapping overclaim; "runs past exact simulation on hundreds of atoms" | main_iclr.tex:98, :497 | Constraints-qualified wording; future work | applied 01:38 (revision tex, 9 pp) |
| 11 | info | Post-submission live caption "rows are added as ... finish" | current tex only | Rewrite caption (not in submitted PDF) | applied 01:38 (revision tex, 9 pp) |
| 13 | major | Missing SoTA: PhysiX (arXiv:2506.17774, Nguyen et al. 2025) reports one-step VRMSE shear 0.0700, RB 0.1470 (its Table 1; C-U-Net 0.8080 / 0.6699); our Table 1 calls 0.8080 / 0.6566 published SoTA and does not cite it | arxiv.org/html/2506.17774 Table 1 (verified 01:22) | Cite PhysiX; published column = 0.0700 / 0.1470; compare full-test numbers only (shear routed 0.0205 full test; RB needs full test, not the stratified sample); note persistence 0.0646 < PhysiX 0.0700 on shear | applied 01:38 (revision tex, 9 pp) |
| 14 | major | Missing SoTA: PDE-FM (Soares et al., arXiv:2511.21861, Table 3, official TEST, no ensemble/TTA): shear 0.0345, RB 0.0415, TRL2D 0.2321 (verified 01:29 from arXiv HTML) | Codex 012700 | Table 1 / gate table / fig_gate / tab:sota published = PDE-FM; shear: ours 0.0207 full test (classical part 0.0236) lower; RB: full-test ensemble 0.0403 vs 0.0415 = parity, not a win; RB routed rows stratified (§ mark, no bold across samples) | applied 01:38 (revision tex, 9 pp) |
| 12 | info | Fusion leaderboard entries under the author's account (not in the paper) | Codabench 947029/947046 | Keep out of the paper during review; disclose only after | done |

Positive evidence for the rebuttal (post-submission, registered): §25 K = 100/200 bounds held (12/12); §26 replication
in progress (shear K = 100 3/3, RB 2/2 so far); §29b fusion 3-D register vs MLP 3/3 seeds (p 0.039 one-sided, 0.078
two-sided: not claimed); §29c MHD physics constraints; §28c Shapley credit (core earns 3.6% of the gain on shear s1).

Surviving contribution (answer to the reviewer attack "mean-pooling algebra + linear corrector gets +26.26% + quantum
core costlier yet tied"): an allocation result, core-agnostic. [SUPERSEDED 01:50: no quantum-specific equivalence result; see below, +-1% fails and no equivalence claim is made.]
A (heterogeneous, as conceded to Codex 01:23; my ">= 10x" and ">= 4x" drafts were both wrong: I had read the TRL2D
pooled row as shear's): routing matches or beats pooling on every row, with a heterogeneous ratio (runs/alloc/summary.md,
lin/mlp/register): shear 7.5/4.1/3.9x (pooled +3.48/+6.38/+6.77% vs routed ~+26.3%), RB 10.5/6.1/5.4x, heat plume
~6-7x, RealPDE FNO 1.16x, TRL2D and MHD ~0 both. Claim: the ratio is large exactly where the gated residual is
spatially structured and predictable; "predictable" must be defined on VALID before TEST (§31) and the per-row
distribution reported, no universal multiplier.
B [SUPERSEDED: no TOST/equivalence claim; final wording 'no consistent quantum superiority detected'] (falsifiable, quantum-specific, negative): given routing, the analog register is equivalent to parameter-matched
classical cores within a stated margin (TOST, e.g. +-2% VRMSE) on every row; Shapley credit of the core is small
(3.6% of the gain on shear s1, §28c). Falsifier: any row outside the margin in either direction.
C (needs §31): which rows benefit is predictable from VALID residual predictability, without test labels.

Codex 01:24 (accepted): pooled vs routed confounds allocation with the output subspace and parameter count
(paper/alloc/coproc.py: PooledCoproc writes a spatially constant C-channel correction, ~8,500 params on shear;
TileCoproc writes selected spatial Haar modes, 5,612 for ryd). The clean allocation comparison fixes TileCoproc and
the output basis and varies only the strategy: register shear static +22.36% vs learned +26.29%; RB static +32.86% vs
proxy +36.25%. Revised claim A: most of the 26%/36% comes from writing non-DC spatial modes; adaptive routing adds ~3-4
points on top; pooled-vs-routed is reported as an output-subspace effect, not "same co-processor, different routing".
§31: every TEST outcome is already known, so no threshold fixed now can be a prospective cross-benchmark test on these
TEST sets; a new holdout (unseen benchmark or fresh split) is required. Codex drafting.

§31 registered verbatim (doc 35): new rows in order active_matter, gray_scott_reaction_diffusion, viscoelastic_instability
(first two with released CNextU-net + usable splits); V1/V2 split of VALID; S = 1 - VRMSE(route_lin)/VRMSE(pooled_lin)
on V2; S >= 0.01 predicts gains in both cores, S <= 0 predicts none, else abstain; falsifiers as stated. To run: download,
caches, checkpoints (days). Claim B: no equivalence claim (+-1% fails); "no consistent quantum superiority detected".
To verify: PDE-FM (Codex: RB ~0.0415) as another missing published number (item 14).

Revision batch 01:38: items 1-11, 13, 14 applied in main_iclr.tex (abstract, contributions, SoTA paragraph, Prop. 1 scope, scaling wording, proper-score wording, Table 1 PDE-FM + stratified mark + resizebox, Fig. 1 dashed training target, new app:claims confirmatory/exploratory table, AI statement discloses the Codex reviewer). fig:fill moved to app:hscale (it was only cited there) to keep the main text at 9 pages. Gemini's 0.0428 'RB full test' is the single 3-D model (ft.log:357); the E-ft ensemble full test is 0.04027 (ft.log:286). Sent to sol for a hostile read.
[SUPERSEDED by doc 35 §32: weights were not saved, so §32 is a locked-recipe retraining plus full-TEST evaluation of every specified arm and seed; exploratory, not a frozen-model rescore.]

Round 2 01:44 (Codex hostile read 20260927-014016 + 013011/013058/013121/013315/013331/014249/014300, all verified): published column = current best per row with citations, verified from primary sources: PDE-FM shear 0.0345 / RB 0.0415; Flowers (arXiv:2603.04430 Table 1, 4->1) TRL2D 0.1930 (conditioned 1->1 Table 8: 0.1907, labelled); EddyFormer (NeurIPS 2025 Table 3) MHD 0.1160 (Flowers 0.1165). Bold marks the best of our models only (published = reference). Main SoTA sentence: shear lower; RB ensemble numerically close; TRL2D 1% lower same protocol with more compute; MHD above; RealPDE dropped from the published-best claim (only our reproduced released U-Net). app:claims: only the first audit round is confirmatory; TRL2D 4th seed, K bounds (U from test labels), 3-D controls reclassified TEST-informed; captured<=U called an algebraic check. Abstract: 'ten scientific and optimization tasks, released checkpoints where available'; router predicts modes from input + forecast; bounds = oracle-coverage diagnostics. Conclusion scoped to DC-only pooled insertion and the insertion latent. app:fgn: lower variance-normalised score, pending §30; dressing wins some controls. well2d_backbone.py docstring fixed.

01:58 K=100 shear (seeds 1-7, verified from result files): register TEST VRMSE lower than the matched MLP on 7/7, but its captured normalised error energy is lower on 4/7 (s1 0.3499 vs 0.3519, s3 0.3475 vs 0.3624, s4 0.3517 vs 0.3529, s5 0.3458 vs 0.3644). The edge is in how the correction spreads across windows (VRMSE = mean of per-window square roots), not in captured energy; the training loss is the energy share. Before any quantum-specific wording: (1) metric-aligned MLP control (§33 B2), (2) GNN / mean-field replay on the register's saved graph (the MLP ignores coordinates, cores.py:140). If §26 passes its rule, the paper says only: 'on shear flow at 100 qubits the register beat a matched MLP that does not see the interaction graph (registered replication, TEST-informed); graph-aware and metric-aligned controls pending'.

02:04 §28c correction (Gemini 020402, verified from coalition_test): Shapley over gate / head / core dilutes the core, because half its coalitions lack the classical head. At the operating point, removing the core from the fitted model raises shear TEST VRMSE by 6.3% (register s1: 0.020572 -> 0.021958), 7.2% (register s2: 0.020707 -> 0.022317) and 14.5% (MLP s1: 0.020886 -> 0.024439). So 'core nearly idle' is wrong; the correct reading: the register's core has a smaller direct effect than the MLP's, while the register arm's gate+head alone is better (0.02196 vs 0.02444), a training interaction that §33a tests. Use the operating-point ablation, not the 3.6% Shapley share, in any rebuttal sentence.

2026-09-27 03:45 SUPERSEDES the 01:58 K=100 note's "The edge is in how the correction spreads across windows ... not in captured energy": use the mechanism-neutral form (register capture lower on 3/5 new seeds; VRMSE and captured share weight fields and windows differently; gate/head co-trained per arm). Also: the §26 edge is against a smaller graph-blind MLP (199 vs 241 trainable) only; §26b's graph-aware GNN (239) beat the register on s4/s5, so the edge is not established against it. Rebuttal must not present §26 as a register-specific edge.
