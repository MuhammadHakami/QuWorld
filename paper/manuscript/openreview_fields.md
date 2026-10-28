# WM-PAI OpenReview submission fields (final, 2026-09-06)

Submit at: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/WM_PAI
(invitation `NeurIPS.cc/2026/Workshop/WM_PAI/-/Submission`)
Due: **2026-09-06 13:59 UTC (16:59 local)** · hard form close 14:29 UTC (17:29 local)
PDF: `paper/manuscript/main4p.pdf` (single file, ~0.5 MB; 11 pages = 8 body + references)
Prerequisite: every listed author needs an existing OpenReview profile.
Required consents on the form: email-sharing checkbox + public-release-on-acceptance checkbox.

## Title
QUOTA: A Few Configurable Qubits Improve Long-Horizon Rollouts of Physical World Models

## Keywords
world models, hybrid quantum-classical, quantum machine learning, neutral atoms, Rydberg Hamiltonian, PDE surrogates, physical simulation, scaling, pre-registration, ablation

## TLDR (<=250 chars)
World models fail at long horizons; we assign only the analog part of the signal to a few configurable neutral-atom qubits. Pre-registered, matched-control rollout gains on The Well; flat from 5 qubits; inductive bias, not advantage.

## Abstract (plain text for the form; <=5000 chars)
World models degrade past the spatial-mean floor over long rollouts, even at foundation scale. We hypothesize that the failure sits in the analog part of the signal: physical fields separate into an easily digitized component and a continuous analog component, so assigning only the analog part to a small configurable quantum register should recover long-horizon accuracy with today's qubit counts. QUOTA (QUantum Offload of The Analog core) implements this as one parametrized world model: a Rydberg register whose per-qubit detunings are conditioned on the input state and whose geometry is an optimized blockade-coupling graph, inside a pretrained classical model that carries the rest. Under a pre-registered protocol with parameter- and tuning-matched controls and kill rules on The Well, the register drops into the official CNextU-net checkpoint without harm and, trained through rollouts, is the only arm that improves the hardest long-horizon window on every seed (test VRMSE 0.9528+/-0.0542, 7.5-7.7% lower than vanilla fine-tuning 1.0301+/-0.0449 and a matched MLP 1.0322+/-0.0259; directional at nine trajectories). The trained register is inference-inert yet depends on its interactions during training, accuracy is flat from five qubits upward, and a 3-D MHD grid (three arms, three seeds) is a null; a ten-benchmark industrial ladder spanning fusion, fission proxies, geothermal heat exchange, and video physics extends the protocol. Thirty simulated qubits are classically simulable: the claim is inductive bias on the hardware that exists, not quantum advantage.

# ICLR 2027 submission fields (draft, 2026-09-19)
## Primary area
foundation or frontier models, including LLMs (user decision 2026-09-19; justified by the Phase-C frozen LingBot-World v2 results)
## Keywords (ICLR form, comma-separated)
world models, foundation models, hybrid quantum-classical machine learning, neutral-atom quantum computing, Rydberg Hamiltonian, parameter-efficient adaptation, physical simulation, PDE surrogates, long-horizon rollout, pre-registered evaluation, scaling with qubit count, inductive bias
## Notes
- Omit "quantum advantage" (not claimed) and "QuWorld" (public repo name; double-blind).
- If limited to ~8: drop "Rydberg Hamiltonian" and "PDE surrogates".
