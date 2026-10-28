# QUOTA conditional scaling formalization

Lean 4.19.0, mathlib v4.19.0; dependencies pinned by `lake-manifest.json`.

```bash
cd formal
lake update          # first setup; resolves the pinned dependency
lake build
lake env lean Audit.lean
```

If elan is not on PATH, use `$HOME/.elan/bin/lake` for these commands.

`QuotaScaling/Basic.lean` formalizes conditional algebraic scaling properties,
an inverse-resource allocation bound, and independent-register amplitude-entry
accounting. These are consequences of definitions and explicit assumptions,
not proofs of a universal accuracy law, a fitted exponent, or quantum advantage.
See `../24-lean-hybrid-scaling.md` for the empirical model and experiment design.

The real-valued proofs use mathlib's standard logical foundations (reported by
`#print axioms`). They must not depend on `sorryAx` or new project axioms.

## Bounded signal routing

`QuotaScaling/Routing.lean` adds five theorems in namespace `QuotaRouting`:
componentwise correction bound, zero gate, zero allocation, per-token squared
energy bound, and at most30 qubits from at most3 independent10-qubit registers.
For relative scale0.1 the squared correction norm is at most0.01 of the input
squared norm. The RMS identity and absolute bounds on gate, direction and
probability are explicit hypotheses. Distinct-location dispatch is checked in
Python contract tests; its implementation is not formally verified by Lean.

The proof concerns exact real arithmetic, not PyTorch floating-point execution,
physical-device feasibility, global optimality, downstream rollout stability or
prediction accuracy. All20 scaling/routing theorems build; the axiom audit uses
only standard Lean/mathlib axioms. Evidence and source hashes:
`../runs/scaling/lean_routing_verification.json` and
`../runs/scaling/lean_axioms_routing.txt`.

## Competition decision bounds

`QuotaScaling/Competition.lean` formalizes exact finite-sample threshold,
mixture, and routing identities. `CompetitionRobust.lean` makes two limits
explicit. If a fraction `η` of hidden Weak Lensing anomalies is outside the
evaluated proxy families, a `-ε` lower bound on those families gives only
`-(1-η)ε-η` overall when the unknown family can lose one full metric unit.
An absolute proxy-family minimum `m` gives only `(1-η)m` overall if the
unrepresented metric could be zero. Both bounds require an external upper
bound on `η` to yield a numerical hidden-set guarantee.
For one raw Fusion R² target, two evaluation subsets combine through their
SSE reductions divided by pooled target SST, not shot-count weights. When
subset SSTs are centred separately, pooled SST also contains a between-group
mean term. The competition clips R² at zero, so the raw identity must not be
applied through a clipping threshold. Neither theorem estimates hidden-set
distribution shift or makes a validation improvement a private-set guarantee.
