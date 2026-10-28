"""Doc 35 §28c: quantum-to-classical handoff under hardware noise (M4) and Shapley credit over the co-processor's players.

Hardware readout model (eval only, same for the register and its matched MLP, whose tanh outputs are read as
expectation values): X/Y contrast loss (dephasing), asymmetric state-preparation-and-measurement flips
P(read 1 | 0) = e01, P(read 0 | 1) = e10, then `shots` samples per basis setting.

Handoff: the routed correction is c = s (W_q phi + t A o); t = 1 always trusts the readout, t = 0 hands every routed mode
back to the classical path. `const` fits one t on VALID; `snr` sets t per input and per register group from the shots
themselves, t = sigmoid(a + b log SNR), SNR = mean o^2 / mean (1 - o^2) / N, with (a, b) fitted on VALID.

Credit: exact Shapley values of three players on the TEST error reduction (removal, no retraining): the band gate
(off: its input-independent fit alpha_fit), the classical head (off: W_cls = W_q = 0) and the core readout (off: A o = 0).
"""
from __future__ import annotations

import itertools
import math

import torch

PROFILES = {                                   # name: (shots, (e01, e10), X/Y contrast)
    "shots1k": (1000, (0.0, 0.0), 1.0),
    "hw1k": (1000, (0.01, 0.03), 0.9),
    "hw100": (100, (0.01, 0.03), 0.9),
    "hw10": (10, (0.01, 0.03), 0.9),          # doc 35 §28c addendum: added after the VALID smoke, before any TEST run
}
CONST_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SNR_GRID = [(a, b) for a in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0) for b in (0.5, 1.0, 2.0)]
PLAYERS = ("gate", "classical", "core")


def readout_noise(out: torch.Tensor, shots: int, spam: tuple[float, float], contrast: float) -> torch.Tensor:
    """out (..., >=3) exact <Z>, <X>, <Y> -> the value estimated from `shots` noisy measurements per basis."""
    o = out.clone()
    o[..., 1:3] = o[..., 1:3] * contrast
    e01, e10 = spam
    p1 = ((1 - o) / 2).clamp(0, 1)                                   # probability of reading bit 1 in that basis
    p1 = p1 * (1 - e10) + (1 - p1) * e01
    if shots:
        p1 = torch.distributions.Binomial(total_count=shots, probs=p1.clamp(0, 1)).sample() / shots
    return 1 - 2 * p1


def group_trust(o: torch.Tensor, ng: int, shots: int, a: float, b: float) -> torch.Tensor:
    """o (B, K, R) noisy readouts -> t (B, K, 1), one trust per register group of ng atoms."""
    B, K, R = o.shape
    og = o.reshape(B, K // ng, ng * R)
    var = ((1 - og.square()).clamp_min(1e-6) / max(shots, 1)).mean(-1)
    snr = og.square().mean(-1) / var
    t = torch.sigmoid(a + b * torch.log(snr.clamp_min(1e-6)))
    return t.repeat_interleave(ng, 1)[..., None]


def shapley(v: dict[frozenset, float], players=PLAYERS) -> dict[str, float]:
    """Exact Shapley values from the value of every coalition (v[frozenset()] is the reference)."""
    n, out = len(players), {}
    for p in players:
        rest = [q for q in players if q != p]
        phi = 0.0
        for k in range(n):
            for S in itertools.combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi += w * (v[frozenset(S) | {p}] - v[frozenset(S)])
        out[p] = phi
    return out


def credit_report(model, gate, va, te, evaluate, set_gate_off) -> dict:
    """Shapley split on TEST (noiseless), then the handoff study per noise profile (fits on VALID, TEST 3 draws)."""
    values = {}
    for k in range(len(PLAYERS) + 1):
        for S in itertools.combinations(PLAYERS, k):
            S = frozenset(S)
            model.keep_cls = 1.0 if "classical" in S else 0.0
            model.keep_q = 1.0 if "core" in S else 0.0
            set_gate_off(gate is not None and "gate" not in S)
            err = evaluate(te)[0]
            values[S] = err
    model.keep_cls, model.keep_q = 1.0, 1.0; set_gate_off(False)
    base = values[frozenset()]
    gain = {S: base - e for S, e in values.items()}
    res = {"coalition_test": {"+".join(sorted(S)) or "none": e for S, e in values.items()},
           "shapley": shapley(gain), "total_gain": gain[frozenset(PLAYERS)]}
    exact_test = values[frozenset(PLAYERS)]
    res["handoff"] = {}
    no_core = frozenset({"gate", "classical"} if gate is not None else {"classical"})
    for name, (shots, spam, contrast) in PROFILES.items():
        model.core.hw = {"shots": shots, "spam": spam, "contrast": contrast}
        model.trust = None
        always = _mean_draws(evaluate, te, (101, 102, 103))
        consts = {}
        for t in CONST_GRID:
            model.trust = {"kind": "const", "t": t}
            consts[t] = _mean_draws(evaluate, va, (11,))
        t_best = min(consts, key=consts.get)
        model.trust = {"kind": "const", "t": t_best}
        const_test = _mean_draws(evaluate, te, (101, 102, 103))
        snrs = {}
        for a, b in SNR_GRID:
            model.trust = {"kind": "snr", "a": a, "b": b, "shots": shots}
            snrs[(a, b)] = _mean_draws(evaluate, va, (11,))
        a, b = min(snrs, key=snrs.get)
        model.trust = {"kind": "snr", "a": a, "b": b, "shots": shots}
        snr_test = _mean_draws(evaluate, te, (101, 102, 103))
        model.trust = None; model.core.hw = None
        res["handoff"][name] = {"exact_test": exact_test, "always_core": always, "classical_only": values[no_core],
                                "const": {"t": t_best, "valid": {str(k): v for k, v in consts.items()}, "test": const_test},
                                "snr": {"a": a, "b": b, "test": snr_test},
                                "recovered_const": _recovered(exact_test, always, const_test),
                                "recovered_snr": _recovered(exact_test, always, snr_test)}
    return res


def _mean_draws(evaluate, d, seeds) -> float:
    errs = []
    for sd in seeds:
        torch.manual_seed(sd); errs.append(evaluate(d)[0])
    return sum(errs) / len(errs)


def trust_factor(trust: dict | None, o: torch.Tensor, ng: int):
    if not trust:
        return 1.0
    if trust["kind"] == "const":
        return trust["t"]
    return group_trust(o, ng, trust["shots"], trust["a"], trust["b"])


def _recovered(exact, always, handed):
    lost = always - exact
    return (always - handed) / lost if lost > 1e-12 else None
