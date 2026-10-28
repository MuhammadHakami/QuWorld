"""Pre-registered training-side ablation (Phase E follow-up, 2026-08-23):
TD fine-tune identical to the mr30ryd arms EXCEPT the quantum registers are
product-state FROM STEP 0 (blockade energies zeroed for the whole run).

Decides the paper's central mechanism claim:
- advantage vanishes  -> entanglement matters DURING optimization (novel claim)
- advantage persists  -> the benefit is a generic high-dim bottleneck
  training effect; the deeper kill ships honestly.

Usage: uv run paper/phase_b/finetune_td_ablate.py noent [seed] [budget_s] [lr] [target] [k]
Checkpoints: runs/phase_b2/noent_s{seed}[.pt|_latest.pt]
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import qlayer  # noqa: E402


def _zero_blockade(self):
    return torch.zeros(2 ** self.n, device=self.pos.device)


qlayer.QuantumBlock._blockade_energy = _zero_blockade  # every register, all run

import finetune_td  # noqa: E402  (inherits the patch at build time)

if __name__ == "__main__":
    # variant name must map to the same architecture as mr30ryd
    sys.argv[1] = "mr30ryd"
    finetune_td.VARIANT = "mr30ryd"
    # separate checkpoint namespace so nothing collides with the real arms
    finetune_td.main.__globals__["build_hybrid"] = finetune_td.build_hybrid
    _orig_main = finetune_td.main

    def main():
        import types
        _orig_main()

    # redirect checkpoint names by patching pathlib joins is brittle; instead
    # rename outputs post-hoc is worse — simplest: chdir trick not needed,
    # finetune_td derives names from VARIANT; override to a distinct tag:
    finetune_td.VARIANT = "noent"

    class _Shim:
        pass

    # build_hybrid("noent") must still build the mr30ryd architecture:
    _orig_build = finetune_td.build_hybrid
    finetune_td.build_hybrid = lambda v, **kw: _orig_build("mr30ryd", **kw)
    finetune_td.SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1337
    finetune_td.BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 2400.0
    finetune_td.LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    finetune_td.TARGET_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    finetune_td.K = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    finetune_td.main()
