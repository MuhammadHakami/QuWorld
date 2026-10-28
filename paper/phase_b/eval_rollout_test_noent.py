"""Official-protocol TEST rollout eval for the no-entanglement ablation arms.

Identical to eval_rollout_test.py except QuantumBlock blockade energies are
zeroed — matching how the noent checkpoints were trained (product-state from
step 0, see finetune_td_ablate.py). Evaluating them with the blockade intact
would score an architecture the weights never saw.

Usage: uv run paper/phase_b/eval_rollout_test_noent.py runs/phase_b2/noent_s<seed>.pt
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import qlayer  # noqa: E402


def _zero_blockade(self):
    return torch.zeros(2 ** self.n, device=self.pos.device)


qlayer.QuantumBlock._blockade_energy = _zero_blockade  # before any forward

import cnext_hybrid  # noqa: E402

_orig_build = cnext_hybrid.build_hybrid


def _build(variant, *args, **kwargs):
    # noent checkpoints record their own variant name; same arch as mr30ryd
    return _orig_build("mr30ryd" if variant == "noent" else variant,
                       *args, **kwargs)


cnext_hybrid.build_hybrid = _build

import eval_rollout_test  # noqa: E402

if __name__ == "__main__":
    eval_rollout_test.main()
