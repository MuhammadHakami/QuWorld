"""Sliding-start rollout eval for the no-entanglement ablation checkpoints.

Same shim as eval_rollout_test_noent.py: blockade energies zeroed (as trained)
and the checkpoint's variant name 'noent' mapped to the mr30ryd architecture.
Usage: uv run paper/phase_b/eval_rollout_multistart_noent.py runs/phase_b2/noent_s<seed>.pt noent_s<seed>
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
    return _orig_build("mr30ryd" if variant == "noent" else variant, *args, **kwargs)


cnext_hybrid.build_hybrid = _build

import eval_rollout_multistart  # noqa: E402

if __name__ == "__main__":
    eval_rollout_multistart.main()
