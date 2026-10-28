"""Load-time compatibility for the official RouteFinder checkpoints on this stack (torch 2.14 / torchrl 0.14).

The checkpoints pickle the training environment, whose specs use torchrl class names that were later renamed
(CompositeSpec -> Composite, ...). Rollouts never use that pickled environment (test.py and our harness build a
fresh MTVRPEnv), so aliasing the old names to the new classes is enough to unpickle it; weights are untouched.

    python paper/vrp/rf_compat.py <script.py> [args...]     # runs the script with the aliases installed
"""
from __future__ import annotations

import os
import runpy
import sys

LEGACY = {
    "CompositeSpec": "Composite",
    "UnboundedContinuousTensorSpec": "Unbounded",
    "UnboundedDiscreteTensorSpec": "Unbounded",
    "BoundedTensorSpec": "Bounded",
    "DiscreteTensorSpec": "Categorical",
    "OneHotDiscreteTensorSpec": "OneHot",
    "BinaryDiscreteTensorSpec": "Binary",
    "MultiDiscreteTensorSpec": "MultiCategorical",
    "MultiOneHotDiscreteTensorSpec": "MultiOneHot",
}


def install() -> list[str]:
    """Alias legacy spec names in torchrl.data.tensor_specs; returns the names that were added."""
    import torchrl.data.tensor_specs as ts
    added = []
    for old, new in LEGACY.items():
        if not hasattr(ts, old) and hasattr(ts, new):
            setattr(ts, old, getattr(ts, new)); added.append(old)
    # official checkpoints (trusted source: ai4co release) pickle non-tensor objects
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return added


DEAD_END_ROWS = []   # per decode step, rows with no feasible action (GPU tensors; summed at the end of a run)


def guard_dead_ends() -> None:
    """Multi-depot deadlock guard (doc 35 §16 amendment, Sep 25 19:55).

    MTVRPEnv (mtdvrp) marks the current depot unavailable after each depot deadlock; when a remaining customer cannot be
    served from any depot (time window, distance limit), the agent cycles through the depots until the action mask is
    all False and the sampler raises on NaN probabilities. Here such rows may visit any unvisited customer instead.
    The guard changes nothing on a step where some action is feasible, so every run that finished without it is
    unchanged; the number of guarded rows is reported with each run.
    """
    import torch
    from routefinder.envs.mtdvrp import MTVRPEnv
    orig = MTVRPEnv.get_action_mask

    def get_action_mask(td):
        td = orig(td)
        mask = td["action_mask"]; dead = ~mask.any(-1)
        nd = int(td["num_depots"][0].item())
        fallback = ~td["visited"]; fallback[:, :nd] = False
        td.set("action_mask", torch.where(dead[:, None], fallback, mask))
        DEAD_END_ROWS.append(dead.sum())
        if len(DEAD_END_ROWS) > 4096:
            DEAD_END_ROWS[:] = [torch.stack(DEAD_END_ROWS).sum()]
        return td

    MTVRPEnv.get_action_mask = staticmethod(get_action_mask)


def dead_end_count() -> int:
    import torch
    return int(torch.stack(DEAD_END_ROWS).sum()) if DEAD_END_ROWS else 0


if __name__ == "__main__":
    install()
    script, sys.argv = sys.argv[1], sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
