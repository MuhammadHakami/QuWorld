"""Structure-search bake-off runtime (AG-2c, 14-structure-bakeoff.md).

Modules:
  stream     — continual A/B/C adaptation stream (TRL2D tcool regimes + TORAX)
  substrate  — shared kmix30 bank substrate + sealed constraint evaluator
  qreadout   — ZZ/Z observable readout for Rydberg registers (e-prop factor 1)
  strategies — gradient / evolution / eprop / spsa / rls / frozen arms
  runner     — eval-unit-parity protocol, metrics, JSONL results, kill checks
"""

try:  # imported as a package (paper/ on sys.path)
    from ._bootstrap import add_paths as _add_paths
except ImportError:  # flat import (paper/bakeoff/ on sys.path)
    from _bootstrap import add_paths as _add_paths

_add_paths()
