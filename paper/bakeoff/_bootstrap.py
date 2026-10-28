"""Path bootstrap for the bakeoff package.

The paper/ modules import each other flat (`from qlayer import ...`), so every
bakeoff module calls `add_paths()` to put both paper/bakeoff and paper/ on
sys.path. Entry point is paper/bakeoff/runner.py (its own dir is sys.path[0]).
"""

import pathlib
import sys

BAKEOFF_DIR = pathlib.Path(__file__).resolve().parent
PAPER_DIR = BAKEOFF_DIR.parent
REPO_DIR = PAPER_DIR.parent


def add_paths() -> None:
    for p in (str(BAKEOFF_DIR), str(PAPER_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
