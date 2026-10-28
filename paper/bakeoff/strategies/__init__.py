"""Strategy registry for the bake-off runner."""

from _bootstrap import add_paths

add_paths()

from strategies.base import FrozenStrategy  # noqa: E402
from strategies.eprop import EPropStrategy  # noqa: E402
from strategies.evolution import EvolutionQDStrategy  # noqa: E402
from strategies.gradient import GradientStrategy  # noqa: E402
from strategies.rls import RLSStrategy  # noqa: E402
from strategies.spsa import SPSAStrategy  # noqa: E402

REGISTRY = {cls.name: cls for cls in (
    GradientStrategy, EvolutionQDStrategy, EPropStrategy, SPSAStrategy,
    RLSStrategy, FrozenStrategy)}
