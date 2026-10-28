"""Variant wiring for the Gutzwiller control in both training harnesses:
`gutz30` must build the mr30ryd architecture with mean-field registers."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "phase_b"))
sys.path.insert(0, str(ROOT / "paper"))
import qlayer  # noqa: E402
import model as arena_model  # noqa: E402
import cnext_hybrid  # noqa: E402


def _blocks(m):
    return [b for b in m.modules() if isinstance(b, qlayer.QuantumBlock)]


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_arena_gutz30_is_mean_field_and_parameter_matched():
    torch.manual_seed(0)
    g = arena_model.build_model("gutz30", 16)
    torch.manual_seed(0)
    r = arena_model.build_model("mr30ryd", 16)
    assert [b.ansatz for b in _blocks(g)] == ["rydberg_mf"] * 3
    assert [b.n for b in _blocks(g)] == [10, 10, 10]
    assert _nparams(g) == _nparams(r)
    for (kg, vg), (kr, vr) in zip(g.state_dict().items(), r.state_dict().items()):
        assert kg == kr and torch.equal(vg, vr)      # same initialisation


def test_official_gutz30_is_mean_field_and_parameter_matched():
    g = cnext_hybrid.build_hybrid("gutz30")
    r = cnext_hybrid.build_hybrid("mr30ryd")
    assert [b.ansatz for b in _blocks(g)] == ["rydberg_mf"] * 3
    assert list(g.state_dict().keys()) == list(r.state_dict().keys())
    assert _nparams(g) == _nparams(r)


def test_official_gutz30_checkpoint_roundtrip(tmp_path):
    g = cnext_hybrid.build_hybrid("gutz30")
    path = tmp_path / "gutz30_s0.pt"
    torch.save({"model_state_dict": g.state_dict()}, path)
    m = cnext_hybrid.load_hybrid_checkpoint("gutz30", str(path))
    assert [b.ansatz for b in _blocks(m)] == ["rydberg_mf"] * 3
