"""CUDA contract checks for bounded per-input configurations and strict diagnostics."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config_search.core import BoundedGeometry, ConfigurableCore, RegisterConfig, make_adapter
from config_search.train_trial import validation_ids
from qlayer import MultiRegister
from register_diag import checkpoint_state, effective_rank, load_arena_checkpoint


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("GPU-only project: no CPU inference fallback")
    return torch.device("cuda")


def test_resource_cap_and_finite_config():
    for kwargs in ({"width": 11}, {"registers": 4}, {"tau": float("nan")}, {"spacing": 0.5}):
        with pytest.raises(ValueError):
            RegisterConfig(**kwargs)


def test_legacy_default_matches_current_core(cuda, monkeypatch):
    monkeypatch.setenv("QREADOUT", "z")
    torch.manual_seed(9)
    old = MultiRegister(8, 5, 3, 2, "rydberg").to(cuda)
    torch.manual_seed(9)
    new = ConfigurableCore(8, RegisterConfig(width=5, layers=2)).to(cuda)
    assert set(old.state_dict()) == set(new.state_dict())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    h = torch.randn(2, 8, device=cuda)
    torch.testing.assert_close(old(h), new(h), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["bounded_static", "bounded_dynamic"])
def test_spacing_constraint_under_extreme_parameters(cuda, mode):
    cfg = RegisterConfig(geometry=mode, spacing=0.6)
    block = BoundedGeometry(8, cfg).to(cuda)
    with torch.no_grad():
        block.pos0.normal_(std=100)
        if block.geo is not None:
            for p in block.geo.parameters():
                p.normal_(std=100)
    h = torch.randn(7, 8, device=cuda) * 100
    distances = torch.cdist(block.positions(h), block.positions(h))
    iu = torch.triu_indices(cfg.width, cfg.width, 1, device=cuda)
    assert distances[:, iu[0], iu[1]].min() >= block.MIN_DIST - 1e-6


def test_per_input_geometry_has_gradient(cuda):
    cfg = RegisterConfig(width=5, layers=2, registers=1, geometry="bounded_dynamic", encoding_rms=.1)
    block = ConfigurableCore(8, cfg).to(cuda)
    h = torch.randn(3, 8, device=cuda)
    result = block(h)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    grad = block.regs[0].geo[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        block.regs[0].geo[-1].weight.add_(0.01 * grad)
    positions = block.regs[0].positions(h)
    assert not torch.equal(positions[0], positions[1])


@pytest.mark.parametrize("geometry", ["legacy", "bounded_static", "bounded_dynamic"])
def test_zero_gate_and_control_counts(cuda, geometry):
    cfg = RegisterConfig(width=5, layers=2, geometry=geometry, readout="cond2")
    x = torch.randn(2, 8, 3, 4, device=cuda)
    for arm in ("quantum", "mlp"):
        block, meta = make_adapter(8, cfg, arm)
        block = block.to(cuda)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        assert abs(meta["parameter_gap"]) < 8 + cfg.width * cfg.registers + 1


def test_checkpoint_containers_and_reject_official(tmp_path):
    raw = {"weight": torch.zeros(2, 2)}
    assert checkpoint_state({"model_state_dict": raw, "step": 12}) is raw
    with pytest.raises(ValueError):
        checkpoint_state({"step": 1})
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {"neck.bottleneck.core.weight": torch.zeros(2)}}, path)
    with pytest.raises(ValueError, match="Official CNext"):
        load_arena_checkpoint(torch.nn.Linear(2, 2), str(path))
    torch.save({"model_state_dict": {"wrong": torch.zeros(2)}}, path)
    with pytest.raises(RuntimeError):
        load_arena_checkpoint(torch.nn.Linear(2, 2), str(path))


def test_constant_readout_rank_is_zero(cuda):
    assert effective_rank(torch.ones(32, 10, device=cuda)) == 0
    assert effective_rank(torch.zeros(32, 10, device=cuda)) == 0
    x = torch.arange(32, device=cuda, dtype=torch.float32).unsqueeze(1).expand(-1, 10)
    assert effective_rank(x) == pytest.approx(1, abs=1e-5)


def test_search_confirmation_trajectories_are_disjoint():
    search, confirm = validation_ids(9, "search"), validation_ids(9, "confirmation")
    assert not set(search) & set(confirm)
    assert sorted(search + confirm) == list(range(9))


def test_classical_initialization_independent_of_encoding():
    torch.manual_seed(17)
    first, _ = make_adapter(8, RegisterConfig(), "mlp")
    torch.manual_seed(17)
    second, _ = make_adapter(8, RegisterConfig(encoding_rms=.1), "mlp")
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
