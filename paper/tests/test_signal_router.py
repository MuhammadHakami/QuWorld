"""Sparse routing invariants; tests do not establish predictive improvement."""
from pathlib import Path
import sys
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_search.core import RegisterConfig
from config_search.signal_router import SignalRouter


@pytest.fixture
def router():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(103)
    return SignalRouter(8, RegisterConfig(width=5, registers=3, layers=2,
                        geometry='bounded_dynamic', encoding_rms=.1)).cuda()


class AmplitudeScore(nn.Module):
    def forward(self, features):
        # Transparent test policy to verify causal input-driven allocation.
        return (features[..., -2:-1] - .7) * 10


def test_no_dispatch_means_exact_classical_fallback(router):
    router.eval()
    x = torch.randn(3, 8, 2, 3, device='cuda')
    torch.testing.assert_close(router(x), x, rtol=0, atol=0)
    assert router.last_route['executed_register_batches'] == 0
    assert not router.last_route['active'].any()


def test_input_dependent_budget_and_locality(router):
    router.score = AmplitudeScore()
    router.eval()
    router.alpha.data.fill_(.1)
    x = torch.zeros(3, 8, 1, 4, device='cuda')
    x[1, :, 0, 2] = 4
    x[2] = 4
    y = router(x)
    assert router.last_route['active_qubits'].tolist() == [0, 5, 15]
    assert router.last_route['executed_register_batches'] == 3
    torch.testing.assert_close(y[0], x[0], rtol=0, atol=0)
    torch.testing.assert_close(y[1, :, :, [0, 1, 3]], x[1, :, :, [0, 1, 3]], rtol=0, atol=0)
    assert not torch.equal(y[1, :, :, 2], x[1, :, :, 2])
    for selected in router.last_route['indices']:
        assert len(selected.unique()) == len(selected)


def test_batch_permutation_does_not_change_individual_routing(router):
    router.score = AmplitudeScore()
    router.eval()
    router.alpha.data.fill_(.1)
    x = torch.randn(3, 8, 2, 3, device='cuda') * 3
    y = router(x)
    counts = router.last_route['active_qubits'].clone()
    permutation = torch.tensor([2, 0, 1], device='cuda')
    yp = router(x[permutation])
    torch.testing.assert_close(yp, y[permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(router.last_route['active_qubits'], counts[permutation])


def test_warmup_gradients_and_no_implicit_eval_warmup(router):
    router.train(); router.warmup = True
    router.alpha.data.fill_(.1)
    x = torch.randn(2, 8, 2, 2, device='cuda', requires_grad=True)
    y = router(x)
    loss = y.square().mean() + router.difficulty_loss(x, torch.ones(2, 4, device='cuda'))
    loss.backward()
    for p in (router.alpha, router.score[-1].weight, router.cores[0].regs[0].cond.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    router.eval(); router(x.detach())
    assert not router.last_route['warmup']
    assert router.last_route['executed_register_batches'] == 0


def test_zero_input_gradients_and_vector_shape(router):
    x = torch.zeros(2, 8, device='cuda', requires_grad=True)
    router.warmup = True
    y = router(x)
    (y.sum() + router.difficulty_loss(x, torch.ones(2, 1, device='cuda'))).backward()
    assert torch.isfinite(x.grad).all()
    assert y.shape == x.shape
    assert router.last_route['active_qubits'].tolist() == [5, 5]


def test_classical_control_has_same_router_and_projection_architecture(router):
    control = SignalRouter(8, router.config, arm='classical').cuda()
    assert [p.shape for p in control.score.parameters()] == [p.shape for p in router.score.parameters()]
    assert [p.shape for p in control.projections.parameters()] == [p.shape for p in router.projections.parameters()]
    assert all(abs(gap) < 8 + router.config.width + 1 for gap in control.core_parameter_gaps)


def test_paired_shared_weights_are_identical():
    cfg = RegisterConfig(width=5, registers=3, layers=2, geometry='bounded_dynamic')
    torch.manual_seed(55)
    quantum = SignalRouter(8, cfg)
    torch.manual_seed(55)
    control = SignalRouter(8, cfg, arm='classical')
    for module in ('score', 'projections'):
        qs, cs = getattr(quantum, module).state_dict(), getattr(control, module).state_dict()
        for key in qs:
            torch.testing.assert_close(qs[key], cs[key], rtol=0, atol=0)


def test_per_input_budget_caps_and_zero_identity(router):
    router.train(); router.warmup = True
    router.alpha.data.fill_(.1)
    x = torch.randn(3, 8, 2, 3, device='cuda')
    y = router(x, budget=torch.tensor([0, 1, 3], device='cuda'))
    assert router.last_route['active_qubits'].tolist() == [0, 5, 15]
    torch.testing.assert_close(y[0], x[0], rtol=0, atol=0)
    for invalid in (-1, 4, .5, True, [1, 2]):
        with pytest.raises(ValueError):
            router(x, budget=invalid)


def test_relative_correction_bound_zero_start_and_zero_input(router):
    from config_search.signal_router import RelativeSignalRouter
    for arm in ('quantum', 'classical'):
        relative = RelativeSignalRouter(8, router.config, arm=arm).cuda()
        relative.warmup = True
        x = torch.randn(3, 8, 2, 3, device='cuda') * 100
        torch.testing.assert_close(relative(x), x, rtol=0, atol=0)
        relative.alpha.data.fill_(100)
        y = relative(x)
        delta_norm = (y-x).norm(dim=1)
        assert (delta_norm <= .1*x.norm(dim=1)+1e-5).all()
        zero = torch.zeros_like(x, requires_grad=True)
        z = relative(zero)
        torch.testing.assert_close(z, zero, rtol=0, atol=0)
        z.sum().backward()
        assert torch.isfinite(zero.grad).all()


def test_location_controls_are_distinct_causal_bijections():
    from config_search.signal_router import ControlledSignalRouter
    cfg=RegisterConfig(width=5,registers=3,layers=2)
    probabilities=torch.tensor([[.1,.9,.2,.8,.3,.7],[.8,.2,.7,.1,.6,.3]])
    adaptive=ControlledSignalRouter(8,cfg,routing='adaptive')
    static=ControlledSignalRouter(8,cfg,routing='static')
    permuted=ControlledSignalRouter(8,cfg,routing='permuted')
    values,ids=adaptive.choose_locations(probabilities,3)
    pv,pi=permuted.choose_locations(probabilities,3)
    torch.testing.assert_close(pv,values)
    assert not torch.equal(pi,ids)
    assert all(len(row.unique())==3 for row in pi)
    _,si=static.choose_locations(probabilities,3)
    assert torch.equal(si[0],si[1])
    _,reversed_ids=permuted.choose_locations(probabilities.flip(0),3)
    torch.testing.assert_close(reversed_ids,pi.flip(0))
