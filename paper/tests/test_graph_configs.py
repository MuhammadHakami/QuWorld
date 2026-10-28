"""Wave G (Sep 21): register graph configurations. Contracts:
- topology families give the documented nearest-neighbour spacing and graph degree;
- the reference grammar 'q30' is the mr30ryd register (same output, same parameter count);
- gate / cascade / shared are IDENTICAL to the reference at init (zero-init links, unit gates);
- the gate changes qubit participation once trained (an idle qubit stays in |0>);
- partitions keep the qubit total; the matched control gets the register's exact budget."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import model  # noqa: E402
import qlayer  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, B = 8, 4


def _h(seed=1, d=W):
    torch.manual_seed(seed)
    return torch.randn(B, d, device=DEV)


def _degree(pos, cutoff=1.05 * 1.0):
    d = torch.cdist(pos, pos) + 99 * torch.eye(pos.shape[0])
    return (d <= cutoff).sum(-1).float()


@pytest.mark.parametrize("geo,expect_max_deg", [("chain", 2), ("ring", 2), ("grid", 4), ("tri", 5)])   # 10-atom tri patch: interior max 5
def test_topology_spacing_and_degree(geo, expect_max_deg):
    n = 10
    pos = qlayer.topology_positions(n, geo, 1.0)
    d = torch.cdist(pos, pos) + 99 * torch.eye(n)
    leaves = d
    assert torch.allclose(leaves.min(), torch.tensor(1.0), atol=1e-4), f"{geo}: nearest neighbour {leaves.min():.3f} != 1"
    assert int(_degree(pos).max()) == expect_max_deg
    pos2 = qlayer.topology_positions(n, geo, 0.7)
    d2 = torch.cdist(pos2, pos2)[~torch.eye(n, dtype=bool)]
    assert torch.allclose(d2.min(), torch.tensor(0.7 * (d.min() / leaves.min()).item()), atol=1e-3)


def test_spacing_scales_coupling():
    dense = qlayer.QuantumBlock(W, 6, 2, "rydberg", geometry="chain", spacing=0.7)
    sparse = qlayer.QuantumBlock(W, 6, 2, "rydberg", geometry="chain", spacing=1.5)
    with torch.no_grad():  # remove the 0.05 jitter for an exact check
        dense.pos.copy_(qlayer.topology_positions(6, "chain", 0.7))
        sparse.pos.copy_(qlayer.topology_positions(6, "chain", 1.5))
    vd, vs = dense._pair_couplings()[0, 1], sparse._pair_couplings()[0, 1]
    assert abs(vd.item() - 0.7 ** -6) < 1e-3 and abs(vs.item() - 1.5 ** -6) < 1e-4


def test_reference_grammar_matches_mr30ryd():
    cfg = model.parse_config("q30")
    assert cfg == dict(total=30, registers=3, n=10, geometry="grid", spacing=1.0, link="independent",
                       gate=False, control=False, budget=0, random_mask=False)
    torch.manual_seed(0); a = model.build_model("mr30ryd", 4, width=W).to(DEV)
    torch.manual_seed(0); b = model.build_model("q30", 4, width=W).to(DEV)
    assert model._count(a) == model._count(b)
    x = torch.randn(B, 4, 16, 24, device=DEV)
    assert torch.allclose(a(x), b(x), atol=1e-6)


@pytest.mark.parametrize("variant", ["q30-gate", "q30-link=cascade", "q30-link=shared", "q30-geo=ring", "q30-sp=0.7"])
def test_output_and_budget(variant):
    torch.manual_seed(0)
    m = model.build_model(variant, 4, width=W).to(DEV)
    x = torch.randn(B, 4, 16, 24, device=DEV)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    ctrl = model.build_model(variant + "-mlp", 4, width=W)
    core_q = model.parse_config(variant); core_c = model.parse_config(variant + "-mlp")
    reg, _ = model.build_config_core(core_q, W); mlp, _ = model.build_config_core(core_c, W)
    # MLPCore rounds its hidden width: at toy width 8 the budget lands within a few %, at the arena's
    # width 24 within 0.001% (documented in 13-adaptive-geometry.md)
    assert abs(model._count(mlp) - model._count(reg)) <= 0.03 * model._count(reg), "control not parameter-matched"


def test_gate_and_cascade_identical_to_reference_at_init():
    """Zero-init links and unit gates: the new mechanisms must not change the function at init."""
    h = _h()
    torch.manual_seed(0); ref = qlayer.MultiRegister(W, 5, 3, 2, "rydberg").to(DEV)
    torch.manual_seed(0); gated = qlayer.MultiRegister(W, 5, 3, 2, "rydberg", gate=True).to(DEV)
    torch.manual_seed(0); casc = qlayer.MultiRegister(W, 5, 3, 2, "rydberg", link="cascade").to(DEV)
    # copy the shared parameters so only the mechanism differs
    for tgt in (gated, casc):
        for r_ref, r_t in zip(ref.regs, tgt.regs):
            with torch.no_grad():
                r_t.pos.copy_(r_ref.pos); r_t.base.copy_(r_ref.base); r_t.omega.copy_(r_ref.omega); r_t.tau.copy_(r_ref.tau)
                r_t.cond.weight[:, :W].copy_(r_ref.cond.weight); r_t.cond.bias.copy_(r_ref.cond.bias)
    with torch.no_grad():
        for r in ref.regs: r.cond.weight.normal_(std=0.3)
        for tgt in (gated, casc):
            for r_ref, r_t in zip(ref.regs, tgt.regs): r_t.cond.weight[:, :W].copy_(r_ref.cond.weight)
    y_ref = ref(h)
    assert torch.allclose(gated(h), y_ref, atol=2e-2), "gate bias +4 -> sigmoid 0.982: near-identical drive"
    assert torch.allclose(casc(h), y_ref, atol=1e-6), "cascade with zero-init links must equal independent"
    assert gated.participation(h).min() > 0.98


def test_gate_can_idle_a_qubit():
    """A closed gate leaves the atom in |0>: its <Z> readout is +1 and it drops out of the blockade."""
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(W, 4, 2, "rydberg", gate=True).to(DEV)
    h = _h()
    with torch.no_grad():
        blk.gate_lin.bias[0] = -30.0          # close qubit 0 for every input
    z = blk(h)                                # readout = <Z_i>
    assert torch.allclose(z[:, 0], torch.ones(B, device=DEV), atol=1e-4)
    assert blk.participation(h)[:, 0].max() < 1e-6


def test_shared_link_ties_controllers():
    m = qlayer.MultiRegister(W, 5, 3, 2, "rydberg", link="shared")
    assert m.regs[1].cond is m.regs[0].cond and m.regs[2].omega is m.regs[0].omega
    n_shared = model._count(m); n_indep = model._count(qlayer.MultiRegister(W, 5, 3, 2, "rydberg"))
    assert n_shared < n_indep


@pytest.mark.parametrize("part", ["6x5", "3x10", "2x15"])
def test_partitions_keep_total(part):
    cfg = model.parse_config(f"q30-part={part}")
    assert cfg["registers"] * cfg["n"] == 30
    reg, out = model.build_config_core(cfg, W)
    h = _h()
    assert reg.to(DEV)(h).shape == (B, 30) and out == 30


def test_bad_partition_rejected():
    with pytest.raises(AssertionError):
        model.parse_config("q30-part=4x7")


# ---- per-input budget selector (wave G follow-up) --------------------------------------------------

@pytest.mark.parametrize("k", [1, 3])
def test_budget_drives_exactly_k_atoms_per_input(k):
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(W, 5, 2, "rydberg", budget=k).to(DEV)
    h = _h()
    g = blk.participation(h)
    assert g.shape == (B, 5) and torch.allclose(g.sum(-1), torch.full((B,), float(k), device=DEV), atol=1e-5), "exactly k atoms driven per input"
    assert torch.all((g.abs() < 1e-5) | ((g - 1).abs() < 1e-5))
    z = blk(h)
    undriven = g.abs() < 1e-5
    assert torch.allclose(z[undriven], torch.ones_like(z[undriven]), atol=1e-4), "idle atoms read <Z> = +1"


def test_budget_selection_depends_on_input():
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(W, 6, 2, "rydberg", budget=3).to(DEV)
    with torch.no_grad():
        blk.gate_lin.weight.normal_(std=1.0)
    g = blk.participation(torch.randn(64, W, device=DEV))
    assert (g != g[0]).any(), "different inputs must be able to pick different atoms"


def test_budget_gradient_reaches_selector():
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(W, 5, 2, "rydberg", budget=2).to(DEV)
    h = _h().requires_grad_(False)
    blk(h).sum().backward()
    assert blk.gate_lin.weight.grad is not None and blk.gate_lin.weight.grad.abs().sum() > 0


def test_random_mask_control_is_fixed_across_inputs():
    torch.manual_seed(0)
    blk = qlayer.QuantumBlock(W, 6, 2, "rydberg", budget=3, random_mask=True).to(DEV)
    g = blk.participation(torch.randn(16, W, device=DEV))
    assert torch.allclose(g.sum(-1), torch.full((16,), 3.0, device=DEV)) and torch.all(g == g[0])


def test_budget_grammar_and_budget_match():
    cfg = model.parse_config("q30-k=5")
    assert cfg["budget"] == 5 and not cfg["random_mask"]
    assert model.parse_config("q30-k=5-randk")["random_mask"]
    reg, _ = model.build_config_core(cfg, W); ctl, _ = model.build_config_core(model.parse_config("q30-k=5-mlp"), W)
    h = _h()
    assert reg.to(DEV)(h).shape == (B, 30) and torch.allclose(reg.participation(h).sum(-1), torch.full((B,), 15.0, device=DEV), atol=1e-4)   # 3 registers x 5 atoms
    assert abs(model._count(ctl) - model._count(reg)) <= 0.03 * model._count(reg)
