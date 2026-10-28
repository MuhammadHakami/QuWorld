"""Unit tests for the Physics-IQ harness: naming/duration contract, geometry,
adapter checkpoint contract, projection, and the tap-point hook.

Run: uv run --no-sync python -m pytest paper/phase_c/test_physiq_contract.py -q
CPU only (ffmpeg/ffprobe on PATH); no model weights, no GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import physiq_adapter as pa  # noqa: E402
import physiq_runner as runner  # noqa: E402
import physiq_t5 as pt5  # noqa: E402
import physiq_video as pv  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


# ---------------------------------------------------------------- video I/O

def test_write_and_probe_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(81, 64, 64, 3), dtype=np.uint8)
    info = pv.write_mp4_rgb(frames, tmp_path / "raw.mp4", fps=16)
    assert (info.nb_frames, info.width, info.height, info.fps) == (81, 64, 64, 16.0)
    assert info.duration_s == pytest.approx(81 / 16, abs=1e-3)
    assert not info.duration_ok  # 5.0625 s: raw output must NOT pass as-is
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_trim_gives_exactly_80_frames_5s(tmp_path):
    pv.synthetic_mp4(tmp_path / "raw.mp4", n_frames=81, fps=16)
    info = pv.trim_to_frames(tmp_path / "raw.mp4", tmp_path / "trim.mp4")
    assert info.nb_frames == 80
    assert info.duration_s == pytest.approx(5.0, abs=1e-3)
    assert info.duration_ok and info.fps == 16.0


def _fill_contract_folder(folder: Path, names: list[str], clip: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(clip, folder / name)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    pv.synthetic_mp4(d / "ok.mp4", n_frames=80)     # 5.000 s
    pv.synthetic_mp4(d / "long.mp4", n_frames=81)   # 5.0625 s
    return d


@pytest.fixture(scope="module")
def out_names():
    return [s.out_name for s in runner.load_scenarios(runner.descriptions_csv("op"))]


def test_scenarios_match_naming_contract(out_names):
    assert len(out_names) == 198
    assert out_names == sorted(out_names)
    for i, name in enumerate(out_names, start=1):
        assert name.startswith(f"{i:04d}_")
        assert pv.NAME_RE.match(name), name


def test_validate_contract_accepts_198_five_second_clips(tmp_path, clips, out_names):
    folder = tmp_path / "gen"
    _fill_contract_folder(folder, out_names, clips / "ok.mp4")
    assert pv.validate_contract(folder) == ()


def test_validate_contract_rejects_bad_folders(tmp_path, clips, out_names):
    folder = tmp_path / "gen"
    _fill_contract_folder(folder, out_names, clips / "ok.mp4")
    shutil.copyfile(clips / "long.mp4", folder / out_names[5])          # 81 frames
    (folder / "meta.json").write_text("{}")                             # non-mp4 entry
    problems = pv.validate_contract(folder)
    assert any("5 s" in p for p in problems)
    assert any("non-mp4" in p for p in problems)
    os.remove(folder / out_names[0])                                    # count / prefix
    problems = pv.validate_contract(folder, check_media=False)
    assert any("expected 198" in p for p in problems)
    assert any("prefix 0001_" in p for p in problems)


def test_clear_tmp_files_removes_only_leftovers(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.mp4.tmp.mp4").write_bytes(b"x")
    removed = pv.clear_tmp_files(tmp_path)
    assert [p.name for p in removed] == ["b.mp4.tmp.mp4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp4"]


def _official_validator(folder: Path) -> subprocess.CompletedProcess:
    code = ("import sys; from physiq.run_physics_iq import validate_generations; "
            "validate_generations(sys.argv[1]); print('OFFICIAL_OK')")
    return subprocess.run(["uv", "run", "--no-sync", "python", "-c", code, str(folder)],
                          cwd=runner.PHYSIQ_REPO, capture_output=True, text=True,
                          env={**os.environ, "MPLBACKEND": "Agg"})


@pytest.mark.skipif(not (runner.PHYSIQ_REPO / ".venv" / "bin" / "python").exists(),
                    reason="scorer env not synced")
def test_official_validator_agrees(tmp_path, clips, out_names):
    good = tmp_path / "good"
    _fill_contract_folder(good, out_names, clips / "ok.mp4")
    res = _official_validator(good)
    assert res.returncode == 0 and "OFFICIAL_OK" in res.stdout, res.stderr[-2000:]
    bad = tmp_path / "bad"
    _fill_contract_folder(bad, out_names, clips / "ok.mp4")
    shutil.copyfile(clips / "long.mp4", bad / out_names[0])
    res = _official_validator(bad)
    assert res.returncode != 0 and "needs to be 5s" in res.stderr


# ---------------------------------------------------------------- geometry

def test_crop_box_center_crops_to_480_832_aspect():
    assert runner.crop_box(3840, 2160) == (48, 0, 3792, 2160)
    left, top, right, bottom = runner.crop_box(3840, 2160)
    assert (right - left, bottom - top) == (3744, 2160)
    assert runner.crop_box(832, 480) == (0, 0, 832, 480)          # already exact
    assert runner.crop_box(1000, 1000) == (0, 211, 1000, 788)     # too tall -> crop height


def test_latent_geometry_boundary_fix():
    # the released computation lands at 58x104 for ANY 480:832 image (float floor)
    assert runner.latent_geometry(2160, 3744, 480 * 832) == (58, 104)
    assert runner.latent_geometry(480, 832, 480 * 832) == (58, 104)
    assert runner.latent_geometry(2160, 3840, 480 * 832) == (58, 104)
    # with the crop + margin every switch frame yields the adapters' 60x104
    assert runner.latent_geometry(2160, 3744, runner.MAX_AREA) == (60, 104)
    assert runner.latent_geometry(480, 832, runner.MAX_AREA) == (60, 104)
    assert (runner.LAT_H, runner.LAT_W) == (60, 104)


def test_load_conditioning_image_geometry(out_names):
    s = runner.load_scenarios(runner.descriptions_csv("op"))[0]
    im = runner.load_conditioning_image(s.switch_frame)
    assert im.size == (3744, 2160)
    assert runner.latent_geometry(im.size[1], im.size[0]) == (60, 104)


# ---------------------------------------------------------------- seeds / projection

def test_scenario_seed_is_deterministic_and_paired():
    assert runner.scenario_seed(1, "0001") == runner.scenario_seed(1, "0001")
    assert runner.scenario_seed(1, "0001") != runner.scenario_seed(1, "0002")
    assert runner.scenario_seed(1, "0198") != runner.scenario_seed(2, "0001")


def test_project_wallclock_separates_t5_miss_overhead():
    records = [
        {"event": "pipeline_loaded", "load_s": 120.0},
        {"gen_s": 140.0, "write_s": 10.0, "t5_cache_hit": False, "peak_alloc_gb": 55.0, "rss_peak_gb": 30.0},
        {"gen_s": 95.0, "write_s": 5.0, "t5_cache_hit": True, "peak_alloc_gb": 56.0, "rss_peak_gb": 31.0},
        {"gen_s": 96.0, "write_s": 4.0, "t5_cache_hit": True, "peak_alloc_gb": 54.0, "rss_peak_gb": 29.0},
    ]
    p = runner.project_wallclock(records, remaining_videos=195, remaining_unique_prompts=65, load_s=120.0)
    assert p["baseline_s"] == 100.0 and p["t5_miss_s"] == 50.0
    assert p["hours"] == pytest.approx((195 * 100 + 65 * 50 + 120) / 3600)
    assert p["peak_alloc_gb"] == 56.0 and p["n_timed"] == 3
    with pytest.raises(ValueError):
        runner.project_wallclock([records[0]], 1, 1, 0.0)


def test_t5_cache_key_matches_pipeline_key():
    import hashlib
    prompt = "A ball rolls off a table. Static shot."
    assert pt5.prompt_key(prompt) == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert pt5.cache_path(prompt).name == f"ctx_{pt5.prompt_key(prompt)[:16]}.pt"


# ---------------------------------------------------------------- adapter contract

try:  # module-level so torch.save can pickle whole modules (fallback layout)
    import torch as _torch

    class Add(_torch.nn.Module):
        """Test adapter: x + bias."""

        def __init__(self, scale: float = 0.0):
            super().__init__()
            self.bias = _torch.nn.Parameter(_torch.full((1,), float(scale)))

        def forward(self, x):
            return x + self.bias
except ImportError:  # pragma: no cover
    Add = None


def _p1_checkpoint(torch, path: Path, arm: str, seed: int, alpha: float, valid_rollout: float) -> None:
    """Checkpoint exactly as p1_train.train_unit saves it (:301-304)."""
    import p1_train

    torch.manual_seed(seed)
    model = p1_train.ChunkAdapter(arm)
    with torch.no_grad():
        model.alpha.fill_(alpha)
    torch.save({"model_state_dict": {k: v.clone() for k, v in model.state_dict().items()}, "arm": arm,
                "seed": seed, "lr": 1e-3, "step": 100, "valid": {"rollout": valid_rollout, "onestep": 0.5},
                "params": p1_train.param_report(model), "k_steps": 4, "core_out": 30}, path)


def test_p1_layout_stateful_prev_chain_and_zero_gate(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(pa, "PHASE_C_RUNS", tmp_path)
    monkeypatch.setattr(pa, "P1_RESULTS", tmp_path / "p1_results.json")
    _p1_checkpoint(torch, tmp_path / "p1_lora_s1337.pt", "lora", 1337, alpha=0.0, valid_rollout=0.40)
    _p1_checkpoint(torch, tmp_path / "p1_lora_s1338.pt", "lora", 1338, alpha=0.1, valid_rollout=0.35)
    # zero-init gate -> exact identity on the pipeline dtype, shape preserved
    adapter = pa.load_stateful_adapter(tmp_path / "p1_lora_s1337.pt")
    adapter.reset(1)
    x0 = torch.randn(16, 3, 60, 104)
    assert torch.equal(adapter(x0), x0)
    assert adapter(x0.bfloat16()).dtype == torch.bfloat16
    assert adapter.info["stateful_prev"] and adapter.info["gate_abs_max"] == 0.0
    # non-zero gate: chunk 1 depends on the chain (prev = own chunk-0 output), per sample in a batch
    trained = pa.load_stateful_adapter(tmp_path / "p1_lora_s1338.pt")
    trained.reset(2)
    a0, b0 = trained(x0), trained(x0 + 1.0)                  # chunk 0, samples 0 and 1
    a1 = trained(x0)                                          # chunk 1, sample 0 (prev = a0)
    trained.reset(1)
    fresh1 = trained(x0)                                      # chunk 0 again (prev = zeros)
    assert torch.equal(fresh1, a0) and not torch.equal(a1, a0)
    assert not torch.equal(a0, b0) and trained.calls == 1
    # best-seed selection from p1_train.aggregate()'s schema
    (tmp_path / "p1_results.json").write_text(json.dumps({"arms": {
        "lora": {"per_seed": {"1337": {"lr": 1e-3, "valid_rollout": 0.40, "valid_rollout_step_k": 0.1},
                              "1338": {"lr": 3e-3, "valid_rollout": 0.35, "valid_rollout_step_k": 0.2}},
                 "valid_rollout": {"mean": 0.375, "sd": 0.03}},
        "mr30ryd": {"per_seed": {"1337": {"valid_rollout": 0.30}}}}}))
    path, why = pa.select_ckpt("lora")
    assert path.name == "p1_lora_s1338.pt" and "0.35" in why
    with pytest.raises(pa.AdapterError):                     # best quantum seed 1337 but no ckpt file
        pa.select_ckpt("quantum")
    info = pa.check_adapter("lora")
    assert info["out_shape"] == [16, 3, 60, 104] and info["module"] == "ChunkAdapter"
    assert info["meta"]["seed"] == 1338 and info["valid"]["rollout"] == 0.35


def test_fallback_layouts(tmp_path):
    torch = pytest.importorskip("torch")
    x0 = torch.randn(16, 3, 60, 104)
    torch.save(Add(0.5), tmp_path / "whole.pt")
    module, call = pa.load_adapter(tmp_path / "whole.pt")
    assert torch.allclose(pa.StatefulAdapter(module, call)(x0), x0 + 0.5)
    torch.save({"builder": "torch.nn:Identity", "config": {}, "state_dict": {}, "call": {"returns": "residual"}},
               tmp_path / "builder.pt")
    module, call = pa.load_adapter(tmp_path / "builder.pt")
    assert torch.allclose(pa.StatefulAdapter(module, call)(x0), 2 * x0)
    torch.save({"weights": 1}, tmp_path / "bad.pt")
    with pytest.raises(pa.AdapterError):
        pa.load_adapter(tmp_path / "bad.pt")


def test_best_seed_from_results_shapes():
    assert pa.best_seed_from_results({"mr30ryd": {"best_seed": 1339}}, "quantum") == (1339, "best_seed field")
    seed, why = pa.best_seed_from_results({"lora": {"s1337": {"vrmse": 0.5}, "s1338": {"vrmse": 0.4}}}, "lora")
    assert seed == 1338 and "lowest" in why
    seed, _ = pa.best_seed_from_results({"arms": {"mr30ryd": {"per_seed": {"1337": 0.2, "1338": 0.3}}}}, "quantum")
    assert seed == 1337
    with pytest.raises(pa.AdapterError):
        pa.best_seed_from_results({"arms": {"mlp30": {}}}, "quantum")


def test_check_arm_supported(monkeypatch, tmp_path):
    runner.check_arm_supported("vanilla")
    monkeypatch.setattr(pa, "PHASE_C_RUNS", tmp_path)
    monkeypatch.setattr(pa, "P1_RESULTS", tmp_path / "p1_results.json")
    for arm in ("lora", "quantum"):
        with pytest.raises(NotImplementedError):
            runner.check_arm_supported(arm)


# ---------------------------------------------------------------- tap-point hook

class _StubScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps


class _StubPipe:
    device = "cpu"

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep, scheduler):
        return xt - flow_pred


def test_wrap_final_step_x0_adapts_only_last_timestep():
    torch = pytest.importorskip("torch")
    sched = _StubScheduler(torch.linspace(1000, 1, 1000))
    idx = (0, 250, 500, 750)
    pipe = runner.wrap_final_step_x0(_StubPipe(), lambda x0: x0 + 100.0, idx)
    flow, xt = torch.ones(2), torch.full((2,), 3.0)
    for i in idx[:-1]:
        assert torch.equal(pipe._convert_flow_pred_to_x0(flow, xt, sched.timesteps[i], sched), torch.full((2,), 2.0))
    assert torch.equal(pipe._convert_flow_pred_to_x0(flow, xt, sched.timesteps[idx[-1]], sched), torch.full((2,), 102.0))
    runner.unwrap_final_step_x0(pipe)
    assert torch.equal(pipe._convert_flow_pred_to_x0(flow, xt, sched.timesteps[idx[-1]], sched), torch.full((2,), 2.0))


def test_wrap_final_step_x0_rejects_degenerate_timesteps():
    torch = pytest.importorskip("torch")
    sched = _StubScheduler(torch.zeros(1000))
    pipe = runner.wrap_final_step_x0(_StubPipe(), lambda x0: x0)
    with pytest.raises(RuntimeError):
        pipe._convert_flow_pred_to_x0(torch.ones(1), torch.ones(1), sched.timesteps[0], sched)


def test_install_arm_adapter_identity_for_vanilla():
    pipe = _StubPipe()
    out, info = runner.install_arm_adapter(pipe, "vanilla")
    assert out is pipe and info["adapter"] is None
