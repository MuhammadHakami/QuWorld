"""Strict official-checkpoint training-data intervention probe; never a test score."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from phase_b import data_ram
from phase_b.cnext_hybrid import build_hybrid
from config_search.train_trial import sha


def difference(a, b):
    delta = (a - b).float()
    return dict(relative_l2=float(delta.norm() / a.float().norm().clamp_min(1e-12)),
                max_absolute=float(delta.abs().max()))


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cudnn.benchmark = False
    train, _ = data_ram.load()
    count = min(6, len(train))
    if count < 2:
        raise ValueError('Need distinct training trajectories')
    inputs = train[:count, :4].flatten(1, 2).to('cuda')
    rows = []
    for seed in (1337, 1338, 1339):
        checkpoint = ROOT / f'runs/phase_b2/mr30ryd_s{seed}.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = build_hybrid('mr30ryd')
        model.load_state_dict(state['model_state_dict'], strict=True)
        del state
        model = model.to('cuda').eval()
        adapter = model.neck.bottleneck
        captured = {}
        def capture(module, args, output):
            captured['context'] = args[0].detach()
            captured['readout'] = output.detach()
        hook = adapter.core.register_forward_hook(capture)
        try:
            original = model(inputs)
        finally:
            hook.remove()
        repeated = model(inputs)
        repeat_difference = difference(original, repeated)
        context = captured['context']
        readout = captured['readout']
        interventions = {}
        for name in ('zero_readout', 'shuffle_context'):
            if name == 'zero_readout':
                hook = adapter.core.register_forward_hook(lambda m, a, o: torch.zeros_like(o))
            else:
                hook = adapter.core.register_forward_pre_hook(lambda m, a: (a[0].roll(1, 0),))
            try:
                changed = model(inputs)
            finally:
                hook.remove()
            measured = difference(original, changed)
            measured['detectable'] = (
                measured['relative_l2'] > max(1e-5, 10 * repeat_difference['relative_l2'])
                and measured['max_absolute'] > max(1e-6, 10 * repeat_difference['max_absolute']))
            interventions[name] = measured
        detunings = torch.cat([r.cond(context).flatten(1) for r in adapter.core.regs], -1)
        residual = adapter.alpha * adapter.up(readout)
        row = dict(seed=seed, checkpoint=str(checkpoint.relative_to(ROOT)),
                   checkpoint_sha256=sha(checkpoint), strict_load=True,
                   context_rms=float(context.square().mean().sqrt()),
                   context_between_sample_std=float(context.std(0, unbiased=False).mean()),
                   detuning_between_sample_std=float(detunings.std(0, unbiased=False).mean()),
                   readout_between_sample_std=float(readout.std(0, unbiased=False).mean()),
                   alpha_rms=float(adapter.alpha.square().mean().sqrt()),
                   residual_rms=float(residual.square().mean().sqrt()),
                   repeat_difference=repeat_difference, interventions=interventions)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, adapter, original, repeated, changed, captured, context, readout, residual, detunings
        torch.cuda.empty_cache()
    result = dict(kind='training_data_mechanism_probe', device='cuda', test_access=False,
                  trajectories=list(range(count)), history_frames=[0, 1, 2, 3],
                  data_cache=dict(path=str(data_ram.CACHE.relative_to(ROOT)),
                                  size_bytes=data_ram.CACHE.stat().st_size,
                                  mtime_ns=data_ram.CACHE.stat().st_mtime_ns),
                  source_sha256=sha(__file__), rows=rows,
                  interpretation='Diagnostic only; no benchmark accuracy or compute comparison')
    out = ROOT / 'runs/config_search/mechanism_probe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(out)


if __name__ == '__main__':
    main()
