"""Small GPU experiment: predict classical error from causal backbone latents.

Training split only. This tests the difficulty predictor, not quantum benefit.
No changes to existing campaign checkpoints or configuration-screen protocol.
"""
from pathlib import Path
import json
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from config_search.core import RegisterConfig
from config_search.signal_router import SignalRouter
from config_search.train_trial import sha
from phase_b import data_ram
from phase_b.cnext_hybrid import load_official


@torch.no_grad()
def metrics(logits, labels):
    probabilities = logits.sigmoid().flatten()
    y = labels.flatten().bool()
    positive, negative = probabilities[y], probabilities[~y]
    auc = None
    if len(positive) and len(negative):
        comparisons = positive[:, None] - negative[None, :]
        auc = float(((comparisons > 0).float() + .5 * (comparisons == 0).float()).mean())
    return dict(bce=float(F.binary_cross_entropy_with_logits(logits, labels)),
                prevalence=float(labels.mean()), auroc=auc,
                accuracy=float(((probabilities > .5) == y).float().mean()))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(20260920)
    torch.cuda.set_per_process_memory_fraction(.15)
    train, _ = data_ram.load()
    teacher, cfg = load_official(str(ROOT / 'data/cnext_ckpt'))
    teacher = teacher.cuda().eval()
    contexts, errors = [], []
    captured = []
    hook = teacher.neck.register_forward_hook(lambda m, a, o: captured.append(o.detach()))
    try:
        with torch.no_grad():
            for i in range(0, 24, 2):
                # Frames0..3 are available at inference; frame4 is a training label only.
                pred = teacher(train[i:i+2, :4].flatten(1, 2).cuda())
                latent = captured.pop()
                mse = (pred - train[i:i+2, 4].cuda()).square().mean(1, keepdim=True)
                errors.append(F.adaptive_avg_pool2d(mse, latent.shape[2:]).flatten(1))
                contexts.append(latent)
    finally:
        hook.remove()
    context, error = torch.cat(contexts), torch.cat(errors)
    del teacher, contexts, errors, captured
    threshold = error[:12].median()
    labels = (error > threshold).float()
    router = SignalRouter(context.shape[1], RegisterConfig(geometry='bounded_dynamic', encoding_rms=.1)).cuda()
    with torch.no_grad():
        features = router.features(context)[2].detach()
    initial = metrics(router.score(features[12:]).squeeze(-1), labels[12:])
    optimizer = torch.optim.AdamW(router.score.parameters(), lr=1e-3, weight_decay=.01)
    inputs = features[:12].flatten(0, 1)
    targets = labels[:12].flatten()
    for step in range(200):
        ids = torch.randint(len(inputs), (256,), device='cuda')
        loss = F.binary_cross_entropy_with_logits(router.score(inputs[ids]).squeeze(-1), targets[ids])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    router.eval()
    with torch.no_grad():
        logits = router.score(features[12:]).squeeze(-1)
        heldout = metrics(logits, labels[12:])
        prior = labels[:12].mean().clamp(1e-6, 1-1e-6)
        constant_bce = float(F.binary_cross_entropy(prior.expand_as(labels[12:]), labels[12:]))
        selected = logits.topk(3, dim=1).indices
        selected_prevalence = float(labels[12:].gather(1, selected).mean())
        # Actual dispatch through 30-qubit-capable adapter, still zero-gated.
        output = router(context[12:])
        unchanged = torch.equal(output, context[12:])
        allocation = router.last_route['active_qubits'].tolist()
    result = dict(kind='difficulty_predictor_feasibility', seed=20260920, device='cuda',
                  fitting_training_trajectories=list(range(12)),
                  heldout_training_trajectories=list(range(12,24)),
                  history_frames=[0,1,2,3], training_target_frame=4,
                  test_access=False, validation_access=False,
                  steps=200, training_error_threshold=float(threshold), initial=initial,
                  heldout=heldout, constant_prior_bce=constant_bce,
                  selected_top3_error_exceedance_rate=selected_prevalence,
                  active_qubits_per_input=allocation, zero_gate_identity=unchanged,
                  quantum_accuracy_improvement_tested=False,
                  upstream_sha256=sha(ROOT/'data/cnext_ckpt/model.safetensors'),
                  router_sha256=sha(ROOT/'paper/config_search/signal_router.py'), source_sha256=sha(__file__),
                  limitation='One seed and one window on training trajectories; error is not quantum benefit')
    out = ROOT/'runs/config_search/signal_routing_probe'
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(score_state_dict=router.score.state_dict(), error_threshold=threshold.cpu(),
                    metadata=result), out/'difficulty_predictor.pt')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
