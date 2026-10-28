"""Registered 64-step shared-allocation pilot; no official validation/test access."""
from pathlib import Path
import json
import argparse
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'paper'))
from config_search.core import RegisterConfig
from config_search.signal_router import SignalRouter, RelativeSignalRouter
from config_search.train_trial import train_loss, sha, atomic_save
from phase_b import data_ram
from phase_b.cnext_hybrid import load_official, NeckWithBottleneck
from model import _count


@torch.no_grad()
def evaluate(model, adapter, trajectories):
    model.eval()
    rows = []
    for budget in range(4):
        adapter.register_budget = budget
        history = trajectories[:, :4].flatten(1,2).cuda()
        allocations = []
        for _ in range(8):
            pred = model(history)
            allocations.append(adapter.last_route['active_qubits'].tolist())
            history = torch.cat([history[:,4:],pred],1)
        target = trajectories[:,11].cuda()
        mse = (pred-target).square().mean((2,3))
        variance = (target-target.mean((2,3),keepdim=True)).square().mean((2,3))
        loss = (mse/(variance+1e-7)).sqrt().mean(1)
        rows.append(dict(budget_registers=budget, vrmse8=loss.tolist(),
                         mean_vrmse8=float(loss.mean()), active_qubits_by_step=allocations))
    adapter.register_budget = None
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['legacy','relative'], default='legacy')
    parser.add_argument('--out', type=Path)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cudnn.benchmark=False
    out=args.out or ROOT/('runs/config_search/allocation_relative_v1' if args.variant=='relative' else 'runs/config_search/allocation_pilot_v1')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() or any(out.glob('*_checkpoint.pt')):
        raise RuntimeError('Existing pilot evidence; choose a new registered experiment rather than overwrite')
    train,_=data_ram.load()
    predictor=ROOT/'runs/config_search/signal_routing_probe/difficulty_predictor.pt'
    predictor_state=torch.load(predictor,map_location='cpu',weights_only=True)['score_state_dict']
    results=[]
    for arm in ('quantum','classical'):
        torch.manual_seed(20260920)
        model,cfg=load_official(str(ROOT/'data/cnext_ckpt'))
        channels=cfg['init_features']*2**cfg['stages']
        torch.manual_seed(20260920)
        router_class=RelativeSignalRouter if args.variant=='relative' else SignalRouter
        adapter=router_class(channels,RegisterConfig(geometry='bounded_dynamic',encoding_rms=.1),arm=arm)
        adapter.score.load_state_dict(predictor_state,strict=True)
        adapter.score.requires_grad_(False)
        adapter.warmup=True
        model.neck=NeckWithBottleneck(model.neck,adapter)
        model=model.cuda()
        before=evaluate(model,adapter,train[48:60])
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        batch_rng=torch.Generator().manual_seed(20260920)
        budget_rng=torch.Generator().manual_seed(904)
        started=time.perf_counter()
        model.train()
        budget_counts=[0]*4
        for step in range(64):
            budgets=torch.randint(4,(2,),generator=budget_rng)
            for b in budgets.tolist():
                budget_counts[b]+=1
            adapter.register_budget=budgets
            optimizer.zero_grad(set_to_none=True)
            loss=train_loss(model,train[:48],batch_rng,2)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%16==0:
                print(f'{arm} step={step+1} loss={float(loss.detach()):.6f}',flush=True)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        after=evaluate(model,adapter,train[48:60])
        row=dict(arm=arm,variant=args.variant,steps=64,total_params=_count(model),core_parameter_gaps=adapter.core_parameter_gaps,
                 training_budget_counts=budget_counts,before=before,after=after,
                 shared_gpu_training_wall_seconds=elapsed,
                 alpha_rms=float(adapter.alpha.detach().square().mean().sqrt()))
        atomic_save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                         row=row),out/f'{arm}_checkpoint.pt')
        (out/f'{arm}_result.json').write_text(json.dumps(row,indent=2)+'\n')
        results.append(row)
        print(json.dumps(dict(arm=arm,after_means=[r['mean_vrmse8'] for r in after])),flush=True)
        del model,adapter,optimizer,loss
        torch.cuda.empty_cache()
    result=dict(kind='shared_allocation_feasibility_pilot',variant=args.variant,test_access=False,validation_access=False,
                training_trajectories=list(range(48)),probe_training_trajectories=list(range(48,60)),
                seed=20260920,results=results,source_sha256=sha(__file__),
                router_sha256=sha(ROOT/'paper/config_search/signal_router.py'),
                difficulty_predictor_sha256=sha(predictor),
                protocol_sha256=sha(ROOT/'27-causal-signal-routing.md'),
                limitation='One seed, short budget, inference interventions on shared-budget training; no promotion')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
