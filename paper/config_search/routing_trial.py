"""Resumable long routing comparison under document28; no official test access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'))
from config_search.core import RegisterConfig
from config_search.signal_router import ControlledSignalRouter
from config_search.train_trial import atomic_save,sha,train_loss,validation_ids
from config_search.allocation_pilot import evaluate
from phase_b import data_ram
from phase_b.cnext_hybrid import load_official,NeckWithBottleneck
from model import _count


@torch.no_grad()
def measure(model,adapter,valid,ids):
    model.eval()
    starts=sorted(set(torch.linspace(0,valid.shape[1]-12,3).long().tolist()))
    rows=[]
    for start in starts:
        if adapter is not None:
            values=evaluate(model,adapter,valid[ids,start:start+12])
        else:
            history=valid[ids,start:start+4].flatten(1,2).cuda()
            for _ in range(8):
                pred=model(history)
                history=torch.cat([history[:,4:],pred],1)
            target=valid[ids,start+11].cuda()
            variance=(target-target.mean((2,3),keepdim=True)).square().mean((2,3))
            loss=((pred-target).square().mean((2,3))/(variance+1e-7)).sqrt().mean(1)
            values=[dict(budget_registers=0,vrmse8=loss.tolist(),mean_vrmse8=float(loss.mean()),
                         active_qubits_by_step=[[0]*len(ids) for _ in range(8)])]
        rows.append(dict(start=start,trajectory_ids=ids,allocations=values))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['vanilla','quantum','classical'],required=True)
    p.add_argument('--routing',choices=['adaptive','static','permuted'],default='adaptive')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--max-wall-seconds',type=float,default=7200)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    if args.steps<=0 or args.max_wall_seconds<=0: p.error('Positive budgets required')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    train,valid=data_ram.load()
    train=train[:48]
    if args.smoke:
        train,valid=train[:2,:12],valid[:3,:12]
        args.steps=2
    ids=validation_ids(len(valid),'search')
    predictor=ROOT/'runs/config_search/signal_routing_probe/difficulty_predictor.pt'
    torch.manual_seed(args.seed)
    model,upstream=load_official(str(ROOT/'data/cnext_ckpt'))
    adapter=None
    if args.arm!='vanilla':
        torch.manual_seed(args.seed)
        adapter=ControlledSignalRouter(upstream['init_features']*2**upstream['stages'],
                   RegisterConfig(geometry='bounded_dynamic',encoding_rms=.1),args.arm,routing=args.routing)
        adapter.score.load_state_dict(torch.load(predictor,map_location='cpu',weights_only=True)['score_state_dict'],strict=True)
        adapter.score.requires_grad_(False); adapter.warmup=True
        model.neck=NeckWithBottleneck(model.neck,adapter)
    model=model.cuda()
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01)
    batch_rng=torch.Generator().manual_seed(args.seed)
    budget_rng=torch.Generator().manual_seed(904)
    sources=[__file__,ROOT/'paper/config_search/signal_router.py',ROOT/'paper/config_search/core.py',
             ROOT/'paper/config_search/train_trial.py',ROOT/'paper/config_search/allocation_pilot.py',
             ROOT/'paper/qlayer.py',ROOT/'paper/model.py',ROOT/'paper/phase_b/cnext_hybrid.py',
             ROOT/'paper/phase_b/data_ram.py',ROOT/'28-long-routing-comparison.md',
             predictor,ROOT/'data/cnext_ckpt/model.safetensors',data_ram.STATS]
    protocol=dict(arm=args.arm,routing=args.routing,seed=args.seed,smoke=args.smoke,
                  batch=2,lr=1e-4,td_k=4,train_shape=list(train.shape),valid_shape=list(valid.shape),
                  validation_ids=ids,total_params=_count(model),source_hashes={str(f):sha(f) for f in sources},
                  data_cache=dict(size=data_ram.CACHE.stat().st_size,mtime_ns=data_ram.CACHE.stat().st_mtime_ns))
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    latest=args.out/'latest.pt'
    step=0; previous_wall=0.
    curves=[];counts=[0]*4
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=True)
        if state['protocol_hash']!=digest: raise ValueError('Resume protocol mismatch')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        step=state['step'];curves=state['curves'];counts=state['budget_counts'];previous_wall=state['wall_seconds']
        batch_rng.set_state(state['batch_rng']);budget_rng.set_state(state['budget_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    elif (args.out/'protocol.json').exists():
        raise ValueError('Metadata without checkpoint; use a fresh output folder')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def save():
        atomic_save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
              protocol_hash=digest,step=step,curves=curves,budget_counts=counts,batch_rng=batch_rng.get_state(),
              budget_rng=budget_rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
              wall_seconds=previous_wall+time.perf_counter()-started),latest)
    def evaluate_now():
        rows=measure(model,adapter,valid,ids)
        curves.append(dict(step=step,windows=rows))
        (args.out/f'eval_{step}.json').write_text(json.dumps(curves[-1],indent=2)+'\n')
        print(f'{args.arm}/{args.routing} evaluated step={step}',flush=True)
        model.train()
    if not curves:
        evaluate_now();save()
    model.train()
    while step<args.steps and time.perf_counter()-started<args.max_wall_seconds:
        budgets=torch.randint(4,(2,),generator=budget_rng)
        for b in budgets.tolist(): counts[b]+=1
        if adapter is not None: adapter.register_budget=budgets
        optimizer.zero_grad(set_to_none=True)
        loss=train_loss(model,train,batch_rng,2)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
        if step%256==0 or step==args.steps:evaluate_now()
        if step%128==0:save()
    save()
    complete=step>=args.steps
    result=dict(status=('smoke_complete' if args.smoke else 'complete') if complete else 'incomplete',
                step=step,protocol_hash=digest,arm=args.arm,routing=args.routing,
                test_access=False,curves=curves,wall_seconds=previous_wall+time.perf_counter()-started)
    tmp=args.out/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.out/'result.json')
    if not complete:raise SystemExit(3)


if __name__=='__main__':main()
