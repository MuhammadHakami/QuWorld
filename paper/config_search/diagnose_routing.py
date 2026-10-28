"""Training-data signal and gradient audit of the completed routing pilot."""
from pathlib import Path
import json
import argparse
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'))
from config_search.core import RegisterConfig
from config_search.signal_router import SignalRouter, RelativeSignalRouter
from config_search.train_trial import sha
from phase_b.cnext_hybrid import load_official, NeckWithBottleneck
from phase_b import data_ram


def stats(x):
    x=x.detach().float()
    return dict(rms=float(x.square().mean().sqrt()),
                between_sample_std=float(x.std(0,unbiased=False).mean()) if len(x)>1 else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/config_search/allocation_pilot_v1')
    args=parser.parse_args()
    metadata=json.loads((args.run/'result.json').read_text())
    router_class=RelativeSignalRouter if metadata.get('variant')=='relative' else SignalRouter
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(.15)
    train,_=data_ram.load()
    rows=[]
    for arm in ('quantum','classical'):
        path=args.run/f'{arm}_checkpoint.pt'
        model,cfg=load_official(str(ROOT/'data/cnext_ckpt'))
        adapter=router_class(cfg['init_features']*2**cfg['stages'],RegisterConfig(geometry='bounded_dynamic',encoding_rms=.1),arm)
        model.neck=NeckWithBottleneck(model.neck,adapter)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
        model=model.cuda().eval(); adapter.register_budget=3
        row=dict(arm=arm,checkpoint_sha256=sha(path),registers={})
        hooks=[]
        def adapter_hook(m,a,o):
            row['latent']=stats(a[0]); row['correction']=stats(o-a[0])
            row['correction_to_latent_rms']=row['correction']['rms']/row['latent']['rms']
            row['corrected_location_fraction']=float(((o-a[0]).abs().sum(1)>0).float().mean())
        hooks.append(adapter.register_forward_hook(adapter_hook))
        for i,core in enumerate(adapter.cores):
            def core_hook(m,a,o,index=i):
                values=dict(input=stats(a[0]),readout=stats(o))
                if arm=='quantum':
                    reg=m.regs[0]
                    raw=reg.cond(a[0])
                    values['encoding']=stats(raw)
                    values['encoding_tanh_saturation_fraction']=float((raw.detach().abs()>2).float().mean())
                    values['positions_between_sample_std']=stats(reg.positions(a[0]))['between_sample_std']
                row['registers'][str(index)]=values
            hooks.append(core.register_forward_hook(core_hook))
        try:
            pred=model(train[:6,:4].flatten(1,2).cuda())
            loss=(pred-train[:6,4].cuda()).square().mean()
            loss.backward()
            row['one_step_training_mse']=float(loss.detach())
            row['alpha']=stats(adapter.alpha[None,:])
            row['gradient_norms']={name:float(p.grad.norm()) for name,p in adapter.named_parameters()
                                   if p.grad is not None and (name=='alpha' or 'cond.weight' in name or 'geo.2.weight' in name or name=='projections.0.weight')}
        finally:
            for hook in hooks: hook.remove()
        rows.append(row)
        del model,adapter,pred,loss
        torch.cuda.empty_cache()
    out=args.run/'signal_audit.json'
    out.write_text(json.dumps(dict(training_trajectories=list(range(6)),test_access=False,
                                   source_sha256=sha(__file__),results=rows),indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':
    main()
