"""Strict paired audit of registered long-routing results; never promotes a winner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=[('vanilla','adaptive')]+[(a,r) for a in ('quantum','classical') for r in ('adaptive','static','permuted')]
STEPS=[0,256,512,768,1024]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_curve(curve,ids,starts,budgets):
    if sorted(w['start'] for w in curve['windows'])!=starts:
        raise ValueError('Missing, duplicate or unexpected temporal windows')
    output={b:{} for b in budgets};costs={b:[] for b in budgets}
    for window in curve['windows']:
        if window['trajectory_ids']!=ids:raise ValueError('Trajectory identity/order mismatch')
        if sorted(r['budget_registers'] for r in window['allocations'])!=budgets:
            raise ValueError('Missing, duplicate or unexpected allocation budgets')
        for row in window['allocations']:
            b=row['budget_registers'];losses=row['vrmse8'];trace=row['active_qubits_by_step']
            if len(losses)!=len(ids) or not all(math.isfinite(x) and x>=0 for x in losses):
                raise ValueError('Invalid or missing trajectory losses')
            if len(trace)!=8 or any(len(t)!=len(ids) for t in trace):
                raise ValueError('Incomplete allocation trace')
            for t in trace:
                if any(type(q) is not int or q<0 or q>b*10 or q%10 for q in t):
                    raise ValueError('Invalid active-qubit count')
                costs[b].extend(t)
            for trajectory,loss in zip(ids,losses):output[b][(trajectory,window['start'])]=loss
    return output,costs


def paired(candidate,reference):
    if candidate.keys()!=reference.keys():raise ValueError('Unpaired windows or trajectories')
    # Average windows inside each trajectory before aggregating. No window-level
    # significance test or independent-seed claim is manufactured from one seed.
    ids=sorted({key[0] for key in candidate})
    differences=[];relative=[]
    for trajectory in ids:
        keys=[k for k in candidate if k[0]==trajectory]
        c=statistics.mean(candidate[k] for k in keys)
        r=statistics.mean(reference[k] for k in keys)
        differences.append(r-c)
        relative.append((r-c)/max(r,1e-12))
    return dict(mean_absolute_gain=statistics.mean(differences),
                mean_relative_gain=statistics.mean(relative),trajectory_relative_gains=relative,
                trajectory_ids=ids,improved_trajectories=sum(v>0 for v in differences),
                independent_seeds=1,significance_claim=False)


def audit(folder,seed=1337):
    expected=[folder/f'{a}_{r}_s{seed}' for a,r in ARMS]
    missing=[str(p/'result.json') for p in expected if not (p/'result.json').exists()]
    if missing:return dict(status='pending',missing=missing,promotion=False)
    import torch
    matrices={};summaries=[];provenance={};reference_protocol=None;reference_counts=None
    for (arm,routing),path in zip(ARMS,expected):
        result=json.loads((path/'result.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        if result['status']!='complete' or result['step']!=1024 or protocol['smoke']:
            raise ValueError('Smoke/partial run cannot establish full-trial completion')
        ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if ph!=result['protocol_hash']:raise ValueError('Protocol hash mismatch')
        if protocol['arm']!=arm or protocol['routing']!=routing or protocol['seed']!=seed:
            raise ValueError('Unexpected arm/routing/seed')
        common={k:v for k,v in protocol.items() if k not in ('arm','routing','total_params')}
        if reference_protocol is None:reference_protocol=common
        elif common!=reference_protocol:raise ValueError('Compared arms have different protocols')
        if [c['step'] for c in result['curves']]!=STEPS:raise ValueError('Incomplete learning curve')
        checkpoint=torch.load(path/'latest.pt',map_location='cpu',weights_only=True,mmap=True)
        if checkpoint['step']!=1024 or checkpoint['protocol_hash']!=ph or checkpoint['curves']!=result['curves']:
            raise ValueError('Final checkpoint does not support reported result')
        counts=checkpoint['budget_counts']
        if sum(counts)!=1024*2:raise ValueError('Wrong number of sampled training budgets')
        if reference_counts is None:reference_counts=counts
        elif counts!=reference_counts:raise ValueError('Different sampled budgets across arms')
        del checkpoint
        ids=protocol['validation_ids'];nframes=protocol['valid_shape'][1]
        # Same nonnegative integer linspace truncation as the CUDA trainer's
        # metadata schedule for three starts (no model inference here).
        starts=sorted(set([0,(nframes-12)//2,nframes-12]))
        budgets=[0] if arm=='vanilla' else [0,1,2,3]
        for curve in result['curves']:
            if json.loads((path/f"eval_{curve['step']}.json").read_text())!=curve:
                raise ValueError('Evaluation snapshot differs from result')
            values,costs=extract_curve(curve,ids,starts,budgets)
            for budget,scores in values.items():
                matrices[(arm,routing,curve['step'],budget)]=scores
                summaries.append(dict(arm=arm,routing=routing,step=curve['step'],budget=budget,
                    mean_vrmse8=statistics.mean(scores.values()),mean_active_qubits=statistics.mean(costs[budget]),
                    total_params=protocol['total_params']))
        provenance[str(path)]=dict(result_sha256=digest(path/'result.json'),protocol_sha256=digest(path/'protocol.json'))
    initial=matrices[('vanilla','adaptive',0,0)]
    for key,values in matrices.items():
        if key[2]==0 and any(abs(values[k]-initial[k])>1e-6 for k in initial):
            raise ValueError('Initial baseline differs across arms/allocations')
    comparisons={}
    for step in STEPS[1:]:
        for budget in range(4):
            q=matrices[('quantum','adaptive',step,budget)]
            for name,ref in [('classical',matrices[('classical','adaptive',step,budget)]),
                             ('permuted_quantum',matrices[('quantum','permuted',step,budget)]),
                             ('static_quantum',matrices[('quantum','static',step,budget)]),
                             ('own_zero_budget',matrices[('quantum','adaptive',step,0)]),
                             ('vanilla',matrices[('vanilla','adaptive',step,0)])]:
                comparisons[f'step{step}_budget{budget}_vs_{name}']=paired(q,ref)
    return dict(status='complete_exploratory_audit',promotion=False,independent_seeds=1,
                summaries=summaries,comparisons=comparisons,provenance=provenance,
                interpretation='Positive gain favors adaptive quantum; allocation effects and backbone-training differences are separate. No compute-optimality or cross-task claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    result=audit(args.folder)
    args.folder.mkdir(parents=True,exist_ok=True)
    (args.folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],promotion=result['promotion'],missing=result.get('missing',[])),indent=2))
