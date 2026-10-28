"""Report direct allocation effects separately from differences between trained arms."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(folder):
    result=json.loads((folder/'result.json').read_text())
    summary={'kind':'allocation_pilot_audit','promotion':False,'arms':{}}
    starts=[]
    budgets_seen=[]
    for arm in result['results']:
        before=arm['before']; after=arm['after']
        expected=len(result['probe_training_trajectories'])
        for row in before+after:
            values=row['vrmse8']
            if len(values)!=expected or not all(math.isfinite(v) and v>=0 for v in values):
                raise ValueError('Missing/nonfinite per-trajectory results')
            counts=row['active_qubits_by_step']
            if len(counts)!=8 or any(len(step)!=expected for step in counts):
                raise ValueError('Incomplete rollout allocation trace')
        if [r['budget_registers'] for r in after] != [0,1,2,3]:
            raise ValueError('Need all four budgets')
        starts.append(before[0]['vrmse8'])
        if any(r['vrmse8'] != before[0]['vrmse8'] for r in before):
            raise ValueError('Zero-gate baseline differs across budgets')
        budgets_seen.append(arm['training_budget_counts'])
        baseline=after[0]['vrmse8']
        changes=[]
        for row in after:
            cap=row['budget_registers']*10
            if any(q>cap or q<0 for step in row['active_qubits_by_step'] for q in step):
                raise ValueError('Allocation cap violated')
            gain=[(b-e)/max(b,1e-12) for b,e in zip(baseline,row['vrmse8'])]
            changes.append(dict(budget_registers=row['budget_registers'],
                                mean_vrmse8=statistics.mean(row['vrmse8']),
                                mean_paired_relative_gain_vs_own_zero_budget=statistics.mean(gain),
                                min_relative_gain=min(gain),max_relative_gain=max(gain)))
        oracle=[min(range(4),key=lambda b:after[b]['vrmse8'][i]) for i in range(len(baseline))]
        summary['arms'][arm['arm']]=dict(allocation_effects=changes,
                  retrospective_argmin_budget_counts=[oracle.count(b) for b in range(4)],
                  oracle_is_deployable=False)
    if any(counts != budgets_seen[0] for counts in budgets_seen):
        raise ValueError('Training budget samples differ across arms')
    if any(values != starts[0] for values in starts):
        raise ValueError('Arms do not share the measured initial baseline')
    q=summary['arms']['quantum']['allocation_effects']
    c=summary['arms']['classical']['allocation_effects']
    summary['between_trained_arms_relative_gain']={str(b):(c[b]['mean_vrmse8']-q[b]['mean_vrmse8'])/c[b]['mean_vrmse8'] for b in range(4)}
    summary['interpretation']='Positive gain favors quantum or more allocation; one-seed feasibility only. Budget-zero between-arm effects can arise from backbone training.'
    (folder/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    print(json.dumps(summarize(parser.parse_args().folder),indent=2))
