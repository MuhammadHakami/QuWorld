"""Audit cannot convert missing controls or malformed traces into evidence."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config_search.audit_long_routing import audit,extract_curve,paired


def test_missing_campaign_is_pending(tmp_path):
    result=audit(tmp_path)
    assert result['status']=='pending' and len(result['missing'])==7
    assert result['promotion'] is False


def example():
    return dict(step=1024,windows=[dict(start=0,trajectory_ids=[0,1],allocations=[
        dict(budget_registers=1,vrmse8=[.2,.3],active_qubits_by_step=[[10,0] for _ in range(8)])])])


def test_valid_trace_has_paired_identities():
    values,costs=extract_curve(example(),[0,1],[0],[1])
    assert values[1]=={(0,0):.2,(1,0):.3}
    assert len(costs[1])==16


@pytest.mark.parametrize('failure',['count','budget','duplicate','nonfinite','identity'])
def test_reject_incomplete_or_invalid_evidence(failure):
    curve=copy.deepcopy(example());window=curve['windows'][0];row=window['allocations'][0]
    if failure=='count':row['active_qubits_by_step'].pop()
    if failure=='budget':row['active_qubits_by_step'][0][0]=20
    if failure=='duplicate':curve['windows'].append(copy.deepcopy(window))
    if failure=='nonfinite':row['vrmse8'][0]=float('nan')
    if failure=='identity':window['trajectory_ids']=[1,0]
    with pytest.raises(ValueError):extract_curve(curve,[0,1],[0],[1])


def test_windows_do_not_become_independent_seeds():
    q={(0,0):1.,(0,2):3.,(1,0):2.,(1,2):2.}
    c={k:v+1 for k,v in q.items()}
    result=paired(q,c)
    assert result['mean_absolute_gain']==1.
    assert len(result['trajectory_relative_gains'])==2
    assert result['independent_seeds']==1 and not result['significance_claim']
    with pytest.raises(ValueError):paired(q,{(0,0):2.})
