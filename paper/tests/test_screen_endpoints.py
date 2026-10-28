"""Prevent checkpoint selection from masquerading as fixed-step evidence."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_search.audit_screen_endpoints import audit, endpoint


def test_best_and_endpoint_rank_can_disagree(tmp_path):
    jobs = []
    for index in range(3):
        dest = tmp_path / str(index)
        dest.mkdir()
        curves = []
        for step, score in [(256, index + 1), (1024, 3 - index)]:
            job = dict(name=str(index), arm='quantum', target_steps=step, out=str(dest))
            jobs.append(job)
            curves.append(dict(step=step, vrmse8=score))
            (dest / f'result_at_{step}.json').write_text(json.dumps({**job, 'result': dict(
                status='complete', steps=step, best_validation_vrmse8=.1 * (index + 1))}))
        (dest / 'curve.jsonl').write_text('\n'.join(map(json.dumps, curves)))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(jobs=jobs, budgets=[256, 1024])))
    result = audit(tmp_path)
    assert result['endpoint_fidelity']['spearman'] == -1
    assert result['endpoint_fidelity']['short_fidelity_gate'] is False
    assert result['promotion'] is False
    (tmp_path / '0' / 'result_at_1024.json').unlink()
    result = audit(tmp_path)
    assert result['status'] == 'pending'
    assert 'endpoint_fidelity' not in result


@pytest.mark.parametrize('values', [[], [float('nan')], [-1], [1, 2]])
def test_invalid_endpoint_rejected(tmp_path, values):
    path = tmp_path / 'curve.jsonl'
    path.write_text('\n'.join(json.dumps(dict(step=256, vrmse8=x)) for x in values))
    with pytest.raises(ValueError):
        endpoint(path, 256)
