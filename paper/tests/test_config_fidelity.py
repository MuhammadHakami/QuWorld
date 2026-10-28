"""Guard against falsely trusting short runs or tied rankings."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_search.summarize_screen import summarize


@pytest.mark.parametrize('early,late,expected', [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], True),
    ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], False),
    ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], False),
])
def test_fidelity_gate(tmp_path, early, late, expected):
    jobs = []
    for budget, scores in [(256, early), (1024, late)]:
        for index, score in enumerate(scores):
            out = tmp_path / str(index)
            out.mkdir(exist_ok=True)
            job = dict(target_steps=budget, out=str(out), name=str(index), arm='quantum')
            jobs.append(job)
            (out / f'result_at_{budget}.json').write_text(json.dumps(dict(result=dict(
                status='complete', steps=budget, best_validation_vrmse8=score))))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(budgets=[256,1024], jobs=jobs)))
    result = summarize(tmp_path)
    assert result['short_fidelity_gate'] is expected
    assert result['promotion'] is False
