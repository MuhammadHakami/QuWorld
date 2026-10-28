"""Read-only endpoint audit; never select a best-so-far checkpoint as an endpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def rank(values):
    return [1 + sum(y < x for y in values) + (sum(y == x for y in values) - 1) / 2
            for x in values]


def endpoint(path, step):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scores = [float(row['vrmse8']) for row in rows if row['step'] == step]
    if not scores or any(not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError(f'Missing or invalid endpoint: {path}, step={step}')
    if any(not math.isclose(x, scores[0], rel_tol=1e-7, abs_tol=1e-9) for x in scores):
        raise ValueError(f'Conflicting repeated endpoint evaluations: {path}, step={step}')
    return scores[-1]


def audit(folder):
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows, pending = [], []
    for job in manifest['jobs']:
        step = job['target_steps']
        dest = Path(job['out'])
        snapshot_path = dest / f'result_at_{step}.json'
        identity = {k: job[k] for k in ('name', 'arm', 'target_steps')}
        if not snapshot_path.exists():
            pending.append(identity)
            continue
        snapshot = json.loads(snapshot_path.read_text())
        result = snapshot['result']
        if any(snapshot[k] != job[k] for k in identity):
            raise ValueError(f'Snapshot identity mismatch: {snapshot_path}')
        if result['status'] != 'complete' or result['steps'] != step:
            raise ValueError(f'Incomplete snapshot: {snapshot_path}')
        value = endpoint(dest / 'curve.jsonl', step)
        best = float(result['best_validation_vrmse8'])
        if not math.isfinite(best) or best < 0 or best > value + 1e-7:
            raise ValueError(f'Invalid best-so-far score: {snapshot_path}')
        rows.append({**identity, 'endpoint_vrmse8': value, 'best_so_far_vrmse8': best,
                     'endpoint_minus_best': value - best,
                     'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest()})
    report = dict(status='pending' if pending else 'complete_exploratory', rows=rows,
                  pending=pending, promotion=False,
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  interpretation='Fixed-step scores; one seed, no confirmation or compute claim.')
    budgets = manifest['budgets']
    if not pending and len(budgets) == 2:
        groups = [{r['name']: r['endpoint_vrmse8'] for r in rows
                   if r['arm'] == 'quantum' and r['target_steps'] == b} for b in budgets]
        early, late = groups
        names = sorted(early)
        if set(early) != set(late) or len(names) < 3:
            raise ValueError('Need matching quantum candidate sets')
        try:
            rho = statistics.correlation(rank([early[n] for n in names]), rank([late[n] for n in names]))
        except statistics.StatisticsError:
            rho = None
        top = [n for n in names if early[n] <= sorted(early.values())[1]]
        winners = [n for n in names if late[n] == min(late.values())]
        recovered = len(top) <= 2 and all(n in top for n in winners)
        report['endpoint_fidelity'] = dict(spearman=rho, early_top_two_including_ties=top,
                                          late_winners=winners, top_two_recovered=recovered,
                                          short_fidelity_gate=rho is not None and rho >= .7 and recovered)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.folder)
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload)
