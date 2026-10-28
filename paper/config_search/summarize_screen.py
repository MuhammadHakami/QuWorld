"""Exploratory fidelity agreement from preserved fixed-budget results."""
import argparse
import json
from pathlib import Path
import statistics


def ranks(values):
    ordered = sorted(values)
    return [(ordered.index(x) + len(ordered) - ordered[::-1].index(x) + 1) / 2 for x in values]


def summarize(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    budgets = manifest['budgets']
    if len(budgets) != 2:
        return {'status': 'requires_two_fidelities', 'promotion': False}
    groups = {b: {} for b in budgets}
    for job in manifest['jobs']:
        b = job['target_steps']
        snapshot = json.loads((Path(job['out']) / f'result_at_{b}.json').read_text())
        result = snapshot['result']
        if result['status'] != 'complete' or result['steps'] != b:
            raise ValueError('Incomplete or mismatched stage')
        if job['arm'] == 'quantum':
            groups[b][job['name']] = result['best_validation_vrmse8']
    early, late = [groups[b] for b in budgets]
    names = sorted(early)
    if set(names) != set(late) or len(names) < 3:
        raise ValueError('Need identical candidate sets at both fidelities')
    try:
        rho = statistics.correlation(ranks([early[n] for n in names]), ranks([late[n] for n in names]))
    except statistics.StatisticsError:
        rho = None
    winners = [n for n in names if late[n] == min(late.values())]
    cutoff = sorted(early.values())[1]
    early_top = [n for n in names if early[n] <= cutoff]
    # A tie spanning more than two candidates does not establish top-two recovery.
    recovered = len(early_top) <= 2 and all(n in early_top for n in winners)
    result = dict(status='exploratory_only', budgets=budgets, scores=groups,
                  spearman=rho, early_top_two_including_ties=early_top,
                  late_winners=winners, top_two_recovered=recovered,
                  short_fidelity_gate=bool(rho is not None and rho >= .7 and recovered),
                  promotion=False, reason='One-seed calibration; no confirmation or compute-optimality claim')
    (folder / 'fidelity_report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    print(json.dumps(summarize(parser.parse_args().folder), indent=2))
