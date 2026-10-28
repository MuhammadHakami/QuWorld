"""Sophelio: the single TEST scoring of the VALID-selected ensembles (declared 2026-09-23 17:40, before any TEST score).

    .venv/bin/python paper/sophelio/score_test.py --tags "" _reg _mlp   ->  runs/sophelio_sota/test_scores.json

Rule: each ensemble chosen by ens_select.py (classical pool; + register-coupled; + matched-MLP-coupled) is scored ONCE on
ALL DIII-D TEST shots with the vendored official scorer, at the smoothing sigma its full-VALID score preferred. The
leaderboard entry is the one with the best full-VALID S; register vs MLP is read from the matched pair. Nothing is
re-selected after this runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ens_select as ES  # noqa: E402
import sota  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", default=["", "_reg", "_mlp"])
    a = ap.parse_args()
    out_f = sota.ROOT / "runs" / "sophelio_sota" / "test_scores.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    data = sota.Data(); test = data.shots_of(2)
    trunks = sota.ROOT / "runs" / "nas" / "sophelio" / "trunks"
    for tag in a.tags:
        key = tag or "classical"
        if key in out:
            print(f"[test] {key}: already scored (S {out[key]['test']['S']:.4f}); not re-scored", flush=True)
            continue
        sel = json.loads((sota.ROOT / "runs" / "sophelio_sota" / f"ens_select{tag}.json").read_text())
        vf = sel["valid_full"]; sigma = max(vf, key=lambda sg: vf[sg]["S"])
        members = [ES.member(str(trunks / m), data, test) for m in sel["members"]]
        r = ES.score(data, test, members, float(sigma))
        out[key] = dict(members=sel["members"], sigma=float(sigma), valid_full_S=vf[sigma]["S"], n_test_shots=len(test),
                        test={k: v for k, v in r.items() if k != "per_shot"})
        out_f.write_text(json.dumps(out, indent=1, default=float))
        print(f"[test] {key}: {len(sel['members'])} members sigma={sigma} | FULL VALID S {vf[sigma]['S']:.4f} | TEST S {r['S']:.4f} "
              f"psi {r['r2_psi']:.5f} qb {r['r2_qb']:.4f} dlcfs {r['dlcfs']:.4f} cons {r['consistency']:.4f} ({len(test)} shots)", flush=True)


if __name__ == "__main__":
    main()
