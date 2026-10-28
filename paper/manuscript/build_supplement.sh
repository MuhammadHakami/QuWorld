#!/bin/bash
# Anonymized supplementary material for the ICLR 2027 submission (double-blind).
#   bash paper/manuscript/build_supplement.sh  ->  paper/manuscript/iclr2027/supplementary.zip
# Code (tracked .py under paper/, Lean sources), per-seed result files, the pre-registration notes, a README and
# reproduce_key_results.py. Every text file is scrubbed of the project name, user name and absolute paths, and the
# build fails if any identifying string survives.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=paper/manuscript/iclr2027/supplementary.zip
TMP=$(mktemp -d); S=$TMP/supplement; mkdir -p $S/prereg
copy() { mkdir -p "$S/$(dirname "$1")"; cp "$1" "$S/$1"; }

for f in $(git ls-files 'paper/*.py' 'paper/**/*.py' | grep -v -e '^paper/codex/' -e '^paper/manuscript/'); do copy "$f"; done
for f in formal/*.lean formal/QuotaScaling/*.lean formal/lakefile.toml formal/lean-toolchain formal/lake-manifest.json formal/README.md; do copy "$f"; done
for f in runs/alloc/adrc_results/*.json runs/alloc/adrc_results_s22/*/*.json runs/alloc/results/*/*.json runs/alloc/*.json runs/routefinder/{full,crn,scaling}/*.json \
         runs/routefinder/ladder.json runs/routefinder/rrnco_test_*.log runs/judge/scorecard.json runs/judge/scorecard.md; do
  [ -f "$f" ] && copy "$f"
done
for d in 31-powered-replication-prereg 33-allocation-strategy-study 35-analog-digital-router-configuration-losses \
         36-persistence-audit-band-gated-anchor 37-end-to-end-gated-finetuning 38-trl2d-td-replication; do cp "$d.md" "$S/prereg/"; done
mkdir -p $S/paper/manuscript/iclr2027

cat > $S/README.md <<'EOF'
# Supplementary material (anonymized)

Contents:
- `paper/`: all experiment code.
  - `paper/qlayer.py`: the Rydberg register (exact statevector propagator).
  - `paper/alloc/`: allocation study, routed co-processor ADR-C, hybrid scaling, and noise-conditioned configuration (`fgn.py`).
  - `paper/vrp/`: RouteFinder routed register.
  - `paper/judge/`: the audit statistics behind Table 1.
  - `paper/tests/`: unit tests.
- `formal/`: Lean 4 + Mathlib sources of Proposition 1 and the utilization/band-gate bounds (`lake build`).
- `runs/`: per-seed result files behind the tables and figures. Checkpoints, raw caches and logs are released with the paper.
- `prereg/`: the pre-registration notes, with registration times and amendments, as kept during the project. For review they are self-attested.

Reproduce Table 1's routed block (published SoTA -> classical -> +30/+50 qubits), the 3-D/geometry control (prereg doc 35
§22), and the allocation, hybrid-scaling, noise-conditioned-configuration and routing tables from the shipped result files
(Python 3.11+, numpy, scipy, matplotlib; about a minute on a CPU):

    python reproduce_key_results.py

Table 1 is printed from `runs/judge/scorecard.md`. Recomputing it with `paper/judge/judge.py` needs the per-benchmark
run directories, which are released with the paper.
EOF

cat > $S/reproduce_key_results.py <<'EOF'
"""Regenerate the key tables and figures from the shipped per-seed result files; outputs go to paper/manuscript/iclr2027/."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
STEPS = [("Table 1, routed block (published SoTA -> classical -> +qubits) and the 3-D/geometry control", "paper/alloc/table1_routed.py"),
         ("Allocation study (Table alloc: routed vs pooled, 36% vs 7%)", "paper/alloc/analyze.py latex"),
         ("Hybrid scaling (Fig. hscale, K = 0..200)", "paper/alloc/fig_hybrid_scaling.py"),
         ("Noise-conditioned configuration vs controls (Table fgn)", "paper/alloc/fgn_summary.py"),
         ("RouteFinder qubit ladder (Table rf)", "paper/vrp/rf_ladder.py")]
for title, script in STEPS:
    print(f"\n=== {title}: {script}")
    r = subprocess.run([sys.executable, *script.split()], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-3000:] or r.stderr[-3000:])
print("\n=== Table 1 (audit), from runs/judge/scorecard.md")
print((ROOT / "runs/judge/scorecard.md").read_text()[:6000])
EOF

# scrub identifying strings from every text file
find $S -type f \( -name '*.py' -o -name '*.md' -o -name '*.json' -o -name '*.lean' -o -name '*.toml' -o -name '*.log' \) -print0 |
  xargs -0 sed -i -e 's#/home/mohhakami/Documents/paper/quworld#<repo>#g' -e 's#/home/mohhakami#<home>#g' \
                  -e 's/[Qq][Uu][Ww][Oo][Rr][Ll][Dd]/project/g' -e 's/mohhakami/anon/g'
find $S -name '__pycache__' -prune -exec rm -rf {} +

if grep -r -i -l -e 'quworld' -e 'mohhakami' -e 'hakami' -e 'claude\.ai/code' -e 'Claude-Session' $S; then
  echo "IDENTIFYING STRING SURVIVED; aborting" >&2; exit 1
fi
rm -f "$ROOT/$OUT"; (cd $TMP && zip -qrX "$ROOT/$OUT" supplement)
echo "wrote $OUT ($(du -h "$ROOT/$OUT" | cut -f1)); files: $(find $S -type f | wc -l)"
echo "$TMP"
