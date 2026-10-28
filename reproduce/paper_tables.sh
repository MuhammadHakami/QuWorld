#!/usr/bin/env bash
# From result files to the paper: regenerate the generated tables and figures, then build the PDF. Run after the task
# scripts (each reads only the per-seed result files in runs/). Follows runs/paper_refresh.sh.
#   bash reproduce/paper_tables.sh
source "$(dirname "$0")/common.sh"
T=paper/manuscript/iclr2027

py paper/alloc/table1_routed.py --latex                 # Table 1, routed block   -> tab_routed_rows.tex
py paper/alloc/adrc_summary.py && py paper/alloc/gate_table.py   # gate and fine-tuning appendix rows
py paper/alloc/analyze.py table && py paper/alloc/fill_paper.py  # allocation study table
py paper/alloc/figs.py                                  # allocation figure      -> fig_alloc.pdf
py paper/alloc/fig_hybrid_scaling.py                    # qubit-budget scaling   -> fig_hybrid_scaling.pdf
py paper/alloc/fig_gate.py                              # frequency gate         -> fig_gate.pdf
py paper/alloc/fig_audit.py                             # matched-control audit  -> fig_audit.pdf
py paper/alloc/fig_outputs_all.py                       # Figure 2 (from paper/fig_cache/rows_s23/*.npz)
py paper/alloc/fig_arch.py && py paper/alloc/fig_arch_backbones.py   # Figure 1 and the backbone figure
py paper/phase_b/td_replication_summary.py              # TRL2D four-seed rollout table -> td_repl_table.tex
py paper/alloc/fgn_summary.py                           # probabilistic variant  -> fgn_table.tex
py paper/vrp/rf_ladder.py                               # RouteFinder qubit ladder -> fig_rf_ladder.pdf, rf_table.tex
(cd $T && tectonic main_iclr.tex)                       # -> main_iclr.pdf
