#!/usr/bin/env bash
# Molecules, Caco-2 permeability (TDC ADMET group caco2_wang; Table 1 bottom row, MAE): the frozen ChemBERTa-77M
# embedding (computed once) through a bottleneck, arms {vanilla, mlp30, mr30ryd}, TDC's official 5-seed protocol and
# evaluator. Follows runs/tdc_runner.sh. Needs .venv-tdc (PyTDC 0.3.6, chemprop 2.3.1).
#   bash reproduce/tdc_caco2.sh [train|all]
source "$(dirname "$0")/common.sh"
TPY=.venv-tdc/bin/python; mkdir -p runs/tdc

if step train; then  # PyTDC downloads the ADMET group into data/tdc/ on first use
  for arm in vanilla mlp30 mr30ryd; do
    done_if runs/tdc/caco2_wang_${arm}.json && continue
    $TPY paper/tdc/run_admet.py caco2_wang $arm > runs/tdc/caco2_wang_${arm}.log 2>&1
    grep TEST runs/tdc/caco2_wang_${arm}.log | tail -1
  done
fi
