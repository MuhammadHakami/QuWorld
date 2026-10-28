#!/bin/bash
# Keeps the paper's evidence current (user, Sep 26: "figure 2 and its table updated regularly with new evidence"):
# every 10 min, if a result or figure-row file is newer than the last build, regenerate Table 1's routed rows
# (table1_routed.py --latex), the all-benchmark output figure (fig_outputs_all.py) and the scaling figure, rebuild the
# PDF with tectonic and check the layout (conclusion by page 9, AI-use statement on page 10). Commits stay manual.
cd /home/mohhakami/Documents/paper/quworld; STAMP=runs/paper_refresh.stamp; LOG=runs/paper_refresh.log; S=runs/alloc/status
PY=.venv/bin/python; TEX=paper/manuscript/iclr2027
[ -f $STAMP ] || touch -d '2000-01-01' $STAMP
while true; do
  new=$(find runs/alloc/adrc_results paper/fig_cache/rows_s23 paper/fig_cache runs/torax_s23 runs/arena_s23 -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.npz' \) -newer $STAMP 2>/dev/null | head -3)
  if [ -n "$new" ]; then
    touch $STAMP.next
    exec 9>runs/paper_build.lock; flock 9                       # manual builds take the same lock
    { echo "== $(date -Is) new: $(echo $new | tr '\n' ' ')"
      $PY paper/alloc/table1_routed.py --latex | tail -6
      $PY paper/alloc/fig_outputs_all.py | tail -1
      $PY paper/alloc/fig_hybrid_scaling.py | tail -1
      (cd $TEX && tectonic main_iclr.tex 2>&1 | grep -iE "error|warning: .*undefined" | head -3)
      ai=$(for p in 9 10 11; do pdftotext -f $p -l $p $TEX/main_iclr.pdf - 2>/dev/null | grep -q "AI USE STATEMENT" && echo $p; done | head -1)
      echo "AI-use statement on page ${ai:-?}"
      [ "$ai" = 10 ] || echo "PAPER_LAYOUT_CHECK_FAILED ai_page=${ai:-none} $(date -Is)" >> $S
    } >> $LOG 2>&1
    flock -u 9; exec 9>&-
    mv $STAMP.next $STAMP
  fi
  sleep 600
done
