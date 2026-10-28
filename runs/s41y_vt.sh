#!/bin/bash
# Fusion §41y-b (registered 2026-10-11 before any result): post-processing arms on ALL VALID + TEST shots (743; GPU twin, paired per shot),
# where the extreme-li frames are (sample 0.16% of frames with head li > 2.5 vs VT 0.30%). Re-applying the li warp to F2's public psi halved
# the li miss on the 352 worst frames (rms 0.405 -> 0.197), the cap is not binding. Arms (HJl9 components): VB = current chain;
# VL2 = + a second li pass ("li2"); VX = xp n_it 6, dk 0.006. Rule: adopt if the paired S delta > 0 with CI > 0 and D_LCFS not worse.
# Relaunch 2026-10-11T15:09: the first launch crashed after VB's 55-min inference (parts_all asserted VALID-only rows; fx.scored_rows now
# admits VALID + TEST). The three arms now share one trunk / head inference (--kws) and store psi in float32 (submission precision).
# Relaunch 3 (2026-10-11T16:30): memguard killed relaunch 2 at 16:20 (4 GPU jobs); one VT job at a time (runs/strat/vt.lock).
# Relaunch 4 (2026-10-11T16:45): relaunch 3 hit its own 20 GB torch cap (post_all chunk); cap 30 GB.
# Relaunch 2 (2026-10-11T15:51): the 15:45 OOM (all arms' psi held for 193k frames) -> --split vt now streams parts per row group.
cd /home/mohhakami/Documents/paper/quworld; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1; L=runs/strat/fusion/s41y_vt3.log
J=$(for s in 1 2 3 4 5 6; do printf "head_boundary_v8_J1_s$s+"; done | sed 's/+$//'); Q=$(for s in 1 2 3 4 5 6; do printf "head_qb_QH_s$s+"; done | sed 's/+$//')
KWS='{"VB":{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}},
"VL2":{"_steps":["shift","bw","kw","xp","jgn","li","li2"],"xp":{"sigma":0.15,"cap":0.10},"li":{"cap":0.9}},
"VX":{"_steps":["shift","bw","kw","xp","jgn","li"],"xp":{"sigma":0.15,"cap":0.10,"n_it":6,"dk":0.006},"li":{"cap":0.9}}}'
flock runs/strat/vt.lock env QW_GPU_GB=30 runs/capped.sh s41yvt 40 .venv/bin/python paper/strat/fusion/gpu_build.py --split vt --tag VT3 --psi psiF --qb qh50 --qbheads "$Q" \
  --heads "$J" --kws "$KWS" --axis head >> $L 2>&1 || echo "CRASH VT3" >> $L
cd paper/strat/fusion && ../../../.venv/bin/python vt_pp_compare.py VB VL2 VX >> ../../../$L 2>&1
grep -q "^\[vtpp\] VL2 vs VB" ../../../$L && echo "S41Y_VT3_DONE $(date -Is)" >> ../../../$L || echo "S41Y_VT3_FAIL $(date -Is)" >> ../../../$L
