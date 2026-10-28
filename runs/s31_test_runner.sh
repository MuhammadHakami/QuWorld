#!/bin/bash
# doc 35 §31e: the registered §31 TEST phase, in two locked phases.
#   A (no TEST on disk): register (ryd) and matched MLP arms, pooled and proxy routing, K=30, seeds 1-3, TRAIN fit and
#     V1 early stop (--vsplit), models saved; then everything is hashed into runs/s31/lock/arms_lock.sha256 and the
#     runner WAITS for runs/s31/lock/ARMS_COMMITTED (created by hand after the lock is committed, tagged and audited).
#   B: TEST download at the pinned revision, TEST-only cache from the locked TRAIN tile order (train.pt / valid.pt must
#     hash identically before and after), each frozen model scored ONCE on the full TEST split, then the verdict.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; L=runs/s31/s31_test.log; K=runs/s31/lock
mem() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt ${1:-30} ]; do sleep 60; done; }
ROWS="active_matter gray_scott_reaction_diffusion"
locked_data() {  # the prediction lock's data entries (code entries were superseded by the §31e code, hashed in the arms lock)
  tail -n +2 $K/prediction_lock.sha256 | grep -v "\.py$\|\.sh$" | sha256sum -c --quiet || { echo "CRASH s31e prediction-lock data changed $(date -Is)" >> $S; exit 1; }
}
arms_ok() {      # the frozen models and code must hash as locked at every TEST step (Codex audit 035507)
  tail -n +2 $K/arms_lock.sha256 | sha256sum -c --quiet || { echo "CRASH s31e arms lock changed ($1) $(date -Is)" >> $S; exit 1; }
}
while pgrep -f "capped.sh s31a" >/dev/null; do sleep 30; done    # a relaunch never overlaps a phase-A job still running
grep -q S31_VALID_DONE $S || { echo "CRASH s31e: no S31_VALID_DONE $(date -Is)" >> $S; exit 1; }
for row in $ROWS; do [ -e data/well/$row/data/test ] && { echo "CRASH s31e: TEST present before the arms lock ($row) $(date -Is)" >> $S; exit 1; }; done
locked_data
# ---------------------------------------------------------------- phase A: train and freeze the arms (TRAIN + V1 only)
fail=0
for row in $ROWS; do for core in ryd mlp; do for st in pooled proxy; do for seed in 1 2 3; do
  n=${st}_${core}_K30_s${seed}_s31e                                   # distinct tag: never touches the 18 locked lin results
  [ -f runs/alloc/results/$row/vsplit/ckpt/$n.pt ] && [ -f runs/alloc/results/$row/vsplit/$n.json ] && continue
  ok=0; for try in 1 2 3; do
    mem 30; QW_GPU_GB=8 timeout 7200 runs/capped.sh s31a 16 .venv/bin/python paper/alloc/coproc.py $row --strategy $st --core $core \
      --K 30 --seed $seed --vsplit --save_ckpt --tag _s31e >> $L 2>&1 && [ -f runs/alloc/results/$row/vsplit/ckpt/$n.pt ] && { ok=1; break; }
    echo "RETRY s31e arm $row $n attempt $try failed $(date -Is)" >> $S; sleep 120
  done
  [ $ok = 1 ] || { echo "CRASH s31e arm $row $n (3 attempts) $(date -Is)" >> $S; fail=1; }
done; done; done; done
[ $fail = 0 ] || { echo "S31E_INCOMPLETE (an arm failed; TEST not downloaded) $(date -Is)" >> $S; exit 1; }
{ date -Is; sha256sum runs/alloc/results/{active_matter,gray_scott_reaction_diffusion}/vsplit/ckpt/*_{ryd,mlp}_K30_s*_s31e.pt \
    runs/alloc/results/{active_matter,gray_scott_reaction_diffusion}/vsplit/*_{ryd,mlp}_K30_s*_s31e.json \
    paper/alloc/coproc.py paper/alloc/cache.py paper/alloc/cores.py paper/phase_b/well2d.py paper/phase_b/well2d_backbone.py \
    runs/s31_download.py runs/s31_test_runner.sh runs/s31_verdict.py; } > $K/arms_lock.sha256
[ "$(grep -c '\.pt$' $K/arms_lock.sha256)" = 24 ] || { echo "CRASH s31e: arms lock does not hold 24 models $(date -Is)" >> $S; exit 1; }
echo "S31E_ARMS_LOCKED (waiting for runs/s31/lock/ARMS_COMMITTED) $(date -Is)" >> $S
until [ -f $K/ARMS_COMMITTED ]; do sleep 60; done
arms_ok "before TEST download"
locked_data
# ---------------------------------------------------------------- phase B: TEST acquired, cached, scored once
.venv/bin/python runs/s31_download.py --test >> $L 2>&1 || { echo "CRASH s31e TEST download $(date -Is)" >> $S; exit 1; }
for row in $ROWS; do
  [ -f runs/alloc/cache/$row/test.pt ] || { mem 40; ALLOC_TEST_ONLY=1 QW_GPU_GB=16 timeout 21600 runs/capped.sh s31t 48 \
    .venv/bin/python paper/alloc/cache.py well $row --test 100000000 >> $L 2>&1 || { echo "CRASH s31e TEST cache $row $(date -Is)" >> $S; exit 1; }; }
done
locked_data; arms_ok "after TEST cache"
for row in $ROWS; do for core in ryd mlp; do for st in pooled proxy; do for seed in 1 2 3; do
  n=${st}_${core}_K30_s${seed}_s31e
  [ -f runs/alloc/results/$row/test/$n.json ] && continue
  arms_ok "before scoring $row $n"
  mem 30; QW_GPU_GB=16 timeout 7200 runs/capped.sh s31s 32 .venv/bin/python paper/alloc/coproc.py $row --strategy $st --core $core \
    --K 30 --seed $seed --score_test --tag _s31e >> $L 2>&1 || { echo "CRASH s31e score $row $n $(date -Is)" >> $S; exit 1; }
done; done; done; done
arms_ok "after scoring"; locked_data
.venv/bin/python runs/s31_verdict.py >> $L 2>&1 || { echo "CRASH s31e verdict $(date -Is)" >> $S; exit 1; }
echo "S31_TEST_DONE $(date -Is)" >> $S
