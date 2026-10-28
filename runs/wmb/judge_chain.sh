#!/bin/bash
# WorldModelBench judging chain (my lane): for each arm, wait for the peer's generation sentinel runs/wmb/GEN_<arm>_DONE,
# then judge the 60-instance subset with the official evaluator (SDPA judge), one arm at a time, memory-gated.
# Results: runs/wmb/results_<arm>.json (+ evaluator stdout in runs/wmb/judge_<arm>.log); sentinel runs/wmb/JUDGE_<arm>_DONE.
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1 VILA_VISION_ATTN=sdpa
L=runs/wmb/judge_chain.log; log() { echo "$(date -Is) $*" | tee -a $L; }
mem_gate() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt 15 ]; do sleep 60; done; }
log "judge chain start (arms: vanilla quantum lora; waits for GEN_<arm>_DONE)"
for arm in vanilla quantum lora; do
  while [ ! -f runs/wmb/GEN_${arm}_DONE ]; do
    [ "$arm" = lora ] && [ -f runs/phase_c/physiq_lora_SKIPPED ] && { log "lora generation skipped upstream; no judging"; break; }
    sleep 300
  done
  [ -f runs/wmb/GEN_${arm}_DONE ] || continue
  [ -f runs/wmb/JUDGE_${arm}_DONE ] && { log "$arm already judged"; continue; }
  dir=data/phase_c/wmb/generated_videos/$arm; n=$(ls $dir/*.mp4 2>/dev/null | wc -l)
  mem_gate; log "judging $arm ($n videos in $dir)"
  .venv-vila/bin/python paper/wmb/judge_run.py --video_dir $dir --model_name lingbot-world-v2-$arm --subset runs/wmb/subset60.json --save_name /home/mohhakami/Documents/paper/quworld/runs/wmb/results_${arm} > runs/wmb/judge_${arm}.log 2>&1 \
    && { touch runs/wmb/JUDGE_${arm}_DONE; log "JUDGED $arm: $(grep -E -i 'overall|total|score' runs/wmb/judge_${arm}.log | tail -3 | tr '\n' ' | ')"; } \
    || log "JUDGE FAILED $arm (see runs/wmb/judge_${arm}.log)"
done
log "judge chain end"
