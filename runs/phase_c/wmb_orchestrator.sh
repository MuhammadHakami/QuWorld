#!/bin/bash
# WorldModelBench generation (Phase-C GPU queue): waits until the Physics-IQ orchestrator has finished its arm chain
# and no python process holds the GPU, then per arm (vanilla, quantum, lora if an adapter exists): T5 contexts on the
# GPU -> 60 i2v videos (paper/wmb/generate.py) -> sentinel runs/wmb/GEN_<arm>_DONE (judging = peer quworld-34).
set -u; REPO=/home/mohhakami/Documents/paper/quworld; cd "$REPO" || exit 1; export PATH="$HOME/.local/bin:$PATH"; export PYTHONUNBUFFERED=1
STATUS=runs/phase_c/phase_c_status; RUN="uv run --no-sync python paper/wmb/generate.py"; GPU_SCOPE="systemd-run --user --scope -q -p MemoryMax=80G"; POLL=120
status() { echo "$(date -Is) wmb: $*" | tee -a "$STATUS"; }
gpu_python_pids() { for p in $(fuser /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null | tr -s ' ' '\n' | sort -u); do case "$(readlink /proc/"$p"/exe 2>/dev/null)" in *python*) echo "$p" ;; esac; done; }
status "waiting for the Physics-IQ arm chain to finish + free GPU (poll ${POLL}s)"
while :; do
  if grep -q "physiq: arm chain finished" "$STATUS" 2>/dev/null; then [ -z "$(gpu_python_pids)" ] && break; fi
  sleep $POLL
done
status "GPU free after the Physics-IQ arms; starting WorldModelBench generation (60-instance subset)"
for arm in vanilla quantum lora; do
  [ -f runs/wmb/GEN_${arm}_DONE ] && { status "$arm already DONE"; continue; }
  if [ "$arm" != vanilla ] && ! $RUN --stage plan --arm "$arm" >> runs/wmb/gen_${arm}.log 2>&1; then status "$arm: plan failed (no adapter?) -> skipped"; continue; fi
  status "$arm: T5 contexts (GPU)"; env PHYSIQ_T5_DEVICE=cuda $GPU_SCOPE $RUN --stage t5cache --arm "$arm" >> runs/wmb/gen_${arm}.log 2>&1 || { status "$arm: t5cache failed"; continue; }
  for attempt in 1 2 3; do
    status "$arm: generation attempt $attempt"; $GPU_SCOPE $RUN --stage generate --arm "$arm" >> runs/wmb/gen_${arm}.log 2>&1
    [ -f runs/wmb/GEN_${arm}_DONE ] && break; sleep $POLL
  done
  [ -f runs/wmb/GEN_${arm}_DONE ] && status "$arm DONE -> data/phase_c/wmb/generated_videos/$arm (judge: quworld-34)" || status "$arm: generation incomplete after 3 attempts"
done
status "WorldModelBench generation chain finished"
