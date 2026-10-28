#!/bin/bash
# Boot-time self-healing: installed as a crontab @reboot entry. If the campaign
export PATH="$HOME/.local/bin:$PATH"
# is unfinished, relaunch the detached runner; every stage is step-targeted and
# resumable, so a board latch-off costs only the minutes since the last
# fsync'd checkpoint.

REPO=/home/mohhakami/Documents/paper/quworld
cd "$REPO" || exit 0
sleep 60   # let the system settle after boot

STATUS=runs/phase_b/pilot_status
if ! grep -q "PILOT_DONE" "$STATUS" 2>/dev/null; then
  setsid nohup bash paper/phase_b/run_pilot_detached.sh 3000 \
    >> runs/phase_b/pilot_runner.log 2>&1 < /dev/null &
  exit 0
fi

# Once the pilot is done, the overnight campaign takes over (created when the
# duplication audit returns GO; same detached/resumable pattern).
if [ -x paper/phase_b/run_overnight_detached.sh ] && \
   ! grep -q "OVERNIGHT_DONE" runs/phase_b/overnight_status 2>/dev/null; then
  setsid nohup bash paper/phase_b/run_overnight_detached.sh \
    >> runs/phase_b/overnight_runner.log 2>&1 < /dev/null &
fi

# Gutzwiller control chain (added 2026-09-06): while unfinished it owns the GPU
# and relaunches the MHD campaign itself on GUTZ_DONE (run_gutz.sh is flock-guarded).
if [ -f runs/phase_b2/gutz_status ] && \
   ! grep -q "GUTZ_DONE" runs/phase_b2/gutz_status 2>/dev/null; then
  setsid nohup bash "$REPO/paper/phase_b/run_gutz.sh" \
    >> "$REPO/runs/phase_b2/gutz_runner.log" 2>&1 < /dev/null &
  exit 0
fi

# MHD campaign (Gate B) — added 2026-08-18 after the memmap-OOM reboot gap
# (sentinel fixed 2026-09-06: the campaign writes MHD_CAMPAIGN_DONE)
if [ -f runs/phase_b_mhd/mhd_status ] && \
   ! grep -q "MHD_CAMPAIGN_DONE" runs/phase_b_mhd/mhd_status 2>/dev/null; then
  setsid nohup bash "$REPO/paper/phase_b/run_mhd_campaign.sh" \
    >> "$REPO/runs/phase_b_mhd/mhd_runner.log" 2>&1 < /dev/null &
fi
