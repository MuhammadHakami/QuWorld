#!/bin/bash
# doc 35 §32: full-TEST rescoring of Table 1's rayleigh_benard routed rows (recipe re-run unchanged; TEST once per model).
cd /home/mohhakami/Documents/paper/quworld; export PYTHONUNBUFFERED=1; S=runs/alloc/status; O=runs/alloc/adrc_results_s32; mkdir -p $O
C=runs/alloc/adrc_cache; OLD=$C/rayleigh_benard__gatest_vanilla_s1337; NEW=$C/rayleigh_benard__gatestfull_vanilla_s1337
mem() { while [ "$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)" -lt ${1:-30} ]; do sleep 30; done; }
if [ ! -f $NEW/test.pt ]; then mem 40
  env ADRC_BANDS=5 ADRC_BLEND=5 ADRC_CKPT=runs/well2d/rayleigh_benard/cnext/vanilla_s1337.pt ADRC_VARIANT=vanilla ADRC_VALID_STRAT=2 \
    ADRC_TEST_ALL=1 QW_GPU_GB=24 timeout 14400 runs/capped.sh s32c 40 .venv/bin/python paper/alloc/adrc_cache.py \
    rayleigh_benard:gatestfull_vanilla_s1337 >> runs/alloc/s32.log 2>&1 || { echo "CRASH s32 cache $(date -Is)" >> $S; exit 1; }
fi
# reproducibility check before any TEST use (registered): TRAIN/VALID/basis must match the original cache
.venv/bin/python - >> runs/alloc/s32.log 2>&1 <<PY || { echo "CRASH s32 repro $(date -Is)" >> $S; exit 1; }
import torch
def flat(x):
    if torch.is_tensor(x): return [x.float()]
    if isinstance(x, dict): return [t for k in sorted(x) for t in flat(x[k])]
    if isinstance(x, (list, tuple)): return [t for v in x for t in flat(v)]
    return []
worst = 0.0
for f in ("train.pt", "valid.pt", "basis_meta.pt"):
    a, b = flat(torch.load("$OLD/" + f, weights_only=False)), flat(torch.load("$NEW/" + f, weights_only=False))
    assert len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b)), f"{f}: structure differs"
    d = max(float((x - y).abs().max()) if x.numel() else 0.0 for x, y in zip(a, b)); worst = max(worst, d)
    print(f"[s32] repro {f}: max abs diff {d:.3g}", flush=True)
assert worst < 1e-3, f"rebuilt cache differs (max {worst:.3g}); stop before TEST"
PY
dir=rayleigh_benard__gatestfull_vanilla_s1337
for seed in 1 2 3; do
  for spec in "cls_only_gate cls_only --gate" "adrc_mlp_gate_rs_dyn adrc_mlp --gate --route_sup --dyn_tether" \
              "adrc_ryd_gate_rs_dyn adrc_ryd --gate --route_sup --dyn_tether" \
              "adrc_ryd_gate_rs_k50_dyn adrc_ryd --gate --route_sup --atoms 50 --dyn_tether" \
              "adrc_ryd_gate_rs_k100_dyn adrc_ryd --gate --route_sup --atoms 100 --dyn_tether"; do
    set -- $spec; out=$1; arm=$2; shift 2
    [ -f $O/${dir}_${out}_s${seed}.json ] && continue                      # TEST once: never re-score a finished model
    mem 30; while [ "$(jobs -rp | wc -l)" -ge 1 ]; do sleep 15; done
    QW_OUT=$O QW_GPU_GB=14 timeout 10800 runs/capped.sh s32 24 .venv/bin/python paper/alloc/adrc.py $dir --arm $arm --seed $seed "$@" \
      >> runs/alloc/s32.log 2>&1 || echo "CRASH s32 $arm $* s$seed $(date -Is)" >> $S &
    sleep 20
  done
done; wait; echo "S32_DONE $(date -Is)" >> $S
