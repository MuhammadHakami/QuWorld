# Shared helpers for the per-task reproduction scripts in reproduce/ (sourced, not run).
# Each task script follows the runner that produced the paper's numbers (named in its header), without the
# workstation-specific memory guards. Steps skip finished outputs, so a model is never scored on TEST twice.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEP=${1:-all}

py() { uv run --no-sync python "$@"; }                 # main environment (uv sync)
hf() { uv run --no-sync hf download "$@"; }            # Hugging Face downloads
step() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }   # bash reproduce/<task>.sh [data|train|eval|<extra>|all]
done_if() { [ -e "$1" ]; }                            # an existing output means the step already ran
say() { echo "[$(date '+%H:%M:%S')] $*"; }

# Routed co-processor on a Well row (Table 1, top block): cache the frozen classical part's residual as band-wise
# POD modes with the frequency gate, then train every arm for seeds 1-3 (one TEST score per model).
#   routed_cache <dataset:tag> <checkpoint> [extra KEY=VALUE env for adrc_cache.py ...]
#   routed_arms  <dataset__tag> [result dir]
routed_cache() {
  local tag=$1 ckpt=$2; shift 2; local dir=${tag/:/__}
  done_if runs/alloc/adrc_cache/$dir/test.pt && return 0
  say "cache $tag <- $ckpt"
  env ADRC_BANDS=5 ADRC_BLEND=5 ADRC_CKPT=$ckpt ADRC_VARIANT=vanilla "$@" py paper/alloc/adrc_cache.py $tag
}
routed_arms() {
  local dir=$1 out=${2:-runs/alloc/adrc_results}
  for seed in 1 2 3; do
    for spec in "cls_only_gate cls_only --gate" \
                "adrc_mlp_gate_rs_dyn adrc_mlp --gate --route_sup --dyn_tether" \
                "adrc_ryd_gate_rs_dyn adrc_ryd --gate --route_sup --dyn_tether" \
                "adrc_ryd_gate_rs_k50_dyn adrc_ryd --gate --route_sup --atoms 50 --dyn_tether" \
                "adrc_ryd_gate_rs_k100_dyn adrc_ryd --gate --route_sup --atoms 100 --dyn_tether"; do
      set -- $spec; local name=$1 arm=$2; shift 2
      done_if $out/${dir}_${name}_s${seed}.json && continue
      say "routed $dir $name seed $seed"
      QW_OUT=$out py paper/alloc/adrc.py $dir --arm $arm --seed $seed "$@"
    done
  done
}
