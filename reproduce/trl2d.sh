#!/usr/bin/env bash
# Turbulent radiative layer (The Well, TRL2D): Table 1 routed row, the released-checkpoint anchor and the long-horizon
# rollout row. Follows paper/phase_b/run_overnight_detached.sh (neck campaign), runs/alloc_bgp_trl2d_runner.sh (routed
# cache), runs/td_replication_runner.sh + paper/phase_b/run_round2_td.sh (rollout, four seeds).
#   bash reproduce/trl2d.sh [data|anchor|train|routed|rollout|all]
source "$(dirname "$0")/common.sh"
DS=turbulent_radiative_layer_2D

if step data; then
  hf polymathic-ai/$DS --repo-type dataset --local-dir data/trl2d                         # 7.9 GB
  mkdir -p data/well && ln -sfn ../trl2d data/well/$DS                                      # both loaders find it
  hf polymathic-ai/UNetConvNext-$DS --local-dir data/cnext_ckpt                           # released CNextU-net
fi

if step anchor; then                                                                       # released checkpoint, 873 windows
  done_if runs/eval_official_cnext.csv || py paper/phase_b/eval_official.py data/cnext_ckpt runs/eval_official_cnext.csv
fi

if step train; then   # neck insertion: 2,400 s budget, lr 1e-4, 20,000 target steps, best-on-VALID checkpoint
  for seed in 1337 1338 1339; do for v in vanilla mr30ryd mlp30; do
    done_if runs/phase_b/${v}_s${seed}.pt || py paper/phase_b/finetune.py $v $seed 2400 1e-4 20000
    done_if runs/phase_b/test_${v}_s${seed}.csv || \
      py paper/phase_b/eval_official.py runs/phase_b/${v}_s${seed}.pt runs/phase_b/test_${v}_s${seed}.csv
    py paper/phase_b/paired_stats.py runs/phase_b/test_${v}_s${seed}.csv runs/eval_official_cnext.csv
  done; done
fi

if step routed; then  # routed co-processor on the fine-tuned vanilla trunk, every TEST window
  routed_cache $DS:gate_vanilla_s1337 runs/phase_b/vanilla_s1337.pt ADRC_TEST_ALL=1
  routed_arms ${DS}__gate_vanilla_s1337
fi

if step rollout; then # pushforward (TD, k = 4) fine-tune, 10,000 steps; four seeds; sliding-start rollout TEST
  for seed in 1337 1338 1339 1340; do for v in vanilla mr30ryd mlp30; do
    n=${v}_s${seed}; ck=runs/phase_b2/$n.pt
    done_if $ck || py paper/phase_b/finetune_td.py $v $seed 2400 1e-4 10000 4
    done_if runs/phase_b2/test_$n.csv || py paper/phase_b/eval_official.py $ck runs/phase_b2/test_$n.csv
    done_if runs/phase_b2/pertraj_$n.npz || py paper/phase_b/eval_rollout_pertraj.py $ck $n > runs/phase_b2/rollout_$n.log
    done_if runs/phase_b2/multistart_$n.npz || \
      py paper/phase_b/eval_rollout_multistart.py $ck $n > runs/phase_b2/multistart_$n.log   # the summary parses this log
  done; done
  py paper/phase_b/td_replication_summary.py                    # -> runs/phase_b2/td_replication.json, td_repl_table.tex
fi
