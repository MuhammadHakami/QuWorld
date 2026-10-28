#!/bin/bash
# mephiles (routing lane, doc 35 §39e): paired test-time sampling decode (s8x4t05: x8 aug, 4 samples/start, T 0.5) of the
# bf16 routed hybrid (route30) and classical policies, seeds 1-5, on the 16 official MD test files.
cd ~/Documents/paper/quworld; export PYTHONUNBUFFERED=1; L=runs/strat/routing/s39e.log; mkdir -p runs/strat/routing
systemd-run --user --scope --quiet -p MemoryMax=22G -p MemorySwapMax=0 timeout 72000 .venv-vrp/bin/python \
  paper/strat/routing/test_decode.py --budget 70000 --jobs route30@bf16_mephiles@1:s8x4t05,vanilla@s23_mephiles@1:s8x4t05,route30@bf16_mephiles@2:s8x4t05,vanilla@s23_mephiles@2:s8x4t05,route30@bf16_mephiles@3:s8x4t05,vanilla@s23_mephiles@3:s8x4t05,route30@bf16_mephiles@4:s8x4t05,vanilla@bf16_mephiles@4:s8x4t05,route30@bf16_mephiles@5:s8x4t05,vanilla@bf16_mephiles@5:s8x4t05 >> $L 2>&1 || echo "CRASH $(date -Is)" >> $L
echo "S39E_DONE $(date -Is)" >> $L
