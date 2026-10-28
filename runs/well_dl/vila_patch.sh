#!/bin/bash
# VILA 2.0 hard-codes attn_implementation="flash_attention_2" for the SigLIP vision tower (llava/model/multimodal_encoder/siglip_encoder.py);
# flash-attn has no aarch64/cu130 wheel here (a stub package raises). Make it env-configurable: VILA_VISION_ATTN=sdpa.
F=/home/mohhakami/Documents/paper/quworld/baseline/VILA/llava/model/multimodal_encoder/siglip_encoder.py
grep -q "VILA_VISION_ATTN" "$F" || { sed -i 's/attn_implementation="flash_attention_2"/attn_implementation=os.environ.get("VILA_VISION_ATTN", "flash_attention_2")/g' "$F"; grep -q "^import os" "$F" || sed -i '1i import os' "$F"; }
grep -c "VILA_VISION_ATTN" "$F"
