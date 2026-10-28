#!/bin/bash
# WorldModelBench judge env, attempt 3: python 3.10, no decord (no aarch64 wheel), VILA 2.0 installed WITHOUT its pins (torch==2.3.0 has no cu130/aarch64 wheel),
# explicit inference deps on torch 2.13 cu130. Judge = Efficient-Large-Model/vila-ewm-qwen2-1.5b via llava.load().
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv-vila; uv venv -q .venv-vila --python 3.10 || { echo VENV_VILA_PY310_UNAVAILABLE; exit 1; }
PYV=.venv-vila/bin/python
uv pip install --python $PYV --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "torch==2.13.0" torchvision torchaudio \
 && uv pip install --python $PYV --index-strategy unsafe-best-match --no-deps -e baseline/VILA \
 && uv pip install --python $PYV --index-strategy unsafe-best-match "transformers==4.46.0" "tokenizers>=0.15.2" "sentencepiece==0.1.99" shortuuid accelerate peft pydantic markdown2 numpy scikit-learn requests "einops==0.6.1" "einops-exts==0.0.4" "timm==0.9.12" pillow opencv-python-headless mmengine rich tqdm pyyaml hydra-core loguru fire \
 && echo VENV_VILA_OK || echo VENV_VILA_FAILED_PY310
