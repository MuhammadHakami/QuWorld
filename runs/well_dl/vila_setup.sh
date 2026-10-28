#!/bin/bash
# WorldModelBench judge: clone NVlabs/VILA + download the judge weights (Efficient-Large-Model/vila-ewm-qwen2-1.5b); build .venv-vila
# after the tokamark env (shared uv cache / bandwidth). Failure here is non-fatal: reported in the log.
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH"
while ! grep -q VENV_TOKAMARK_OK runs/well_dl/venv_well_install.log; do sleep 60; done
[ -d baseline/VILA ] || git clone -q --depth 1 https://github.com/NVlabs/VILA baseline/VILA
uv run --no-sync python - << 'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Efficient-Large-Model/vila-ewm-qwen2-1.5b", local_dir="data/wmb_judge/vila-ewm-qwen2-1.5b"); print("judge weights ->", p)
PY
[ -d .venv-vila ] || uv venv -q .venv-vila --python 3.10 || uv venv -q .venv-vila --python 3.13
uv pip install --python .venv-vila/bin/python --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "torch==2.13.0" torchvision \
 && uv pip install --python .venv-vila/bin/python --index-strategy unsafe-best-match -e baseline/VILA mmengine rich \
 && echo VENV_VILA_OK || echo VENV_VILA_FAILED
