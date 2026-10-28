#!/bin/bash
# after .venv-well finishes (shares the uv wheel cache), build .venv-tokamark: tokamark (editable) + tokamark_baseline deps
cd /home/mohhakami/Documents/paper/quworld; export PATH="$HOME/.local/bin:$PATH"
while ! grep -q VENV_WELL_OK runs/well_dl/venv_well_install.log; do sleep 30; done
[ -d .venv-tokamark ] || uv venv -q .venv-tokamark --python 3.13
uv pip install --python .venv-tokamark/bin/python --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "torch==2.13.0" "torchvision" \
 && uv pip install --python .venv-tokamark/bin/python --index-strategy unsafe-best-match -e baseline/tokamark line-profiler torchinfo pyyaml \
 && echo VENV_TOKAMARK_OK
