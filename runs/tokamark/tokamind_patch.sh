#!/bin/bash
# Idempotent setup of the TokaMind trunk (baseline/tokamind is git-ignored, so the environment steps live here).
# 1) editable install without deps (torch 2.13 >= 2.10, numpy 2.5 >= 2.4 already in .venv-tokamark)
# 2) py3.13 compat: mmt/eval/forward.py annotates with TorchDecoder imported only under TYPE_CHECKING (fine on 3.14,
#    NameError on 3.13) -> add `from __future__ import annotations`
# 3) their finalize.py writes to REPO_ROOT/runs -> symlink into our runs/ tree
cd "$(dirname "$0")/../.." || exit 1
uv pip install --python .venv-tokamark/bin/python --no-deps -e baseline/tokamind >/dev/null 2>&1 || exit 2
python3 - <<'PY'
import re
p = "baseline/tokamind/src/mmt/eval/forward.py"; s = open(p).read()
if "from __future__ import annotations" not in s:
    m = re.match(r'\s*("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n', s)
    ins = "from __future__ import annotations  # py3.13 compat: TorchDecoder is only imported under TYPE_CHECKING\n"
    s = (s[:m.end()] + ins + s[m.end():]) if m else ins + s
    open(p, "w").write(s); print("patched forward.py")
PY
mkdir -p runs/tokamark/tokamind
[ -L baseline/tokamind/runs ] || ln -s "$(pwd)/runs/tokamark/tokamind" baseline/tokamind/runs
.venv-tokamark/bin/python -c "import sys; sys.path.insert(0,'baseline/tokamind/scripts_mast'); import run_finetune, run_eval; print('tokamind ok')"
