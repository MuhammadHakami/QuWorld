"""Doc 35 §37, MHD_64: read-only snapshots of the live §34 Flowers run for a weight-averaging (SWA-style) probe.

Polls runs/phase_b_mhd/flower_s1337_latest.pt (never written here). When its mtime changes and the file is older than
120 s, it is copied to a scratch dir, loaded (a failed load discards the copy), and only the model weights and step are
kept (scratch/snaps/flower_step<step>.pt). CPU only, one file at a time.

    uv run --no-sync python paper/strat/mhd/snap_latest.py <scratch_dir> <hours>
"""
import os
import pathlib
import shutil
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "runs/phase_b_mhd/flower_s1337_latest.pt"
DST = pathlib.Path(sys.argv[1]); HOURS = float(sys.argv[2])
torch.set_num_threads(2)
seen, t_end = None, time.time() + HOURS * 3600
while time.time() < t_end:
    try:
        mt = SRC.stat().st_mtime
        if mt != seen and time.time() - mt > 120:
            tmp = DST / "latest_copy.tmp"; shutil.copy2(SRC, tmp)
            if os.stat(tmp).st_mtime == SRC.stat().st_mtime:          # unchanged during the copy
                st = torch.load(tmp, weights_only=True, map_location="cpu")
                out = DST / f"flower_step{st['step']}.pt"
                torch.save({"model_state_dict": st["model_state_dict"], "step": st["step"], "best": st["best"]}, out)
                print(f"{time.strftime('%H:%M:%S')} snap step {st['step']} -> {out.name}", flush=True)
                seen = mt
            tmp.unlink(missing_ok=True)
    except Exception as e:                                             # partial file or race: try again next poll
        print(f"{time.strftime('%H:%M:%S')} skip: {e!r}", flush=True)
    time.sleep(60)
