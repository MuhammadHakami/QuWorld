"""Cache a released trunk's predictions for one split (disk-backed), e.g. TRAIN for the corrector arms.

    .venv/bin/python paper/realpde/predict_split.py --model unet --split train   ->  data/realpde_cache/pred_<model>_<stage>_<split>.npy
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import common as C
import eval_trunk as E


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="unet"); ap.add_argument("--stage", default="finetune")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    norm = C.normaliser()
    X, Y, _ = C.load_windows(a.split)
    path = C.CACHE / f"pred_{a.model}_{a.stage}_{a.split}.npy"; done = path.with_suffix(".done")
    if done.exists():
        return
    _, g = json.loads((E.OUT / f"select_{a.model}_{a.stage}.json").read_text())       # normaliser chosen on VALID
    net = C.build(a.model, X.shape[1:], Y.shape[1:], stage=a.stage)
    t0 = time.time(); path.unlink(missing_ok=True)
    C.predict(net, X, norm, g, batch=E.BATCH[a.model], out=C.open_pred(path, X.shape)); done.touch()
    print(f"[predict] {a.model}/{a.stage} {a.split}: {len(X)} windows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
