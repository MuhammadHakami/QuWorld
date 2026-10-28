"""Doc 35 §29b: the 3-D tethered register (q30-geo=3d) and its matched MLP twin (q30-geo=3d-mlp) coupled to the fusion
equilibrium SoTA pipeline instead of the starter U-Net: trial 22's stage-2 coupling config (frozen trunk trunk1_t5, SRR
detail-band coupling), only the core replaced; seeds 1-3; VALID on the search's 40 fixed shots. Then the frozen trunk
alone and every coupled model are scored ONCE on all DIII-D TEST shots with the vendored official scorer (sigma 0, single
models), and one TEST frame (middle shot, middle frame, fixed in advance) of psi(R, Z) is saved as the Figure 2 row.

    .venv/bin/python paper/nas/sophelio_3d.py   ->  runs/nas/sophelio/sota3d.json, paper/fig_cache/rows_s23/fusion_sota.npz
    .venv/bin/python paper/nas/sophelio_3d.py --k-lo 0 --k-hi 32 --tag sota3dc   (§29e: write band routed to [0, 32),
        where the VALID band oracle puts the consistency error; runs/sophelio_sota/band_oracle*.json)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import optuna
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
import ens_select as ES  # noqa: E402
import sota  # noqa: E402

OUT = ROOT / "runs" / "nas" / "sophelio"
ARMS = (("mlp", "q30-geo=3d-mlp"), ("qubits", "q30-geo=3d"))
SEEDS = (1, 2, 3)


def trial22_cfg() -> dict:
    s = optuna.load_study(study_name="sophelio-coupling1", storage=f"sqlite:///{OUT}/coupling1.db")
    return json.loads(next(t for t in s.trials if t.number == 22).user_attrs["cfg"])


def frame(data, member: dict, shot: int, row: int) -> np.ndarray:
    comps, mean = data.meta["comps"][:ES.KMAX].numpy(), data.meta["mean"].numpy()
    return (member[shot][0][row] @ comps + mean).reshape(65, 65)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--k-lo", type=int, default=None); ap.add_argument("--k-hi", type=int, default=None)
    ap.add_argument("--tag", default="sota3d")
    a = ap.parse_args()
    RES = OUT / f"{a.tag}.json"
    cfg, data = trial22_cfg(), sota.Data()
    if a.k_lo is not None:
        cfg["k_lo"] = a.k_lo
    if a.k_hi is not None:
        cfg["k_hi"] = a.k_hi
    res = json.loads(RES.read_text()) if RES.exists() else {"cfg": cfg}
    for arm, q in ARMS:
        for seed in SEEDS:
            key = f"{arm}-s{seed}"
            if key in res:
                continue
            r = sota.train_eval(cfg | {"qcfg": q, "seed": seed}, data, score_val=40, save=OUT / "trunks" / f"{a.tag}-{arm}_seed{seed}.pt")
            v = r["valid_official"]
            res[key] = dict(qcfg=q, valid_S=v["S"], valid_consistency=v["consistency"], trainable=r["trainable_params"])
            RES.write_text(json.dumps(res, indent=1, default=float))
            print(f"[sota3d] {key} ({q}): VALID S {v['S']:.4f} cons {v['consistency']:.4f} trainable {r['trainable_params']}", flush=True)
    if "test" not in res:                                                    # TEST once, single models, sigma 0
        test = data.shots_of(2)
        paths = {"classical": cfg["init"]} | {f"{arm}-s{s}": str(OUT / "trunks" / f"{a.tag}-{arm}_seed{s}.pt") for arm, _ in ARMS for s in SEEDS}
        members, res["test"] = {}, {}
        for key, p in paths.items():
            members[key] = ES.member(p, data, test)
            r = ES.score(data, test, [members[key]], 0.0)
            res["test"][key] = {k: v for k, v in r.items() if k != "per_shot"}
            print(f"[sota3d] TEST {key}: S {r['S']:.4f} psi {r['r2_psi']:.5f} cons {r['consistency']:.4f} ({len(test)} shots)", flush=True)
        RES.write_text(json.dumps(res, indent=1, default=float))
        shot = test[len(test) // 2]; rows = np.nonzero(data.shot == shot)[0]; i = len(rows) // 2
        truth = data.psi_of(rows[i:i + 1]).reshape(65, 65)
        row = {"label": "fusion eq.", "metric": "S", "kind": "field", "sample": f"TEST shot {data.meta['names'][shot]}, frame {i}",
               "truth": truth, "pred_classical": frame(data, members["classical"], shot, i),
               "pred_mlp": frame(data, members["mlp-s1"], shot, i), "pred_qubits": frame(data, members["qubits-s1"], shot, i),
               "m_classical": res["test"]["classical"]["S"],
               "m_mlp": float(np.mean([res["test"][f"mlp-s{s}"]["S"] for s in SEEDS])),
               "m_qubits": float(np.mean([res["test"][f"qubits-s{s}"]["S"] for s in SEEDS]))}
        dst = ROOT / "paper/fig_cache/rows_s23"; dst.mkdir(parents=True, exist_ok=True)
        fn = "fusion_sota.npz" if a.tag == "sota3d" else f"fusion_{a.tag}.npz"         # §29b keeps its registered name
        np.savez(dst / fn, **row); print("wrote", dst / fn, flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
