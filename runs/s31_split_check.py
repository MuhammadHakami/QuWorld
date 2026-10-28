"""Doc 35 §31c: V1/V2 composition of each prospective row's VALID split from metadata only (HDF5 headers: files and
trajectory counts; no field values). V1 = even, V2 = odd global trajectory index. Exits non-zero unless gray_scott has
10/10 per file and active_matter 12/12 overall.

    .venv/bin/python runs/s31_split_check.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper" / "phase_b"))
import well2d  # noqa: E402

out, ok = {}, True
for row in ("active_matter", "gray_scott_reaction_diffusion"):
    s = well2d.Split(row, "valid")
    per = {}
    for g, (fi, _) in enumerate(s.index):
        f = pathlib.Path(s.files[fi]).name; per.setdefault(f, [0, 0])[g % 2] += 1
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    out[row] = {"trajectories": s.n, "V1_V2_total": tot, "per_file_V1_V2": per}
    if row.startswith("gray_scott"):
        ok &= all(v == [10, 10] for v in per.values()) and len(per) == 6
    else:
        ok &= tot == [12, 12]
    s.close()
pathlib.Path("runs/s31/split_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)); print("[s31] split check", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
