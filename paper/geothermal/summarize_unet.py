"""Audit the completed Pelzer grid without repeating any model evaluation.

Run: uv run --no-sync python paper/geothermal/summarize_unet.py
"""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (1337, 1338, 1339)
ARMS = ("vanilla", "mlp30", "mr30ryd")


def summarize():
    records, sources = {}, []
    for arm in ARMS:
        records[arm] = []
        for seed in SEEDS:
            path = ROOT / "runs/pelzer_unet" / f"{arm}_s{seed}.json"
            raw = path.read_bytes()
            result = json.loads(raw)
            assert (result["variant"], result["seed"], result["device"]) == (arm, seed, "cuda")
            assert result["steps"] == 1500 and result["lr"] == 0.0001
            assert result["stage"] == "arm"
            value = result["test"]["mae_c"]
            assert 0 <= value < float("inf")
            records[arm].append(result)
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
    for i in range(len(SEEDS)):
        pair = [records[a][i] for a in ARMS]
        assert len({r["pretrain"] for r in pair}) == 1
        assert len({r["test_at_start"]["mae_c"] for r in pair}) == 1
    values = {a: [r["test"]["mae_c"] for r in records[a]] for a in ARMS}
    return {
        "metric": "test MAE (degrees C)", "seeds": SEEDS,
        "split": "local 700/200/100 split of 1000 prepared runs; split seed 1",
        "arms": {a: {"values": v, "mean": statistics.mean(v), "sample_sd": statistics.stdev(v),
                     "num_params": [r["num_params"] for r in records[a]]} for a, v in values.items()},
        "quantum_seed_wins": {a: sum(q < c for q, c in zip(values["mr30ryd"], values[a]))
                              for a in ("vanilla", "mlp30")},
        "caveat": "Published 0.046 C is context only: exact published split equivalence is unverified. No significance claim from three seed means.",
        "sources": sources,
    }


def markdown(report):
    lines = ["# Pelzer U-Net completed grid", "", report["split"], "",
             "| Arm | Seeds 1337 / 1338 / 1339 | Mean ± sample SD (°C) |",
             "|---|---|---|"]
    for arm, row in report["arms"].items():
        vals = " / ".join(f"{v:.6f}" for v in row["values"])
        lines.append(f"| {arm} | {vals} | {row['mean']:.6f} ± {row['sample_sd']:.6f} |")
    lines += ["", f"Quantum seed wins: {report['quantum_seed_wins']}.", "", report["caveat"], "",
              "Source hashes and parameter counts: `summary_sep20.json`. No test evaluation was rerun."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report = summarize()
    out = ROOT / "runs/pelzer_unet/summary_sep20"
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    out.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
