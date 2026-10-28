"""Regenerate notebooks/autoresearch.ipynb (the Living Report) from runs/*.log.

Usage: uv run paper/log_notebook.py
Parses every runs/round*_<variant>.log summary block, groups by round, and
rebuilds the notebook: mission, architecture topologies, comparison tables,
tree-divergence map. Idempotent — safe to run after every round.
"""

import glob
import pathlib
import re
import subprocess

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "autoresearch.ipynb"

TOPOLOGY = {
    "none": "FNO2d trunk only (lift -> 4 spectral blocks -> proj)",
    "mlp": "FNO2d + pooled-latent MLP bottleneck residual (matched params)",
    "hea12": "FNO2d + 12q HEA quantum bottleneck (input-conditioned angles)",
    "hea16": "FNO2d + 16q HEA quantum bottleneck (input-conditioned angles)",
    "ryd12": "FNO2d + 12q Rydberg-analog block (data-driven detunings, trainable geometry)",
    "ryd16": "FNO2d + 16q Rydberg-analog block (data-driven detunings, trainable geometry)",
    "ryd20": "FNO2d + 20q Rydberg-analog block (data-driven detunings, trainable geometry)",
    "mlp30": "FNO2d + MLP control matched to mr30 (30-output) parameter budget",
    "mr30hea": "FNO2d + 30 simulated qubits: 3x10q HEA multi-register (blockade-decoupled patches)",
    "mr30ryd": "FNO2d + 30 simulated qubits: 3x10q Rydberg-analog multi-register",
}


def parse_log(path: str) -> dict:
    txt = open(path).read()
    out = {}
    for key in ["variant", "val_vrmse_1", "val_vrmse_8", "train_loss",
                "num_steps", "num_params", "train_seconds", "peak_vram_mb"]:
        m = re.search(rf"^{key}:\s+(\S+)", txt, re.M)
        out[key] = m.group(1) if m else None
    out["crashed"] = out["val_vrmse_1"] is None
    return out


def main() -> None:
    runs = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "round*_*.log"))):
        name = pathlib.Path(f).stem
        rnd, variant = name.split("_", 1)
        runs.setdefault(rnd, {})[variant] = parse_log(f)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(
        "# autoresearch — physics-consistent quantum-classical world models\n\n"
        "**Mission:** prove that adding qubits to a parameter-matched classical "
        "surrogate improves simulation accuracy on The Well "
        "`turbulent_radiative_layer_2D`, via an input-conditioned Rydberg-analog "
        "block (the analog Hamiltonian adapts to the simulation state).\n\n"
        f"Branch `{branch}` @ `{commit}` · benchmark subset: 24 train / 3 valid "
        "trajectories (tcool 0.03/0.56/3.16), 64x192, 4 channels · metric: VRMSE "
        "(1-step and 8-step rollout) · fixed seed 1337 · 5-min sprints on one GB10.")]

    cells.append(nbf.v4.new_markdown_cell(
        "## System architectures\n" + "\n".join(
            f"- **{k}** — {v}" for k, v in TOPOLOGY.items())))

    for rnd in sorted(runs):
        rows = ["| variant | vrmse@1 | vrmse@8 | steps | params | vram MB | status |",
                "|---|---|---|---|---|---|---|"]
        for v, d in sorted(runs[rnd].items()):
            if d["crashed"]:
                rows.append(f"| {v} | — | — | — | — | — | CRASH |")
            else:
                rows.append(
                    f"| {v} | {float(d['val_vrmse_1']):.4f} | "
                    f"{float(d['val_vrmse_8']):.4f} | {d['num_steps']} | "
                    f"{d['num_params']} | {d['peak_vram_mb']} | ok |")
        cells.append(nbf.v4.new_markdown_cell(f"## {rnd} comparison\n" + "\n".join(rows)))

    cells.append(nbf.v4.new_markdown_cell(
        "## Tree divergence map\n```\nbaseline(none)\n"
        " ├─ mlp (classical matched control)\n"
        " ├─ hea12 ── hea16   (qubit scaling, digital ansatz)\n"
        " └─ ryd16            (analog Rydberg mechanism)\n```\n"
        "Advance rule: a branch survives if it beats BOTH `none` and `mlp` on "
        "vrmse@8 (rollout is the physics-consistency proxy)."))

    nb["cells"] = cells
    NB.parent.mkdir(exist_ok=True)
    nbf.write(nb, NB)
    print(f"wrote {NB} with {len(cells)} cells, rounds: {sorted(runs)}")


if __name__ == "__main__":
    main()
