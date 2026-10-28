"""Build notebook 20 (diagram-first, ASD-STE100 text): notebooks/20_competitions_and_geothermal_hybrid.ipynb, executed,
plus the HTML export next to it.

    .venv/bin/python notebooks/nb20_data.py && .venv/bin/python notebooks/nb20_build.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE / "20_competitions_and_geothermal_hybrid.ipynb"

CELLS = [
    ("md", """# Our two competition models and the geothermal hybrid

**Scope.** This notebook shows three systems. Each section starts with a diagram. Text is in ASD-STE100 (Simplified
Technical English).

1. **Fusion Equilibrium** (competition): our best entry and the new edits of its output.
2. **Learn2Design** (competition): our design search (submission 413).
3. **Geothermal heat plume** (paper): the classical U-Net with the 30-atom register, as in Figure 1 of the paper.

**Important.** The two competition entries use **no qubits** now. We tested a register stage in each. The register stages
did not improve the score, so we did not keep them. The geothermal model is our hybrid of classical parts and qubits."""),
    ("code", "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path.cwd() if (pathlib.Path.cwd() / 'nb20_figs.py').exists() else pathlib.Path.cwd() / 'notebooks'))\n"
             "import nb20_figs as F\n%matplotlib inline"),
    ("md", "## Overview\nThe colours show the type of each part. Orange parts use qubits."),
    ("code", "F.overview();"),
    ("md", """## 1. Fusion Equilibrium (Sophelio, NeurIPS 2026)

**Task.** Find the poloidal flux map psi (65 x 65) and two scalars (q95, betaN) for each frame of a plasma shot. Use
only coil currents, the plasma current and Thomson profiles. Do not use magnetic probes.

**Model.** Networks give a first flux map and the target shape values. Six edit steps then move the flux map to these
targets."""),
    ("code", "F.fusion_pipeline();"),
    ("md", """### 1.1 What the edits do on one frame
The white (black) line is the true boundary (LCFS). The dashed lines are our boundaries. The lines look almost the
same. But the scorer finds kappa 1.659 for X8 and 1.717 for HJl9 (truth 1.727). Near the X-point, a very small change
moves the top point by one grid row. The joint fit (HJl9) finds the change that gives the correct row."""),
    ("code", "F.fusion_frame();"),
    ("md", """### 1.2 Score of each step (VALID sample, official scorer)
Each new step adds to the score. The table shows the consistency R2 of each scalar. Green is better than X8.
The axis step gives the largest gain. The joint fit gives most of the kappa gain."""),
    ("code", "F.fusion_ladder();"),
    ("md", """### 1.3 Distance to the #1 entry (public board)
Consistency is the largest part of the gap. HJl9 made the public score 0.00032 higher (rank 16 -> 13). The consistency
part of the gap went from 0.0011 to 0.0008."""),
    ("code", "F.fusion_gap();"),
    ("md", """### 1.4 Why kappa, tri and D_LCFS stop improving
The scorer finds the boundary on a 65 x 65 grid. Thus the top and the bottom points jump from one grid row or column
to the next. A smooth edit cannot put them between two rows. The remaining error is at these points."""),
    ("code", "F.fusion_quantisation();"),
    ("md", """## 2. Learn2Design (gravitational-wave detector design)

**Task.** For a given detector topology, find the parameter values (mirrors, lengths, powers, masses) that give the
lowest noise. The organizers run our code for 4 hours on one H100 GPU for each hidden topology.

**Model.** Our entry is a search procedure, not a trained network."""),
    ("code", "F.l2d_flow();"),
    ("md", """### 2.1 Loss against evaluations
Our search finds a good design quickly. Then it stops near 0.4. The #1 entry continues to improve for the full 4 hours.
Our next test (§40t) starts the search without donors and runs for 10,000 evaluations."""),
    ("code", "F.l2d_curves();"),
    ("md", """## 3. Geothermal heat plume: classical U-Net + 30 qubits (paper design)

**Task.** Find the temperature field around one groundwater heat pump (256 x 16 cells) from four input fields.

**Reference.** The paper shows the full design on TRL2D in Figure 1 (below). The geothermal model uses the same register
with the 1HP_NN U-Net as the classical part."""),
    ("code", "from IPython.display import Image\nImage(filename=str(F.ROOT / 'paper/manuscript/iclr2027/fig_arch.png'), width=900)"),
    ("md", """### 3.1 The geothermal model
The register reads the mean of the bottleneck features. It adds a correction to all bottleneck positions. The write
weight starts at zero, so the hybrid starts as the classical model."""),
    ("code", "F.geo_arch();"),
    ("md", """### 3.2 Inside the register
The input sets the controls of each atom. Thus the register configuration changes with the input."""),
    ("code", "F.register_detail();"),
    ("md", """### 3.3 One test sample
The three predictions are almost the same. The right panel shows where the qubits make the error smaller (blue) or
larger (red)."""),
    ("code", "F.geo_fields();"),
    ("md", """### 3.4 Results
The pooled register can only correct the mean of the bottleneck. Thus it can reach only 4.7% of the error. Routing
sends 30 selected tiles to the qubits and raises this limit to 45.9%. A matched MLP in the same slot gives almost the
same gain (register -0.18%). Thus most of the gain comes from the routing, not from the qubits."""),
    ("code", "F.geo_results();"),
    ("md", """## 4. Status (4 October 2026)

| System | Best now | Next step |
|---|---|---|
| Fusion | HJl9 public S 0.99491 (rank 13; was 0.99459, rank 16) | model quality at the X-point tips; q95 / betaN |
| Learn2Design | submission 413, about 0.41 per topology at 3,000 evaluations | §40t cold start, 10,000 evaluations |
| Geothermal | +30 qubits 0.0288 vs classical 0.0291 | paper result (no new runs) |"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in CELLS]
    nbf.write(nb, NB)
    py = sys.executable
    subprocess.run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NB)], check=True, cwd=HERE)
    subprocess.run([py, "-m", "jupyter", "nbconvert", "--to", "html", "--no-input", str(NB)], check=True, cwd=HERE)
    print("NB20_BUILT", NB, NB.with_suffix(".html"))


if __name__ == "__main__":
    main()
