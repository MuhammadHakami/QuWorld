"""RouteFinder hybrid qubit ladder (doc 35 §16/§19): the vanilla RF-Transformer fine-tuned under EAL is the fixed
classical part (K = 0); the routed node register adds K input-configured atoms (K = 5, 10, 20, 30), against a matched
MLP on the same routing. Every arm trains on vanilla's generated instances (common random numbers).

    .venv/bin/python paper/vrp/rf_ladder.py     # -> paper/manuscript/iclr2027/fig_rf_ladder.pdf, runs/routefinder/ladder.json

Left: pilot budget (1 epoch x 2,000 instances, seed 1). Right: the EAL protocol (10 epochs x 10,000), seeds as available.
Mean test gap to the reference solutions over the 16 multi-depot variants (lower is better).
"""
from __future__ import annotations

import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
RF = ROOT / "runs/routefinder"
KS = (5, 10, 20, 30)
ARMS = (("route{K}", "#d95f02", "register"), ("route{K}-mlp", "#1b9e77", "matched MLP"))


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def budget(kind):
    """kind 'pilot' -> CRN pilots (vanilla pilot from scaling/, same instance stream); 'full' -> EAL protocol."""
    seeds = (1,) if kind == "pilot" else (1, 2, 3)
    d = RF / ("crn" if kind == "pilot" else "full")
    van = [load(RF / "scaling" / "eal_md_vanilla_s1.json")] if kind == "pilot" else [load(d / f"eal_md_vanilla_s{s}.json") for s in seeds]
    out = {"vanilla": [r for r in van if r and r.get("crn", kind == "pilot")]}
    for pat, _, _ in ARMS:
        for K in KS:
            v = pat.format(K=K)
            out[v] = [r for r in (load(d / f"eal_md_{v}_s{s}.json") for s in seeds) if r and r.get("crn")]
    return out


def summary(runs):
    res = {}
    van = {r["seed"]: r for r in runs["vanilla"]}
    for v, rr in runs.items():
        if not rr:
            continue
        g = [r["test_gap_mean"] for r in rr]
        e = {"seeds": [r["seed"] for r in rr], "gap": g, "mean": st.mean(g)}
        paired = [(r, van[r["seed"]]) for r in rr if r["seed"] in van and v != "vanilla"]
        if paired:
            e["variants_better_than_vanilla"] = [sum(r["test_gap"][k] < b["test_gap"][k] for k in b["test_gap"]) for r, b in paired]
        res[v] = e
    for K in KS:
        r, m = res.get(f"route{K}"), res.get(f"route{K}-mlp")
        if r and m:
            common = sorted(set(r["seeds"]) & set(m["seeds"]))
            gr = [r["gap"][r["seeds"].index(s)] for s in common]; gm = [m["gap"][m["seeds"].index(s)] for s in common]
            res[f"reg_vs_mlp_K{K}"] = {"seeds": common, "diff": st.mean(gr) - st.mean(gm), "reg_better_seeds": sum(a < b for a, b in zip(gr, gm))}
    return res


def main():
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4)); out = {}
    for j, (kind, title) in enumerate((("pilot", "pilot budget (1 epoch, 2k instances, seed 1)"),
                                       ("full", "EAL protocol (10 epochs, 10k instances)"))):
        runs = budget(kind); res = summary(runs); out[kind] = res; a = ax[j]
        if "vanilla" not in res:
            a.set_title(title + ": pending", fontsize=7); continue
        v0 = res["vanilla"]["mean"]
        a.axhline(v0, color="0.5", ls="--", lw=0.7, label="vanilla ($K{=}0$)")
        a.scatter([0] * len(res["vanilla"]["gap"]), res["vanilla"]["gap"], s=6, color="0.3")
        for pat, col, lab in ARMS:
            ks = [K for K in KS if pat.format(K=K) in res]
            if not ks:
                continue
            a.plot([0] + ks, [v0] + [res[pat.format(K=K)]["mean"] for K in ks], "o-", color=col, ms=3, lw=0.9, label=lab)
            for K in ks:
                g = res[pat.format(K=K)]["gap"]; a.scatter([K] * len(g), g, s=5, color=col, alpha=0.5)
        a.set_xticks([0, 5, 10, 20, 30]); a.set_xlabel("qubits $K$"); a.set_title(title, fontsize=7)
        if j == 0:
            a.set_ylabel("mean test gap (%)"); a.legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3); fig.savefig(ROOT / "paper/manuscript/iclr2027/fig_rf_ladder.pdf", bbox_inches="tight")
    (RF / "ladder.json").write_text(json.dumps(out, indent=1))
    write_table(out)
    for kind, res in out.items():
        print(kind, {v: (round(e["mean"], 3), e.get("variants_better_than_vanilla")) for v, e in res.items() if "mean" in e})
        print(kind, {k: v for k, v in res.items() if k.startswith("reg_vs")})


def write_table(out):
    """LaTeX rows for app:routing: mean test gap (%) per arm and budget; seeds; variants better than vanilla (seed mean)."""
    lines = ["\\begin{tabular}{llccc}", "\\toprule", "Budget & arm & seeds & mean gap (\\%) & variants better than vanilla \\\\", "\\midrule"]
    for kind, lab in (("pilot", "pilot"), ("full", "EAL protocol")):
        res = out.get(kind, {})
        order = ["vanilla"] + [pat.format(K=K) for K in KS for pat, _, _ in ARMS]
        first = True
        for v in order:
            e = res.get(v)
            if not e:
                continue
            name = "vanilla ($K{=}0$)" if v == "vanilla" else (f"register, $K{{=}}{v[5:].split('-')[0]}$" if not v.endswith("-mlp") else f"matched MLP, $K{{=}}{v[5:].split('-')[0]}$")
            wins = e.get("variants_better_than_vanilla")
            lines.append(f"{lab if first else ''} & {name} & {len(e['seeds'])} & {e['mean']:.2f} & {('/'.join(str(w) for w in wins) + ' of 16') if wins else '---'} \\\\")
            first = False
        if not first and kind == "pilot":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/manuscript/iclr2027/rf_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
