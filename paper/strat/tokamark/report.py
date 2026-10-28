"""§37 TokaMark: tables for runs/strat/tokamark/report.md from diagnosis.json / strategies.json (prints markdown)."""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[3] / "runs" / "strat" / "tokamark"
KEY = ["scr_vanilla", "scr_mlp30", "scr_mr30ryd", "scr_ens3", "ft_vanilla_s1337", "ft_van_ens3", "E4_registered"]
STRATS = ["scr_ens3", "blend_scr3_ftvan3_train", "blend_scr3_ftall_train", "nnls_train", "groupsel_train",
          "scr3_ft9_ens", "E4_registered", "soup_scr3", "soup_ftvan3", "ftc_eq", "ftc_w", "ftc_smoke"]


def fmt_d(e: dict) -> str:
    return f"{e['delta']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] ({100 * e['rel']:+.1f}%)"


def main() -> None:
    d = json.loads((OUT / "diagnosis.json").read_text())
    s = json.loads((OUT / "strategies.json").read_text())
    key = [k for k in KEY if k in d["overall"]]
    print("## Where the error sits (VALID stratified sample, task NRMSE, lower is better)\n")
    print("| slice | " + " | ".join(key) + " |\n|---|" + "---|" * len(key))
    print("| all | " + " | ".join(f"{d['overall'][k]:.4f}" for k in key) + " |")
    for sec in ("per_group", "per_shot_stratum"):
        for g, v in d[sec].items():
            print(f"| {g} | " + " | ".join(f"{v[k]:.4f}" for k in key) + " |")
    for p, v in d["phase"].items():
        print(f"| phase {p} (n={v['n_shots']}) | " + " | ".join(f"{v[k]:.4f}" for k in key) + " |")
    print("\n### Cells (shot stratum x signal group): share of the Scratch error, Scratch vs FT-Base 3-seed ensemble\n")
    print("| cell | share | Scratch | FT-Base ens3 | Scratch ens3 | Scratch - FT ens3 [95% CI] |\n|---|---|---|---|---|---|")
    for c, v in sorted(d["cells"].items(), key=lambda x: -x[1]["share_of_ref_error"]):
        print(f"| {c} | {v['share_of_ref_error']:.3f} | {v['scr_vanilla']:.4f} | {v['ft_van_ens3']:.4f} | {v['scr_ens3']:.4f} | "
              f"{fmt_d(v['scr_vs_ftens'])} |")
    print("\n### Per signal\n\n| signal | Scratch | FT-Base ens3 | Scratch ens3 |\n|---|---|---|---|")
    for f, v in sorted(d["per_feature"].items(), key=lambda x: -x[1]["scr_vanilla"]):
        print(f"| {f.split('-')[1]} | {v['scr_vanilla']:.4f} | {v['ft_van_ens3']:.4f} | {v['scr_ens3']:.4f} |")
    print(f"\nConcentration: {d['concentration']}; metric check vs tokamark.evaluator: {d['metric_check']}\n")
    print("## Strategies vs Scratch vanilla (paired shot bootstrap, 2,000 resamples)\n")
    sc = s["valid_sample_scores"]
    print("| strategy | VALID sample | delta vs Scratch [95% CI] | delta vs Scratch ens3 [95% CI] |\n|---|---|---|---|")
    for n in sorted(sc, key=sc.get):
        a = s["vs_scr_vanilla"].get(n, {}).get("overall")
        b = s["vs_scr_ens3"].get(n, {}).get("overall")
        print(f"| {n} | {sc[n]:.5f} | {fmt_d(a) if a else '-'} | {fmt_d(b) if b else '-'} |")
    for n in [x for x in STRATS if x in s["vs_scr_vanilla"]][:3]:
        e = s["vs_scr_vanilla"][n]
        print(f"\n### {n}: per-stratum deltas vs Scratch vanilla\n\n| stratum | delta [95% CI] | wins/n |\n|---|---|---|")
        for sec in ("groups", "shot_strata", "cells"):
            for c, v in e[sec].items():
                print(f"| {c} | {fmt_d(v)} | {v['wins_a']}/{v['n']} |")


if __name__ == "__main__":
    main()
