"""Wave G summary: per-configuration mean ± sample SD of val VRMSE@8 / @1, seed-paired wins vs the reference
register (q30 ≡ mr30ryd; seeds 1338/1339 from runs/ag15_mr30ryd_k4_s*.log) and vs the matched control,
participation for gated arms, and the pre-registered promotion verdict (3/3 vs ref AND 3/3 vs control)."""
import glob, re, statistics as st, collections, os
def parse(f):
    t = open(f).read(); out = {}
    for k in ("val_vrmse_1", "val_vrmse_8", "num_params", "train_seconds", "participation"):
        m = re.search(rf"{k}:\s+([0-9.]+)", t)
        if m: out[k] = float(m.group(1))
    return out if "val_vrmse_8" in out else None
runs = collections.defaultdict(dict)   # variant -> seed -> metrics
for f in glob.glob("runs/waveG/*_s*.log"):
    v, seed = re.match(r".*/(.+)_s(\d+)\.log$", f).groups(); m = parse(f)
    if m: runs[v][int(seed)] = m
for alias, v in (("mr30ryd", "q30"), ("mlp30", "mlp30"), ("none", "none")):   # reference trio from wave 1.5 logs
    for f in glob.glob(f"runs/ag15_{alias}_k4_s*.log"):
        seed = int(re.search(r"_s(\d+)\.log", f).group(1)); m = parse(f)
        if m and seed not in runs[v]: runs[v][seed] = m
ref = runs.get("q30", {})
print("| config | n | VRMSE@8 mean ± sd | VRMSE@1 | params | wins vs ref (paired) | wins vs control (paired) | participation | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for v in sorted(runs):
    if v.endswith("-mlp"): continue
    r = runs[v]; seeds = sorted(r); v8 = [r[s]["val_vrmse_8"] for s in seeds]; v1 = [r[s]["val_vrmse_1"] for s in seeds]
    sd = st.stdev(v8) if len(v8) > 1 else 0.0
    ctrl = runs.get(v + "-mlp", {}) if v.startswith("q30-") else runs.get("mlp30", {})
    wr = sum(1 for s in seeds if s in ref and r[s]["val_vrmse_8"] < ref[s]["val_vrmse_8"]); nr = sum(1 for s in seeds if s in ref)
    wc = sum(1 for s in seeds if s in ctrl and r[s]["val_vrmse_8"] < ctrl[s]["val_vrmse_8"]); nc = sum(1 for s in seeds if s in ctrl)
    part = st.mean([r[s]["participation"] for s in seeds if "participation" in r[s]]) if any("participation" in r[s] for s in seeds) else float("nan")
    verdict = "PROMOTE" if (v.startswith("q30-") and len(seeds) >= 3 and wr == nr == 3 and wc == nc == 3) else ("screen" if len(seeds) < 3 else "null")
    print(f"| {v} | {len(seeds)} | {st.mean(v8):.4f} ± {sd:.4f} | {st.mean(v1):.4f} | {int(r[seeds[0]].get('num_params', 0))} | {wr}/{nr} | {wc}/{nc} | {part:.2f} | {verdict} |")
