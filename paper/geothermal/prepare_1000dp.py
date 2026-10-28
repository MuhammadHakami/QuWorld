"""Prepare the 1000-run raw PFLOTRAN set (DaRUS darus-3650, dataset_2d_small_1000dp) with
1HP_NN's OWN first-stage pipeline (baseline/1HP_NN, Pelzer et al.), inputs "gksi"
[pressure gradient, permeability, SDF, material id] and the 27.5-year label — the
published U-Net protocol (Pelzer & Schulte 2024). Output: data/geothermal/prepared_1000dp_gksi/
{Inputs,Labels}/RUN_i.pt + info.yaml (Welford stats, Rescale normalization).
Usage: uv run paper/geothermal/prepare_1000dp.py [inputs=gksi]
"""
import pathlib, sys, time, types, os

ROOT = pathlib.Path(__file__).resolve().parents[2]
HP = ROOT / "baseline" / "1HP_NN"
INPUTS = sys.argv[1] if len(sys.argv) > 1 else "gksi"
RAW = ROOT / "data" / "geothermal" / "darus-3650" / "dataset_2d_small_1000dp"
OUT = ROOT / "data" / "geothermal" / f"prepared_1000dp_{INPUTS}"

def main() -> None:
    if not (RAW / "inputs" / "settings.yaml").exists():
        raise FileNotFoundError(f"raw set incomplete: {RAW}/inputs/settings.yaml")
    runs = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("RUN_"))
    if len(runs) != 1000:
        raise ValueError(f"expected 1000 RUN_* dirs, found {len(runs)}")
    os.chdir(HP); sys.path.insert(0, str(HP))
    sys.modules.setdefault("pynvml", types.ModuleType("pynvml"))  # 1HP_NN utils import pynvml for GPU logging only
    from preprocessing.prepare_1ststage import prepare_dataset  # noqa: E402
    settings = types.SimpleNamespace(inputs=INPUTS, len_box=256, problem="2stages", case="train", case_2hp=False)
    paths = types.SimpleNamespace(raw_path=RAW, dataset_1st_prep_path=OUT)
    t0 = time.time()
    power2 = "--no-power2" not in sys.argv  # default: crop to powers of two (256x20 -> 256x16), as 1HP_NN main.py does and as the 100-dp benchmark set is
    info = prepare_dataset(paths, settings, power2trafo=power2, cutlengthtrafo=False, info=None)
    n_in = len(list((OUT / "Inputs").glob("*.pt"))); n_lab = len(list((OUT / "Labels").glob("*.pt")))
    print(f"[prep1000] done in {time.time()-t0:.0f}s: {n_in} inputs, {n_lab} labels -> {OUT}; cells {info['CellsNumber']} hp {info['PositionLastHP']}", flush=True)
    if n_in != 1000 or n_lab != 1000:
        raise RuntimeError(f"incomplete preparation: {n_in} inputs / {n_lab} labels")

if __name__ == "__main__":
    main()
