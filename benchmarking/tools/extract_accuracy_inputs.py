"""Write the cells an accuracy rerun reads, so it can run elsewhere (e.g. hydra).

For a dataset config, writes benchmarking/accuracy_inputs/<D>/<kind>_<control>.h5ad:
kind "qc" (crispyx_qc_filtered counts) or "pre" (the preprocessed file), control
"all" (the accuracy subset's perturbations + every control cell), "ctrl" (the same
perturbations + the capped control sample) or "full" (the whole file, copied).
The cells are drawn exactly as the reruns draw them (_load_perturbation_subset with
the config's accuracy_subset), and rows keep their order, so a rerun on the
extract reads the same cells as on the full file.
Usage: python -m benchmarking.tools.extract_accuracy_inputs <config.yaml> qc:ctrl pre:all ...
"""
import json, shutil, sys
from pathlib import Path
import yaml
from benchmarking.tools.run_benchmarks import _load_perturbation_subset

cfg_path, wanted = Path(sys.argv[1]), sys.argv[2:]
cfg = yaml.safe_load(open(cfg_path))
spec = cfg["accuracy_subset"]
out_root = Path(cfg["output_dir"])
d = out_root.name
cache = out_root / ".benchmark_cache"
source = {"qc": out_root / json.load(open(cache / "crispyx_qc_filtered.json"))["result_path"],
          "pre": out_root / json.load(open(cache / "crispyx_preprocess.json"))["result_path"]}
dest = Path("benchmarking/accuracy_inputs") / d
dest.mkdir(parents=True, exist_ok=True)
for item in wanted:
    kind, control = item.split(":")
    target = dest / f"{kind}_{control}.h5ad"
    if control == "full":
        shutil.copyfile(source[kind], target)
    else:
        cap = int(spec["max_control_cells"]) if control == "ctrl" else None
        adata, perts = _load_perturbation_subset(source[kind], "perturbation", "control",
                                                 int(spec["n_perturbations"]), int(spec["seed"]),
                                                 max_control_cells=cap)
        adata.write_h5ad(target)
    print(f"{d}: {target} from {source[kind].name}", flush=True)
# The QC thresholds the dataset ran with (adaptive ones are cached), for the suite's paths.
(dest / "qc_params.json").write_text(json.dumps(json.load(open(cache / "config.json"))["qc_params"], indent=1))
(dest / "source.json").write_text(json.dumps({k: str(v) for k, v in source.items()} | {"accuracy_subset": spec}, indent=1))
