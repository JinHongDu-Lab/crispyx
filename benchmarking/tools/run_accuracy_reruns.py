"""Run accuracy-only methods on extracted inputs (see extract_accuracy_inputs.py),
anywhere the benchmark container runs (e.g. hydra), writing the same cache entries
and outputs as run_benchmarks would, into --out/<D>/ for syncing back to results/<D>/.

The methods are the benchmark's own definitions (create_benchmark_suite with the
config's accuracy_subset); only their input path is pointed at the extract:
*_subset_ctrl methods read <kind>_ctrl.h5ad, *_subset read <kind>_all.h5ad, others
<kind>_full.h5ad (kind: "qc" for counts, "pre" for preprocessed). Limits: the
accuracy_subset block's, as in run_benchmarks.
Usage: python -m benchmarking.tools.run_accuracy_reruns <config.yaml> --out DIR --methods m1 m2 ...
"""
import argparse, json, time
from pathlib import Path
import yaml
from benchmarking.tools import run_benchmarks as rb
from benchmarking.tools.cache import save_method_result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--inputs", default="benchmarking/accuracy_inputs")
    ap.add_argument("--memory-limit", type=float, default=None,
                    help="GB: overrides each method's memory limit and its own memory budget (edgeR workers)")
    args = ap.parse_args()
    raw = yaml.safe_load(open(args.config))
    cfg = rb.BenchmarkConfig.from_dict(raw)
    d = Path(raw["output_dir"]).name
    out = Path(args.out) / d
    (out / "de").mkdir(parents=True, exist_ok=True)
    (out / "preprocessing").mkdir(parents=True, exist_ok=True)
    inputs = Path(args.inputs) / d
    qc_cached = json.load(open(inputs / "qc_params.json")) if (inputs / "qc_params.json").exists() else None
    qc = cfg.qc_params or rb.QCParams(**{k: qc_cached[k] for k in ("min_genes", "min_cells_per_perturbation", "min_cells_per_gene", "chunk_size")})
    # The suite reads this file's labels only (to confirm the control); any extract will do.
    std = out / "preprocessing" / f"standardized_{d}.h5ad"
    if not std.exists():
        std.symlink_to(next(p for p in sorted(inputs.glob("*.h5ad"))).resolve())
    suite = rb.create_benchmark_suite(
        dataset_path=std, output_dir=out, perturbation_column="perturbation", control_label="control",
        gene_name_column=cfg.gene_name_column, qc_params=qc, n_cores=cfg.n_cores, chunk_size=cfg.chunk_size,
        memory_limit_gb=cfg.resource_limits.memory_limit, accuracy_subset=cfg.accuracy_subset)
    qc_path = out / "preprocessing" / "crispyx_qc_filtered.h5ad"
    pre_path = out / "preprocessing" / f"preprocessed_{std.name}"
    spec = cfg.accuracy_subset or {}
    for name in args.methods:
        m = suite[name]
        control = "ctrl" if name.endswith("_subset_ctrl") else "all" if name.endswith("_subset") else "full"
        swap = {str(qc_path): inputs / f"qc_{control}.h5ad", str(pre_path): inputs / f"pre_{control}.h5ad"}
        def point(kw):
            kw = dict(kw)
            for k, v in kw.items():
                if k == "memory_limit_gb" and args.memory_limit and v:
                    kw[k] = min(float(v), args.memory_limit)
                elif isinstance(v, (str, Path)) and str(v) in swap:
                    kw[k] = swap[str(v)]
                elif k == "function_kwargs":
                    kw[k] = point(v)
            return kw
        m = rb.BenchmarkMethod(name=m.name, description=m.description, function=m.function,
                               kwargs=point(m.kwargs), summary=m.summary, depends_on=m.depends_on)
        limits = {k: spec[k] for k in ("memory_limit", "time_limit") if k in spec} if name.endswith(rb.ACCURACY_SUBSET_SUFFIXES) \
            else {"memory_limit": cfg.resource_limits.memory_limit, "time_limit": cfg.resource_limits.time_limit}
        if args.memory_limit:
            limits["memory_limit"] = min(limits.get("memory_limit") or args.memory_limit, args.memory_limit)
        mem = int(limits["memory_limit"] * 1024**3) if limits.get("memory_limit") else None
        context = {"output_dir": out, "dataset_path": str(std)}
        print(f"{d}: running {name} ({control} input, limits {limits})", flush=True)
        t = time.time()
        result = rb._run_with_limits(m, context, mem, limits.get("time_limit"))
        row = {"method": name, "description": m.description, "status": result.get("status"),
               "elapsed_seconds": result.get("elapsed_seconds"), "spawn_overhead_seconds": result.get("spawn_overhead_seconds"),
               "peak_memory_mb": result.get("peak_memory_mb"), "avg_memory_mb": result.get("avg_memory_mb"),
               "host": "hydra (accuracy_inputs extract)"}
        row.update(result.get("summary", {}) or {})
        if result.get("error"):
            row["error"] = result["error"]
        if name.startswith("crispyx_"):
            row["crispyx_version"] = rb._crispyx_version()
        save_method_result(name, row, out)
        print(f"{d}: {name} {row['status']} in {time.time() - t:.0f} s", flush=True)


# The runner starts methods in spawned processes, which re-import this module.
if __name__ == "__main__":
    main()
