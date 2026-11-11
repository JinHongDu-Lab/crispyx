"""Promote re-timing run-2 outputs (crispyx) into results/<D> (plan 2026-10-25 §6, stage 2).

For each crispyx method cached in results_timing/<D>/.benchmark_cache, the previous
results/<D> cache entry and its output file move to results/<D>/.archive_pre_<tag>/, and
the new cache entry and output file take their place. Inputs that run 2 only linked
(standardized files, sorted copies) are left alone. Nothing is deleted.
Usage: promote_retime.py <tag> <dataset>... [--apply]
"""
import json, os, shutil, sys
B = "/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis/benchmarking"
args = [a for a in sys.argv[1:] if a != "--apply"]; APPLY = "--apply" in sys.argv
tag, datasets = args[0], args[1:]
def do(msg, fn):
    print(("DO   " if APPLY else "DRY  ") + msg)
    if APPLY: fn()
for d in datasets:
    new_root, old_root = f"{B}/results_timing/{d}", os.path.realpath(f"{B}/results/{d}")
    archive = f"{old_root}/.archive_pre_{tag}"
    for name in sorted(os.listdir(f"{new_root}/.benchmark_cache")):
        if not (name.startswith("crispyx_") and name.endswith(".json")):
            continue
        new = json.load(open(f"{new_root}/.benchmark_cache/{name}"))
        rel = new.get("result_path")
        if new.get("status") != "success" or not rel:
            print(f"SKIP {d}/{name}: status {new.get('status')}, result {rel}"); continue
        src = f"{new_root}/{rel}"
        if os.path.islink(src) or not os.path.isfile(src):
            print(f"SKIP {d}/{name}: {rel} is a link or missing in run 2 (input, not output)"); continue
        old_json = f"{old_root}/.benchmark_cache/{name}"
        old_rel = json.load(open(old_json)).get("result_path") if os.path.exists(old_json) else None
        def move(name=name, rel=rel, src=src, old_json=old_json, old_rel=old_rel):
            os.makedirs(f"{archive}/.benchmark_cache", exist_ok=True)
            if os.path.exists(old_json):
                shutil.move(old_json, f"{archive}/.benchmark_cache/{name}")
            for r in {old_rel, rel} - {None}:
                p = f"{old_root}/{r}"
                if os.path.lexists(p):
                    os.makedirs(os.path.dirname(f"{archive}/{r}"), exist_ok=True)
                    shutil.move(p, f"{archive}/{r}")
            os.makedirs(os.path.dirname(f"{old_root}/{rel}"), exist_ok=True)
            shutil.move(src, f"{old_root}/{rel}")
            shutil.copy2(f"{new_root}/.benchmark_cache/{name}", f"{old_root}/.benchmark_cache/{name}")
        do(f"{d}: {name}: {rel} (old: {old_rel})", move)
