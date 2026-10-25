"""Rename cached reference subset runs to the *_subset / *_subset_ctrl scheme (plan 2026-10-25 §3.2).

K=50, seed-0 subsets (Huang) are kept under the new names; other subset sizes are moved to
<D>/.obsolete_subsets_20261025/; the full-data record (performance row) is restored under the
base name from .benchmark_cache_full/ or results_backup_20261011/. Nothing is deleted.
Usage: migrate_subsets.py [--apply]
"""
import json, os, shutil, sys, glob
B = "/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis/benchmarking"
APPLY = "--apply" in sys.argv
REFS = ("scanpy_de_t_test", "scanpy_de_wilcoxon", "pertpy_de_pydeseq2", "edger_de_glm")
def act(msg, fn=None):
    print(("DO   " if APPLY else "DRY  ") + msg)
    if APPLY and fn: fn()
def full_record(d, m):
    for p in (f"{B}/results/{d}/.benchmark_cache_full/{m}.json",
              f"{B}/results_backup_20261011/results/{d}/.benchmark_cache/{m}.json",
              f"{B}/results_backup_20261011/results_archive_20260830/{d}/.benchmark_cache/{m}.json"):
        if os.path.exists(p) and not json.load(open(p)).get("subsampled"):
            return p
for root in sorted(glob.glob(f"{B}/results/*/")):
    d = root.rstrip("/").split("/")[-1]; c = f"{root}.benchmark_cache"
    for m in REFS:
        f = f"{c}/{m}.json"
        if not os.path.exists(f): continue
        r = json.load(open(f))
        if not r.get("subsampled"): continue
        csv = f"{root}de/{m}.csv"
        if r.get("groups") == 50:
            target = m + ("_subset_ctrl" if r.get("control_cells") else "_subset")
            def mv(f=f, csv=csv, target=target, r=r):
                shutil.move(csv, f"{root}de/{target}.csv")
                r.update(method=target, result_path=f"de/{target}.csv")
                json.dump(r, open(f"{c}/{target}.json", "w"), indent=2, sort_keys=True); os.remove(f)
            act(f"{d}: {m} (K=50{', ctrl ' + str(r.get('control_cells')) if r.get('control_cells') else ''}) -> {target}", mv)
        else:
            o = f"{root}.obsolete_subsets_20261025"
            def ob(f=f, csv=csv, o=o):
                os.makedirs(o, exist_ok=True); shutil.move(f, o); os.path.exists(csv) and shutil.move(csv, o)
            act(f"{d}: {m} (K={r.get('groups')}) -> .obsolete_subsets_20261025/", ob)
        src = full_record(d, m)
        act(f"{d}: restore full-data {m}.json from {src.replace(B + '/', '') if src else 'NONE (no full-data record)'}",
            (lambda src=src, f=f: shutil.copy2(src, f)) if src else None)
    # Failed K=50 full-control PyDESeq2 (Huang, 480 GB) = the *_subset record that led to *_subset_ctrl
    failed = f"{root}.benchmark_cache_failed/pertpy_de_pydeseq2_480GB_fullcontrol.json"
    if os.path.exists(failed) and not os.path.exists(f"{c}/pertpy_de_pydeseq2_subset.json"):
        def fl(failed=failed):
            r = json.load(open(failed)); r.update(method="pertpy_de_pydeseq2_subset")
            json.dump(r, open(f"{c}/pertpy_de_pydeseq2_subset.json", "w"), indent=2, sort_keys=True)
        act(f"{d}: failed K=50 full-control PyDESeq2 -> pertpy_de_pydeseq2_subset.json (status error)", fl)
