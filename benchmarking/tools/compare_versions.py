"""New (results_timing/<D>, crispyx 0.2.3) vs previous (results/<D>) crispyx outputs."""
import sys, glob, os, numpy as np, anndata as ad, scipy.sparse as sp, json
B = "benchmarking"
def dense(m):
    return m.toarray() if sp.issparse(m) else np.asarray(m)
def cmp_mat(a, b, name):
    if a.shape != b.shape: return f"{name}: shape {a.shape} vs {b.shape}"
    if sp.issparse(a) and sp.issparse(b):
        d = abs(a - b); mx = d.max() if d.nnz else 0.0
        return f"{name}: max|diff| {mx:.3g}" + ("" if mx else " (identical)")
    a, b = dense(a).astype(float), dense(b).astype(float)
    both = np.isnan(a) & np.isnan(b); d = np.where(both, 0, np.abs(a - b))
    nan_mismatch = int((np.isnan(a) ^ np.isnan(b)).sum())
    mx = np.nanmax(d) if d.size else 0
    return f"{name}: max|diff| {mx:.3g}, NaN mismatches {nan_mismatch}" + (" (identical)" if mx == 0 and nan_mismatch == 0 else "")
def pair(new, old):
    a, b = ad.read_h5ad(new), ad.read_h5ad(old)
    out = [f"cells {a.n_obs} vs {b.n_obs}, genes {a.n_vars} vs {b.n_vars}"]
    same_obs = list(a.obs_names) == list(b.obs_names); same_var = list(a.var_names) == list(b.var_names)
    if not (same_obs and same_var):
        common_o = a.obs_names.intersection(b.obs_names); common_v = a.var_names.intersection(b.var_names)
        out.append(f"names differ: obs same={same_obs} var same={same_var}; comparing {len(common_o)} x {len(common_v)} common")
        a, b = a[common_o, common_v], b[common_o, common_v]
    out.append(cmp_mat(a.X, b.X, "X"))
    for k in sorted(set(a.layers) & set(b.layers)):
        out.append(cmp_mat(a.layers[k], b.layers[k], f"layers/{k}"))
    return "; ".join(out)
for d in sys.argv[1:]:
    print(f"## {d}")
    new_root, old_root = f"{B}/results_timing/{d}", f"{B}/results/{d}"
    def old_de(m):
        try: return json.load(open(f"{old_root}/.benchmark_cache/{m}.json"))["result_path"]
        except Exception: return None
    pairs = [("preprocessing/crispyx_qc_filtered.h5ad", "preprocessing/crispyx_qc_filtered.h5ad")]
    for m, n in (("crispyx_de_t_test", "de/de_cx_t_test.h5ad"), ("crispyx_de_wilcoxon", "de/de_cx_wilcoxon.h5ad")):
        if old_de(m): pairs.append((n, old_de(m)))
    pre_new = glob.glob(f"{new_root}/preprocessing/preprocessed_standardized*.h5ad"); pre_old = glob.glob(f"{old_root}/preprocessing/preprocessed_standardized*.h5ad")
    pre_new = [p for p in pre_new if "_csc_" not in p]; pre_old = [p for p in pre_old if "_csc_" not in p and "_csr_" not in p] or [p for p in pre_old if "_csc_" not in p]
    if pre_new and pre_old: pairs.append((os.path.relpath(pre_new[0], new_root), os.path.relpath(pre_old[0], old_root)))
    for pat in ("pb_avg_log", "pb_pseudobulk"):
        n = glob.glob(f"{new_root}/preprocessing/*{pat}*.h5ad"); o = glob.glob(f"{old_root}/preprocessing/*{pat}*.h5ad")
        o = [x for x in o if "crispyx_pb_avg_log_avg_log" not in x or pat == "pb_avg_log"]
        if n and o: pairs.append((os.path.relpath(n[0], new_root), os.path.relpath(sorted(o)[0], old_root)))
    for n, o in pairs:
        try:
            print(f"  {n}  vs  {o}:\n     {pair(f'{new_root}/{n}', f'{old_root}/{o}')}")
        except Exception as e:
            print(f"  {n}: ERROR {type(e).__name__}: {e}")
