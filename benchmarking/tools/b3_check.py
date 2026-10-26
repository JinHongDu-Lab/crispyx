"""B3 check: crispyx NB-GLM vs PyDESeq2 with the same size factors, on the same cells.

For each dataset, pairs crispyx's NB-GLM result with the PyDESeq2 accuracy rerun
(``*_subset`` against crispyx's full-data result, ``*_subset_ctrl`` against
crispyx's rerun on the same capped cells) and reports, over the (perturbation,
gene) pairs both tested: median |delta z|, Pearson r of LFC, z, log10 p, SE and
log dispersion. Usage: python -m benchmarking.tools.b3_check <dataset>...
"""
import json, sys
import h5py, numpy as np, pandas as pd

B = "benchmarking/results"
PAIRS = (("crispyx_de_nb_glm", "pertpy_de_pydeseq2_subset"),
         ("crispyx_de_nb_glm_subset_ctrl", "pertpy_de_pydeseq2_subset_ctrl"))


def result_path(d, m):
    try:
        r = json.load(open(f"{B}/{d}/.benchmark_cache/{m}.json"))
    except FileNotFoundError:
        return None, None
    return (f"{B}/{d}/{r['result_path']}" if r.get("status") == "success" and r.get("result_path") else None), r


def crispyx_long(path, perts):
    with h5py.File(path, "r") as f:
        obs = f["obs/perturbation"]
        labels = (obs["categories"][:].astype(str)[obs["codes"][:]] if isinstance(obs, h5py.Group) else obs[:].astype(str))
        g = f["var/_index"][:].astype(str)
        rows = np.flatnonzero(np.isin(labels, list(perts)))
        get = lambda k: f[k][rows, :] if k in f else np.full((len(rows), len(g)), np.nan)
        cols = {"lfc": get("X"), "z": get("layers/z_score"), "p": get("layers/pvalue"),
                "se": get("layers/standard_error"), "disp": get("layers/dispersion")}
    n = len(rows)
    df = pd.DataFrame({"perturbation": np.repeat(labels[rows], len(g)), "gene": np.tile(g, n)})
    for k, v in cols.items():
        df[k] = np.asarray(v, dtype=float).ravel()
    return df


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan


for d in sys.argv[1:]:
    for cx_m, ref_m in PAIRS:
        cx_path, cx_r = result_path(d, cx_m)
        ref_path, ref_r = result_path(d, ref_m)
        if not (cx_path and ref_path):
            continue
        ref = pd.read_csv(ref_path)
        cx = crispyx_long(cx_path, set(ref["perturbation"].astype(str)))
        m = cx.merge(ref, on=["perturbation", "gene"], how="inner")
        m = m[np.isfinite(m["z"]) & np.isfinite(m["statistic"])]
        dz = np.abs(m["z"] - m["statistic"])
        lp = lambda p: -np.log10(np.clip(p, 1e-300, 1))
        disp = corr(np.log(m["disp"]), np.log(m["dispersion"])) if "dispersion" in m else np.nan
        print(f"{d}: {cx_m} vs {ref_m}: {m['perturbation'].nunique()} perts, {len(m):,} pairs; "
              f"median |dz| {dz.median():.2e} (90% {dz.quantile(0.9):.2e}); r LFC {corr(m['lfc'], m['log_fc']):.4f}, "
              f"z {corr(m['z'], m['statistic']):.4f}, -log10p {corr(lp(m['p']), lp(m['pvalue'])):.4f}, "
              f"SE {corr(m['se'], m['log_fc_se']):.4f}, log disp {disp:.4f}; "
              f"size factors {ref_r.get('size_factor_method')}, crispyx {cx_r.get('crispyx_version')}")
