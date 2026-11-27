"""Per-cell QC metrics (``cx.pp.calculate_qc_metrics``) and MAD outlier flags
(``cx.pp.mad_outliers``)."""

from __future__ import annotations

import platform
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import crispyx as cx


def _counts(n_obs: int = 60, n_vars: int = 30, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.uniform(0.1, 3.0, n_vars), size=(n_obs, n_vars)).astype(np.float32)
    X[0] = 0  # a cell without counts
    adata = ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]))
    adata.var["mt"] = np.arange(n_vars) < 3
    adata.var["ribo"] = np.arange(n_vars) >= n_vars - 5
    adata.var["none"] = False
    return adata


def _write(adata: ad.AnnData, path, fmt: str):
    adata = adata.copy()
    if fmt != "dense":
        adata.X = sp.csr_matrix(adata.X) if fmt == "csr" else sp.csc_matrix(adata.X)
        if fmt == "csr":
            adata.X.data[:3] = 0  # stored zeros are not detected genes
    adata.write_h5ad(path)
    return path


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_qc_metrics_match_scanpy(tmp_path, fmt, chunk_size):
    adata = _counts()
    path = _write(adata, tmp_path / "counts.h5ad", fmt)
    expected_obs, expected_var = sc.pp.calculate_qc_metrics(
        ad.read_h5ad(path), qc_vars=["mt", "ribo", "none"], percent_top=None,
    )
    obs, var = cx.pp.calculate_qc_metrics(
        path, qc_vars=["mt", "ribo", "none"], chunk_size=chunk_size, inplace=False, verbose=False,
    )
    pd.testing.assert_frame_equal(obs[expected_obs.columns], expected_obs, check_dtype=False)
    pd.testing.assert_frame_equal(var[expected_var.columns], expected_var, check_dtype=False)
    assert set(obs.columns) == set(expected_obs.columns)
    assert set(var.columns) == set(expected_var.columns)


def test_qc_metrics_written_to_file(tmp_path):
    path = _write(_counts(), tmp_path / "counts.h5ad", "csr")
    assert cx.pp.calculate_qc_metrics(path, qc_vars="mt", log1p=False, verbose=False) is None
    obs, var = cx.load_obs(path), cx.load_var(path)
    assert {"n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"} <= set(obs.columns)
    assert "log1p_total_counts" not in obs.columns
    assert {"mt", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts"} <= set(var.columns)
    assert np.isnan(obs["pct_counts_mt"].iloc[0])


def test_qc_metrics_take_a_nullable_gene_set(tmp_path):
    """concat_h5ad(join="outer") stores a gene-set flag some files lack as a
    nullable boolean; a missing flag is not in the set."""
    adata = _counts()
    path = _write(adata, tmp_path / "counts.h5ad", "dense")
    var = cx.load_var(path)
    var["mt"] = pd.array([True, None] + [False] * (adata.n_vars - 2), dtype="boolean")
    cx.write_var(path, var)
    obs, _ = cx.pp.calculate_qc_metrics(path, qc_vars="mt", inplace=False, verbose=False)
    np.testing.assert_allclose(obs["total_counts_mt"], adata.X[:, 0])


def test_qc_metrics_layer_and_errors(tmp_path):
    adata = _counts()
    adata.layers["counts"] = sp.csr_matrix(adata.X)
    adata.X = sp.csr_matrix(adata.X * 0)
    path = tmp_path / "layer.h5ad"
    adata.write_h5ad(path)
    obs, _ = cx.pp.calculate_qc_metrics(path, layer="counts", inplace=False, verbose=False)
    assert obs["total_counts"].sum() == adata.layers["counts"].sum()
    with pytest.raises(ValueError, match="not a boolean var column"):
        cx.pp.calculate_qc_metrics(path, qc_vars=["missing"], verbose=False)
    backed = ad.read_h5ad(path, backed="r")
    with pytest.raises(ValueError, match="holds its file open"):
        cx.pp.calculate_qc_metrics(backed, verbose=False)
    backed.file.close()


def _per_sample_obs(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two samples ten times apart in depth."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        "sample": np.repeat(["S1", "S2"], n),
        "log_counts": np.log1p(np.r_[rng.normal(2000, 50, n), rng.normal(20000, 500, n)]),
        "pct_mt": rng.uniform(0.5, 1.0, 2 * n),
    }, index=[f"c{i}" for i in range(2 * n)])
    obs.loc["c0", "log_counts"] = np.log1p(500)     # low in S1
    obs.loc[f"c{n}", "log_counts"] = np.log1p(2000)  # normal for S1, far below S2's cells
    obs.loc["c1", "pct_mt"] = 30.0
    return obs


def test_mad_outliers_per_group():
    obs = _per_sample_obs()
    flags = cx.pp.mad_outliers(obs, ["log_counts", "pct_mt"], groupby="sample", nmads=5)
    assert list(flags.columns) == ["log_counts_low", "log_counts_high", "pct_mt_low", "pct_mt_high"]
    assert flags.loc["c0", "log_counts_low"] and flags.loc["c200", "log_counts_low"]
    assert flags.loc["c1", "pct_mt_high"]
    assert flags.drop(index=["c0", "c1", "c200"]).to_numpy().sum() <= 4
    # Pooled, the deep sample's low cell sits inside the spread of both samples.
    pooled = cx.pp.mad_outliers(obs, "log_counts", nmads=5, side="low")
    assert list(pooled.columns) == ["log_counts_low"] and not pooled.loc["c200", "log_counts_low"]


def test_mad_outliers_matches_direct_rule(tmp_path):
    rng = np.random.default_rng(1)
    obs = pd.DataFrame({"x": rng.standard_t(3, 500), "g": rng.choice(list("abc"), 500)},
                       index=[f"c{i}" for i in range(500)])
    obs.loc["c3", "x"] = np.nan
    path = tmp_path / "obs.h5ad"
    ad.AnnData(obs=obs).write_h5ad(path)
    flags = cx.pp.mad_outliers(path, "x", groupby="g", nmads=3)
    for group, values in obs.groupby("g")["x"]:
        median = values.median()
        mad = (values - median).abs().median()
        np.testing.assert_array_equal(flags.loc[values.index, "x_high"], values - median > 3 * mad)
        np.testing.assert_array_equal(flags.loc[values.index, "x_low"], median - values > 3 * mad)
    assert not flags.loc["c3"].any()


def test_mad_outliers_edge_cases():
    obs = pd.DataFrame({"x": [0.0, 0.0, 0.0, 1.0], "flag": [True] * 4, "g": ["a", "a", None, "a"]})
    assert cx.pp.mad_outliers(obs, "x", side="high")["x_high"].tolist() == [False, False, False, True]
    with pytest.raises(TypeError, match="not numeric"):
        cx.pp.mad_outliers(obs, "flag")
    with pytest.raises(ValueError, match="missing values"):
        cx.pp.mad_outliers(obs, "x", groupby="g")
    with pytest.raises(ValueError, match="side"):
        cx.pp.mad_outliers(obs, "x", side="up")


# Scanpy's approximate neighbours (pynndescent) vary from run to run; exact ones do not.
EXACT = dict(use_approx_neighbors=False)


def _doublet_data(tmp_path, fmt: str = "csr"):
    """Two samples of cell types plus summed pairs (doublets) within each.
    The in-memory copy is CSR, as crispyx hands Scrublet each batch: Scanpy
    takes other numeric paths on a dense matrix."""
    rng = np.random.default_rng(0)
    profiles = rng.gamma(0.5, 2.0, (3, 200))
    blocks, samples = [], []
    for sample, n in (("S1", 300), ("S2", 200)):
        kinds = rng.integers(0, 3, n)
        counts = rng.poisson(profiles[kinds] * rng.uniform(0.5, 1.5, (n, 1)))
        doublets = counts[: n // 10] + counts[n // 10 : 2 * (n // 10)]
        blocks += [counts, doublets]
        samples += [sample] * (n + n // 10)
    X = np.vstack(blocks).astype(np.float32)
    adata = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"sample": pd.Categorical(samples)},
                                           index=[f"c{i}" for i in range(len(X))]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]))
    stored = adata.copy()
    if fmt != "csr":
        stored.X = X if fmt == "dense" else sp.csc_matrix(X)
    path = tmp_path / "doublets.h5ad"
    stored.write_h5ad(path)
    return path, adata


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_scrublet_matches_scanpy_per_batch(tmp_path, fmt):
    path, adata = _doublet_data(tmp_path, fmt)
    # Scanpy's Scrublet is not reproducible to the neighbour: run twice on the
    # same input with the same seed, it moved a cell by one neighbour (0.005)
    # once memory freed in between had changed the arrays' layout. The
    # automatic threshold then moves with the simulated scores and calls
    # several cells differently, so the calls are compared at a fixed one
    # that no score lies near (none between 0.112 and 0.149).
    kwargs = dict(EXACT, threshold=0.13)
    expected = adata.copy()
    sc.pp.scrublet(expected, batch_key="sample", random_state=0, verbose=False, **kwargs)
    cx.pp.scrublet(path, batch_key="sample", verbose=False, **kwargs)
    got = cx.load_obs(path)
    differ = ~np.isclose(got["doublet_score"], expected.obs["doublet_score"], rtol=1e-6)
    assert differ.sum() <= 2
    np.testing.assert_allclose(got["doublet_score"][differ], expected.obs["doublet_score"][differ], atol=0.02)
    np.testing.assert_array_equal(got["predicted_doublet"], expected.obs["predicted_doublet"].astype(bool))


def test_scrublet_releases_memory_between_samples(tmp_path, monkeypatch):
    """Each sample's freed arrays go back to the system before the next one,
    which would otherwise grow from them (twenty samples held twice one's peak)."""
    from crispyx import qc

    path, _ = _doublet_data(tmp_path)
    calls = []
    monkeypatch.setattr(qc, "release_freed_memory", lambda: calls.append(None))
    cx.pp.scrublet(path, batch_key="sample", verbose=False, **EXACT)
    # Once after the memory estimate's probe, once after Scanpy's run.
    assert len(calls) == 2 * cx.load_obs(path)["sample"].nunique()


def test_scrublet_refuses_a_sample_over_the_memory_limit(tmp_path, monkeypatch):
    """The estimate is checked before Scanpy runs, and names the sample."""
    path, _ = _doublet_data(tmp_path)
    monkeypatch.setattr(sc.pp, "scrublet", lambda *a, **k: pytest.fail("Scanpy ran"))
    with pytest.raises(MemoryError, match="sample 'S1': 330 cells .* memory_limit_gb"):
        cx.pp.scrublet(path, batch_key="sample", memory_limit_gb=1e-6, verbose=False)
    assert "doublet_score" not in cx.load_obs(path)


def test_scrublet_estimate_counts_the_dense_genes_scrublet_selects(tmp_path):
    from crispyx.qc import _SCRUBLET_BYTES_PER_DENSE, _scrublet_peak_bytes

    _, adata = _doublet_data(tmp_path)
    selected = adata.copy()
    sc.pp.filter_genes(selected, min_cells=3)
    sc.pp.normalize_total(selected)
    sc.pp.log1p(selected)
    sc.pp.highly_variable_genes(selected)
    full, n_genes = _scrublet_peak_bytes(adata, {})
    assert n_genes == selected.var["highly_variable"].sum() > 0
    # mean_center=False keeps Scrublet sparse (truncated SVD): no dense term.
    sparse_only, _ = _scrublet_peak_bytes(adata, {"mean_center": False})
    assert full - sparse_only == pytest.approx(_SCRUBLET_BYTES_PER_DENSE * 3 * adata.n_obs * n_genes)
    one_doublet_each, _ = _scrublet_peak_bytes(adata, {"sim_doublet_ratio": 1.0})
    assert full - one_doublet_each == pytest.approx(_SCRUBLET_BYTES_PER_DENSE * adata.n_obs * n_genes)


@pytest.mark.skipif(
    sys.platform != "darwin" and platform.libc_ver()[0] != "glibc", reason="glibc and macOS have a release call"
)
def test_the_allocator_release_call_is_found():
    from crispyx import _memory

    assert _memory._allocator_release() is not None
    _memory.release_freed_memory()


def test_scrublet_on_a_cell_mask(tmp_path):
    path, adata = _doublet_data(tmp_path)
    mask = np.arange(adata.n_obs) % 4 != 0
    expected = adata[mask].copy()
    sc.pp.scrublet(expected, random_state=0, verbose=False, **EXACT)
    got = cx.pp.scrublet(path, cell_mask=mask, inplace=False, verbose=False, **EXACT)
    assert got["doublet_score"][~mask].isna().all() and not got["predicted_doublet"][~mask].any()
    np.testing.assert_allclose(got["doublet_score"][mask], expected.obs["doublet_score"], rtol=1e-6)
