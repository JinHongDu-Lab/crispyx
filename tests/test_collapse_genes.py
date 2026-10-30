"""pp.collapse_genes: sum the rows of var that are one gene."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


GENES = ["ENSG1", "ENSG2", "ENSG3", "ENSG4_PAR_Y", "ENSG4", "ENSG5", "ENSG6"]
SYMBOLS = ["A", "B", "A", "D", "D", None, "F"]


def _write(path: Path, *, fmt="csr", n_cells=60, seed=0) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, (n_cells, len(GENES))).astype(np.float32)
    counts[rng.random(counts.shape) < 0.3] = 0
    x = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](counts)
    var = pd.DataFrame(
        {"symbol": SYMBOLS, "biotype": ["pc", "pc", "pc", "pc", "pc", "lnc", "pc"],
         "chromosome": ["1", "2", "3", "Y", "X", "5", "6"]},
        index=GENES,
    )
    obs = pd.DataFrame({"perturbation": ["control", "A"] * (n_cells // 2)}, index=[f"c{i}" for i in range(n_cells)])
    a = ad.AnnData(x, obs=obs, var=var)
    a.obsm["X_pca"] = np.zeros((n_cells, 2), dtype=np.float32)
    a.layers["copy"] = sp.csr_matrix(counts)
    a.varm["loadings"] = np.zeros((len(GENES), 2), dtype=np.float32)
    a.uns["note"] = "kept"
    a.write(path)
    return path, counts


def _expected(counts):
    return np.column_stack([
        counts[:, 0] + counts[:, 2],   # A
        counts[:, 1],                  # B
        counts[:, 3] + counts[:, 4],   # D
        counts[:, 6],                  # F
    ])


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_rows_sharing_a_key_are_summed_and_missing_keys_dropped(tmp_path, fmt):
    src, counts = _write(tmp_path / "src.h5ad", fmt=fmt)
    with pytest.warns(UserWarning, match="layers, varm"):
        out = cx.pp.collapse_genes(src, output_path=tmp_path / "out.h5ad", by="symbol", verbose=False,
                                   format_mismatch_policy="off")
    got = ad.read_h5ad(out.filename)
    assert sp.isspmatrix_csr(got.X) and got.X.dtype == np.float32
    assert list(got.var_names) == ["A", "B", "D", "F"]
    np.testing.assert_array_equal(got.X.toarray(), _expected(counts))
    assert got.var["n_source_genes"].tolist() == [2, 1, 2, 1]
    assert got.var["biotype"].tolist() == ["pc"] * 4      # constant within every group: kept
    assert "chromosome" not in got.var.columns              # differs between A's rows: dropped
    assert "X_pca" in got.obsm and not got.layers and not got.varm
    assert got.uns["note"] == "kept"
    assert got.uns["crispyx_collapse_genes"]["n_genes"] == 4 and got.uns["crispyx_collapse_genes"]["n_dropped"] == 1
    assert list(got.obs_names) == [f"c{i}" for i in range(60)]


def test_a_key_series_collapses_par_y_rows(tmp_path):
    src, counts = _write(tmp_path / "src.h5ad")
    var = cx.load_var(src)
    keys = pd.Series(var.index.str.replace(r"_PAR_Y$", "", regex=True), index=var.index)
    with pytest.warns(UserWarning):
        out = cx.pp.collapse_genes(src, output_path=tmp_path / "out.h5ad", by=keys, verbose=False)
    got = ad.read_h5ad(out.filename)
    assert list(got.var_names) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5", "ENSG6"]
    np.testing.assert_array_equal(got.X[:, 3].toarray().ravel(), counts[:, 3] + counts[:, 4])
    assert got.var.loc["ENSG4", "symbol"] == "D" and got.var.loc["ENSG4", "n_source_genes"] == 2
    assert got.uns["crispyx_collapse_genes"]["by"] == "keys"


def test_result_is_independent_of_chunk_size(tmp_path):
    src, _ = _write(tmp_path / "src.h5ad", n_cells=70)
    with pytest.warns(UserWarning):
        a = cx.pp.collapse_genes(src, output_path=tmp_path / "a.h5ad", by="symbol", chunk_size=7, verbose=False)
        b = cx.pp.collapse_genes(src, output_path=tmp_path / "b.h5ad", by="symbol", chunk_size=1000, verbose=False)
    np.testing.assert_array_equal(ad.read_h5ad(a.filename).X.toarray(), ad.read_h5ad(b.filename).X.toarray())


def test_bad_keys_raise_and_leave_no_output(tmp_path):
    src, _ = _write(tmp_path / "src.h5ad")
    with pytest.raises(KeyError, match="no column 'nope'"):
        cx.pp.collapse_genes(src, output_path=tmp_path / "out.h5ad", by="nope", verbose=False)
    with pytest.raises(ValueError, match="3 keys for 7 genes"):
        cx.pp.collapse_genes(src, output_path=tmp_path / "out.h5ad", by=["a", "b", "c"], verbose=False)
    with pytest.raises(ValueError, match="every gene has a missing key"):
        cx.pp.collapse_genes(src, output_path=tmp_path / "out.h5ad", by=[None] * 7, verbose=False)
    assert not (tmp_path / "out.h5ad").exists()


def test_in_memory_anndata_is_accepted(tmp_path):
    src, counts = _write(tmp_path / "src.h5ad")
    adata = ad.read_h5ad(src)
    del adata.layers["copy"], adata.varm["loadings"]
    out = cx.pp.collapse_genes(adata, output_path=tmp_path / "out.h5ad", by="symbol", verbose=False)
    np.testing.assert_array_equal(ad.read_h5ad(out.filename).X.toarray(), _expected(counts))
