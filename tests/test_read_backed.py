"""read_backed: X and layers stay on disk; the other slots are read as
anndata's backed mode reads them."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx.data import read_backed


@pytest.fixture
def path(tmp_path):
    rng = np.random.default_rng(0)
    counts = sp.random(60, 12, density=0.3, format="csr", dtype=np.float32, random_state=0)
    counts.data = np.ceil(counts.data * 10)
    adata = ad.AnnData(
        counts.copy(),
        obs=pd.DataFrame({"g": pd.Categorical(list("ab") * 30)}, index=[f"c{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(12)]),
    )
    adata.layers["counts"] = counts.copy()
    adata.layers["csc"] = counts.tocsc()
    adata.layers["dense"] = counts.toarray()
    adata.obsm["X_pca"] = rng.normal(size=(60, 3))
    adata.obsp["conn"] = sp.random(60, 60, density=0.1, format="csr", random_state=1)
    adata.uns["note"] = "hello"
    adata.raw = adata
    out = tmp_path / "a.h5ad"
    adata.write(out)
    return out


def test_layers_stay_on_disk_and_read_as_sliced(path):
    expected = ad.read_h5ad(path)
    backed = read_backed(path)
    try:
        for key in ("counts", "csc", "dense"):
            layer = backed.layers[key]
            assert not isinstance(layer, (np.ndarray, sp.spmatrix)), key
            block = layer[5:9]
            block = block.toarray() if sp.issparse(block) else np.asarray(block)
            want = expected.layers[key][5:9]
            np.testing.assert_array_equal(block, want.toarray() if sp.issparse(want) else want)
        assert backed.isbacked and backed.obs["g"].tolist() == expected.obs["g"].tolist()
        np.testing.assert_array_equal(backed.obsm["X_pca"], expected.obsm["X_pca"])
        assert sp.issparse(backed.obsp["conn"]) and backed.uns["note"] == "hello"
        assert backed.raw is not None and backed.raw.var_names.tolist() == expected.raw.var_names.tolist()
        assert backed[:7].layers["counts"].shape == (7, 12)
        np.testing.assert_array_equal(backed.to_memory().layers["dense"], expected.layers["dense"])
    finally:
        backed.file.close()


def test_the_anndata_wrapper_and_functions_on_a_layer_use_it(path, tmp_path):
    result = cx.AnnData(path)
    try:
        assert not isinstance(result.backed.layers["counts"], (np.ndarray, sp.spmatrix))
    finally:
        result.close()
    from_layer = cx.pb.aggregate(path, groupby="g", method="sum", layer="counts", min_cells=1,
                                 output_path=tmp_path / "pb.h5ad", verbose=False)
    from_x = cx.pb.aggregate(path, groupby="g", method="sum", min_cells=1,
                             output_path=tmp_path / "pb_x.h5ad", verbose=False)
    np.testing.assert_allclose(_dense(from_layer.backed.X[:]), _dense(from_x.backed.X[:]))
    from_layer.close()
    from_x.close()


def _dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("dense", [False, True])
def test_read_rows_csc_is_the_rows_as_csc(tmp_path, dense):
    """read_rows_csc builds the selected rows' CSC in two passes over chunks of
    rows; it is the CSR rows converted, with stored zeros dropped."""
    from crispyx.data import read_rows_csc

    rng = np.random.default_rng(0)
    counts = sp.random(97, 23, density=0.2, random_state=1, format="csr", dtype=np.float32)
    counts.data = np.round(counts.data * 10)  # some become stored zeros
    path = tmp_path / "x.h5ad"
    ad.AnnData(counts.toarray() if dense else counts).write_h5ad(path)
    rows = np.sort(rng.choice(97, 60, replace=False))
    got = read_rows_csc(path, rows, chunk_rows=7)
    want = counts[rows].tocsc()
    want.eliminate_zeros()
    assert sp.isspmatrix_csc(got) and got.dtype == np.float32
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_array_equal(got.data, want.data)
