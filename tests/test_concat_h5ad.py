"""concat_h5ad: on-disk concatenation that keeps one copy of shared cells.

Each result is checked against anndata.concat of the in-memory inputs with
the shared cells removed by hand."""

from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _pilot(tmp_path, *, dense_second=False):
    """Two files sharing 6 control cells. The second lists its genes in
    another order, has one gene the first lacks and lacks one the first has."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(8)]
    controls = rng.poisson(2, (6, 8)).astype(np.float32)

    def build(n_pert, target, var_names, seed):
        local = np.random.default_rng(seed)
        perturbed = local.poisson(3, (n_pert, len(var_names))).astype(np.float32)
        ctrl = pd.DataFrame(controls, columns=genes).reindex(columns=var_names, fill_value=1.0).to_numpy()
        X = np.vstack([ctrl, perturbed]).astype(np.float32)
        obs = pd.DataFrame(
            {"perturbation": ["control"] * 6 + [target] * n_pert,
             "batch": pd.Categorical(["b"] * (6 + n_pert))},
            index=[f"ctrl{i}" for i in range(6)] + [f"{target}_{i}" for i in range(n_pert)],
        )
        a = ad.AnnData(X, obs=obs, var=pd.DataFrame({"chrom": "1"}, index=var_names))
        a.layers["counts"] = a.X.copy()
        a.obsm["X_pca"] = local.normal(size=(a.n_obs, 3))
        return a

    first = build(5, "KO_A", genes, 1)
    second_genes = ["g7", "g0", "g3", "g2", "g1", "g5", "g4", "extra"]  # no g6
    second = build(4, "KO_B", second_genes, 2)
    first.X = sp.csr_matrix(first.X)
    first.layers["counts"] = sp.csr_matrix(first.layers["counts"])
    if not dense_second:
        second.X = sp.csr_matrix(second.X)
        second.layers["counts"] = sp.csr_matrix(second.layers["counts"])
    second.layers["only_here"] = second.X.copy()
    paths = [tmp_path / "pilot_a.h5ad", tmp_path / "pilot_b.h5ad"]
    first.write(paths[0])
    second.write(paths[1])
    return paths, first, second


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _expected(first, second):
    """anndata.concat of the inputs, the second without the shared cells."""
    shared = second.obs_names.isin(first.obs_names)
    joined = ad.concat([first, second[~shared]], join="inner")
    return joined, int(shared.sum())


@pytest.mark.parametrize("dense_second", [False, True], ids=["csr", "dense"])
def test_concat_keeps_one_copy_of_shared_cells(tmp_path, dense_second, capsys):
    paths, first, second = _pilot(tmp_path, dense_second=dense_second)
    result = cx.pp.concat_h5ad(paths, output_path=tmp_path / "pilot.h5ad", label="source")
    out = capsys.readouterr().out
    got = ad.read_h5ad(result.path)
    expected, n_shared = _expected(first, second)

    assert n_shared == 6 and "6 cells of" in out
    assert got.obs_names.is_unique
    assert got.n_obs == first.n_obs + second.n_obs - n_shared
    assert got.obs_names.tolist() == expected.obs_names.tolist()
    # Genes every file has, in the first file's order.
    assert got.var_names.tolist() == ["g0", "g1", "g2", "g3", "g4", "g5", "g7"]
    assert "2 genes not in every file are dropped" in out
    assert sp.issparse(got.X)
    expected = expected[:, got.var_names]
    np.testing.assert_array_equal(got.X.toarray(), _dense(expected.X))
    np.testing.assert_array_equal(got.layers["counts"].toarray(), _dense(expected.layers["counts"]))
    assert "only_here" not in got.layers and "layers/only_here" in out
    np.testing.assert_array_equal(got.obsm["X_pca"], expected.obsm["X_pca"])
    assert got.obs["source"].tolist() == ["pilot_a"] * first.n_obs + ["pilot_b"] * 4
    assert isinstance(got.obs["batch"].dtype, pd.CategoricalDtype)
    assert got.var["chrom"].tolist() == ["1"] * 7
    assert got.uns["crispyx"]["kind"] == "concat"


def test_shared_cells_error_mode_raises(tmp_path):
    paths, _, _ = _pilot(tmp_path)
    with pytest.raises(ValueError, match="6 cells of .* also occur in an earlier file"):
        cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", shared_cells="error", verbose=False)


def test_same_barcode_different_cell_raises(tmp_path):
    paths, _, second = _pilot(tmp_path)
    second.X = sp.csr_matrix(second.X.toarray() + np.eye(second.n_obs, second.n_vars, dtype=np.float32))
    second.write(paths[1])
    with pytest.raises(ValueError, match="with different expression"):
        cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)

    paths, _, second = _pilot(tmp_path)
    second.obs.loc["ctrl0", "perturbation"] = "KO_B"
    second.write(paths[1])
    with pytest.raises(ValueError, match=r"different obs\['perturbation'\]"):
        cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)


def test_rerun_reuses_until_an_input_changes(tmp_path, capsys):
    paths, _, _ = _pilot(tmp_path)
    output = tmp_path / "pilot.h5ad"
    cx.pp.concat_h5ad(paths, output_path=output)
    capsys.readouterr()
    cx.pp.concat_h5ad(paths, output_path=output)
    assert "Loading existing result" in capsys.readouterr().out
    stat = paths[1].stat()
    os.utime(paths[1], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    cx.pp.concat_h5ad(paths, output_path=output)
    assert "an input file has changed since" in capsys.readouterr().out
    cx.pp.concat_h5ad(paths[::-1], output_path=output)
    assert "is not reused" in capsys.readouterr().out


def test_output_must_not_be_an_input(tmp_path):
    paths, _, _ = _pilot(tmp_path)
    with pytest.raises(ValueError, match="must not be one of the input files"):
        cx.pp.concat_h5ad(paths, output_path=paths[0], verbose=False)
