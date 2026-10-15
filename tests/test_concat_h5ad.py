"""concat_h5ad: on-disk concatenation that keeps one copy of shared cells.

Each result is checked against anndata.concat of the in-memory inputs with
the shared cells removed by hand."""

from __future__ import annotations

import os
import warnings

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



def test_shared_cells_with_different_obs_warn_and_keep_the_first(tmp_path):
    paths, first, second = _pilot(tmp_path)
    second.obs.loc["ctrl0", "perturbation"] = "KO_B"
    second.write(paths[1])
    with pytest.warns(UserWarning, match=r"different values in obs columns \['perturbation'\]"):
        result = cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)
    assert ad.read_h5ad(result.path).obs.loc["ctrl0", "perturbation"] == "control"


def test_obs_compared_by_value_not_dtype(tmp_path):
    """An int column in one file and a float one in the other agree."""
    paths, first, second = _pilot(tmp_path)
    first.obs["n"] = np.arange(first.n_obs)
    second.obs["n"] = np.arange(second.n_obs).astype(np.float64)
    first.write(paths[0])
    second.write(paths[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)


def test_obs_columns_only_some_files_have(tmp_path):
    """A bool or int column one file lacks stays writable; a shared cell
    takes the values its first file lacks from its later copy."""
    paths, first, second = _pilot(tmp_path)
    first.obs["flag"] = np.arange(first.n_obs) % 2 == 0
    second.obs["guide_umi"] = np.arange(second.n_obs) + 10
    first.write(paths[0])
    second.write(paths[1])
    got = ad.read_h5ad(cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False).path)
    assert got.obs["flag"].iloc[: first.n_obs].tolist() == first.obs["flag"].tolist()
    assert got.obs["flag"].iloc[first.n_obs :].isna().all()
    # ctrl0..5 come from the first file, which has no guide_umi.
    expected = pd.Series(second.obs["guide_umi"].to_numpy(), index=second.obs_names)
    assert got.obs["guide_umi"].reindex(second.obs_names).tolist() == expected.tolist()
    assert got.obs["guide_umi"].iloc[6 : first.n_obs].isna().all()


def test_sparse_and_dataframe_obsm(tmp_path):
    paths, first, second = _pilot(tmp_path)
    for a in (first, second):
        a.obsm["guides"] = sp.csr_matrix(np.eye(a.n_obs, 4, dtype=np.float32))
        a.obsm["meta"] = pd.DataFrame({"q": np.arange(a.n_obs)}, index=a.obs_names)
    first.write(paths[0])
    second.write(paths[1])
    got = ad.read_h5ad(cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False).path)
    expected, _ = _expected(first, second)
    assert sp.issparse(got.obsm["guides"])
    np.testing.assert_array_equal(got.obsm["guides"].toarray(), _dense(expected.obsm["guides"]))
    assert list(got.obsm["meta"].columns) == ["q"]
    assert got.obsm["meta"]["q"].tolist() == expected.obsm["meta"]["q"].tolist()


def test_label_falls_back_to_paths_when_stems_repeat(tmp_path):
    paths, first, second = _pilot(tmp_path)
    (tmp_path / "rep1").mkdir()
    (tmp_path / "rep2").mkdir()
    renamed = [tmp_path / "rep1" / "filtered.h5ad", tmp_path / "rep2" / "filtered.h5ad"]
    for old, new in zip(paths, renamed):
        old.rename(new)
    got = ad.read_h5ad(
        cx.pp.concat_h5ad(renamed, output_path=tmp_path / "x.h5ad", label="src", verbose=False).path
    )
    assert got.obs["src"].cat.categories.tolist() == [str(p) for p in renamed]
    assert got.obs["src"].value_counts()[str(renamed[1])] == 4


def test_label_clash_fails_before_reading_matrices(tmp_path, monkeypatch):
    paths, _, _ = _pilot(tmp_path)

    def no_reads(*args, **kwargs):
        raise AssertionError("a matrix was read")

    monkeypatch.setattr(cx.data, "_segment_blocks", no_reads)
    with pytest.raises(ValueError, match="label column 'batch' already exists"):
        cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", label="batch", verbose=False)


def test_each_matrix_is_read_twice(tmp_path, monkeypatch):
    """One pass sizes the output and checks the shared cells, one writes it."""
    paths, _, _ = _pilot(tmp_path)
    reads = []
    original = cx.data._segment_blocks

    def counting(segment, chunk_size):
        reads.append(segment.source_path)
        return original(segment, chunk_size)

    monkeypatch.setattr(cx.data, "_segment_blocks", counting)
    cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)
    # X: scan + write per file; the "counts" layer: counting + write per file.
    assert reads.count(paths[0]) == 4 and reads.count(paths[1]) == 4


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
    with pytest.raises(ValueError, match="is the input file"):
        cx.pp.concat_h5ad(paths, output_path=paths[0], verbose=False)


def test_a_column_with_numbers_in_one_file_and_text_in_another_is_stored_as_text(tmp_path):
    paths, first, second = _pilot(tmp_path)
    first.obs["lane"] = np.arange(first.n_obs)
    second.obs["lane"] = [f"L{i}" for i in range(second.n_obs)]
    first.write(paths[0])
    second.write(paths[1])
    got = ad.read_h5ad(cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False).path)
    assert got.obs["lane"].tolist() == [str(i) for i in range(first.n_obs)] + ["L6", "L7", "L8", "L9"]


@pytest.mark.parametrize("same_categories", [True, False])
def test_categorical_order_is_kept(tmp_path, same_categories):
    paths, first, second = _pilot(tmp_path)
    phases = ["G1", "S", "G2M"]
    first.obs["phase"] = pd.Categorical(np.resize(["S", "G2M", "G1"], first.n_obs), categories=phases, ordered=True)
    second_phases = phases if same_categories else ["M", *phases]
    second.obs["phase"] = pd.Categorical(np.resize(["M" if not same_categories else "G1", "S"], second.n_obs),
                                         categories=second_phases, ordered=True)
    first.write(paths[0])
    second.write(paths[1])
    got = ad.read_h5ad(cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False).path)
    if same_categories:
        assert got.obs["phase"].dtype == pd.CategoricalDtype(phases, ordered=True)
    else:
        assert got.obs["phase"].cat.categories.tolist() == ["G1", "S", "G2M", "M"]
    assert got.obs["phase"].tolist()[:first.n_obs] == first.obs["phase"].tolist()


@pytest.mark.parametrize("policy", ["convert", "warn"])
def test_csc_inputs(tmp_path, policy):
    """A CSC file is streamed from a CSR copy (or as is, with a warning);
    the result is the same, and no copy is left behind."""
    paths, first, second = _pilot(tmp_path)
    for path, adata in zip(paths, (first, second)):
        adata.X = sp.csc_matrix(adata.X)
        adata.write(path)
    context = pytest.warns(UserWarning, match="CSC") if policy == "warn" else warnings.catch_warnings()
    with context:
        result = cx.pp.concat_h5ad(paths, output_path=tmp_path / "out" / "x.h5ad",
                                   format_mismatch_policy=policy, verbose=False)
    got = ad.read_h5ad(result.path)
    expected, _ = _expected(first, second)
    np.testing.assert_array_equal(got.X.toarray(), _dense(expected[:, got.var_names].X))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["x.h5ad"]


def test_no_file_is_opened_as_a_backed_anndata(tmp_path, monkeypatch):
    """A backed AnnData loads every layer and obsm into memory on opening."""
    paths, _, _ = _pilot(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("read_backed loads layers into memory")

    monkeypatch.setattr(cx.data, "read_backed", refuse)
    cx.pp.concat_h5ad(paths, output_path=tmp_path / "x.h5ad", verbose=False)
