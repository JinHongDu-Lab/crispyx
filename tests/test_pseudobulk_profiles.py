"""Tests for absolute batch-level pseudo-bulk profiles and effects."""

from __future__ import annotations

import typing
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _write_screen(tmp_path: Path, *, transformed: bool = False) -> tuple[Path, np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    records: list[dict[str, str]] = []
    for batch_index, batch in enumerate(("b1", "b2")):
        for group_index, group in enumerate(("control", "A", "B")):
            n_cells = 4 + batch_index + group_index
            for cell_index in range(n_cells):
                rows.append(
                    np.asarray(
                        [
                            batch_index + group_index + cell_index,
                            2 * group_index + (cell_index % 3),
                            batch_index + (cell_index % 2),
                        ],
                        dtype=float,
                    )
                )
                records.append({"perturbation": group, "batch": batch})
    counts = np.vstack(rows)
    X = np.log1p(counts) if transformed else counts
    obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(len(records))])
    path = tmp_path / ("log.h5ad" if transformed else "counts.h5ad")
    ad.AnnData(
        sp.csr_matrix(X),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    ).write(path)
    return path, counts, obs


def _materialise(result: cx.AnnData) -> tuple[np.ndarray, pd.DataFrame, dict]:
    return np.asarray(result.backed.X[:]), result.backed.obs.copy(), dict(result.backed.uns)


@pytest.mark.parametrize("transformed", [False, True])
def test_mean_log1p_retains_group_batch_profiles(tmp_path, transformed, capsys):
    path, counts, source_obs = _write_screen(tmp_path, transformed=transformed)
    result = cx.pb.aggregate(
        path,
        groupby=["perturbation", "batch"],
        method="mean_log1p",
        min_cells=5,
        chunk_size=4,
        output_path=tmp_path / f"profiles_{transformed}.h5ad",
        verbose=True,
    )
    actual, obs, uns = _materialise(result)
    marker = uns["crispyx_pseudobulk"]

    expected = []
    for _, row in obs.iterrows():
        mask = (
            (source_obs["perturbation"].to_numpy() == row["perturbation"])
            & (source_obs["batch"].to_numpy() == row["batch"])
        )
        expected.append(np.log1p(counts[mask]).mean(axis=0))
    np.testing.assert_allclose(actual, expected)
    assert (obs["n_cells"] >= 5).all()
    assert obs.groupby("perturbation", observed=True)["batch"].nunique().max() == 2
    assert marker["method"] == "mean_log1p"
    assert marker["input_scale"] == ("log1p" if transformed else "counts")
    message = capsys.readouterr().out
    assert ("applying log1p" in message) is (not transformed)
    result.close()


def test_sum_requires_counts_and_supports_count_layer(tmp_path):
    path, counts, obs = _write_screen(tmp_path, transformed=True)
    with pytest.raises(ValueError, match="requires raw.*integer counts"):
        cx.pb.aggregate(
            path,
            groupby=["perturbation", "batch"],
            method="sum",
            min_cells=4,
            output_path=tmp_path / "invalid.h5ad",
        )

    adata = ad.read_h5ad(path)
    adata.layers["counts"] = sp.csr_matrix(counts)
    adata.write(path)
    result = cx.pb.aggregate(
        path,
        groupby=["perturbation", "batch"],
        method="sum",
        layer="counts",
        min_cells=4,
        output_path=tmp_path / "sums.h5ad",
    )
    actual, result_obs, uns = _materialise(result)
    expected = []
    for _, row in result_obs.iterrows():
        mask = (
            (obs["perturbation"].to_numpy() == row["perturbation"])
            & (obs["batch"].to_numpy() == row["batch"])
        )
        expected.append(counts[mask].sum(axis=0))
    np.testing.assert_allclose(actual, expected)
    assert uns["crispyx_pseudobulk"]["source_layer"] == "counts"
    result.close()


def test_bootstrap_is_reproducible_and_chunk_independent(tmp_path):
    path, _, _ = _write_screen(tmp_path)
    kwargs = dict(
        groupby=["perturbation", "batch"],
        method="sum",
        min_cells=4,
        bootstrap_size=20,
    )
    first = cx.pb.aggregate(
        path,
        **kwargs,
        random_state=7,
        chunk_size=3,
        output_path=tmp_path / "first.h5ad",
    )
    second = cx.pb.aggregate(
        path,
        **kwargs,
        random_state=7,
        chunk_size=11,
        output_path=tmp_path / "second.h5ad",
    )
    different = cx.pb.aggregate(
        path,
        **kwargs,
        random_state=8,
        output_path=tmp_path / "different.h5ad",
    )
    np.testing.assert_array_equal(first.backed.X[:], second.backed.X[:])
    assert not np.array_equal(first.backed.X[:], different.backed.X[:])
    assert (first.backed.obs["n_cells_aggregated"] == 20).all()
    assert first.backed.obs["sampled_with_replacement"].all()
    first.close()
    second.close()
    different.close()


def test_effects_from_cells_equal_effects_from_saved_bulk(tmp_path):
    path, _, _ = _write_screen(tmp_path, transformed=True)
    bulk_path = tmp_path / "bulk.h5ad"
    bulk = cx.pb.aggregate(
        path,
        groupby=["perturbation", "batch"],
        method="mean_log1p",
        min_cells=4,
        bootstrap_size=7,
        random_state=3,
        output_path=bulk_path,
    )
    bulk.close()
    from_bulk = cx.pb.effects(
        bulk_path,
        groupby="perturbation",
        batch_column="batch",
        reference="control",
        output_path=tmp_path / "from_bulk.h5ad",
    )
    from_cells = cx.pb.effects(
        path,
        groupby="perturbation",
        batch_column="batch",
        reference="control",
        min_cells=4,
        bootstrap_size=7,
        random_state=3,
        output_path=tmp_path / "from_cells.h5ad",
    )
    np.testing.assert_allclose(from_bulk.backed.X[:], from_cells.backed.X[:])
    assert from_bulk.backed.obs[["perturbation", "batch"]].equals(
        from_cells.backed.obs[["perturbation", "batch"]]
    )

    aggregated = cx.pb.effects(
        bulk_path,
        groupby="perturbation",
        batch_column="batch",
        reference="control",
        aggregate_batches=True,
        output_path=tmp_path / "aggregated.h5ad",
    )
    pair_effect = np.asarray(from_bulk.backed.X[:])
    pair_obs = from_bulk.backed.obs
    for output_index, perturbation in enumerate(aggregated.backed.obs_names):
        mask = pair_obs["perturbation"].to_numpy() == perturbation
        weights = (
            pair_obs.loc[mask, "n_cells_aggregated_target"].to_numpy()
            * pair_obs.loc[mask, "n_cells_aggregated_reference"].to_numpy()
            / (
                pair_obs.loc[mask, "n_cells_aggregated_target"].to_numpy()
                + pair_obs.loc[mask, "n_cells_aggregated_reference"].to_numpy()
            )
        )
        expected = np.average(pair_effect[mask], axis=0, weights=weights)
        np.testing.assert_allclose(
            aggregated.backed.X[output_index], expected, atol=1e-15
        )
    from_bulk.close()
    from_cells.close()
    aggregated.close()


def test_matching_profile_result_reloads(tmp_path):
    path, _, _ = _write_screen(tmp_path)
    output = tmp_path / "cached.h5ad"
    first = cx.aggregate_pseudobulk(
        path,
        groupby=["perturbation", "batch"],
        method="mean_log1p",
        min_cells=4,
        output_path=output,
    )
    first.close()
    mtime = output.stat().st_mtime_ns
    second = cx.pb.aggregate(
        path,
        groupby=["perturbation", "batch"],
        method="mean_log1p",
        min_cells=4,
        output_path=output,
    )
    second.close()
    assert output.stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    "groupby", [["perturbation", "batch"], ["batch", "perturbation"]]
)
def test_perturbations_filter_is_independent_of_groupby_order(tmp_path, groupby):
    """A requested label matches any grouping column, preserving other combinations."""
    path, _, _ = _write_screen(tmp_path)
    result = cx.pb.aggregate(
        path,
        groupby=groupby,
        method="mean_log1p",
        perturbations=["A"],
        min_cells=1,
        output_path=tmp_path / f"filtered_{'_'.join(groupby)}.h5ad",
        force=True,
    )
    _, obs, _ = _materialise(result)
    result.close()

    # Every batch is retained for the requested perturbation, and nothing else.
    assert set(obs["perturbation"].astype(str)) == {"A"}
    assert sorted(obs["batch"].astype(str)) == ["b1", "b2"]


def test_unknown_perturbation_still_raises(tmp_path):
    path, _, _ = _write_screen(tmp_path)
    with pytest.raises(ValueError, match="Requested perturbation"):
        cx.pb.aggregate(
            path,
            groupby=["perturbation", "batch"],
            method="mean_log1p",
            perturbations=["missing"],
            min_cells=1,
            output_path=tmp_path / "missing.h5ad",
            force=True,
        )


@pytest.mark.parametrize(
    ("fn", "allowed"),
    [
        (cx.aggregate_pseudobulk, cx.PseudobulkMethod),
        (cx.compute_pseudobulk_effects, cx.PseudobulkMethod),
        (cx.compute_normalized_effects, cx.pseudobulk.NormalizedEffectMethod),
    ],
    ids=["aggregate", "pseudobulk_effects", "normalized_effects"],
)
def test_method_literal_is_exactly_what_each_function_accepts(tmp_path, fn, allowed):
    """Every value a function's type hint advertises runs; any other is
    rejected up front."""
    path, _, _ = _write_screen(tmp_path)
    if fn is cx.aggregate_pseudobulk:
        kwargs = dict(groupby=["perturbation", "batch"], min_cells=1)
    elif fn is cx.compute_pseudobulk_effects:
        kwargs = dict(perturbation_column="perturbation", batch_column="batch",
                      control_label="control", min_cells=1)
    else:
        kwargs = dict(perturbation_column="perturbation", control_label="control")
    for method in typing.get_args(allowed):
        fn(path, method=method, output_path=tmp_path / f"{method}.h5ad", **kwargs).close()
    rejected = {"mean_log1p", "sum", "log_mean"} - set(typing.get_args(allowed))
    for method in rejected:
        with pytest.raises(ValueError, match="method must be"):
            fn(path, method=method, output_path=tmp_path / "bad.h5ad", **kwargs)


def _write_donor_screen(tmp_path: Path) -> Path:
    """Two donors x two subclasses; sex and disease are donor-level, the
    cell-level QC metric varies within every profile."""
    rng = np.random.default_rng(0)
    records = []
    for donor, sex, disease in (("d1", "F", "AD"), ("d2", "M", "control")):
        for subclass, n in (("L2/3 IT", 6), ("L4 IT", 3)):
            for _ in range(n):
                records.append({"donor": donor, "subclass": subclass, "sex": sex,
                                "disease": disease, "pct_mito": rng.random()})
    obs = pd.DataFrame(records, index=[f"c{i}" for i in range(len(records))])
    obs["sex"] = obs["sex"].astype("category")
    counts = rng.poisson(3, (len(obs), 4)).astype(np.float32)
    path = tmp_path / "donors.h5ad"
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(4)])).write(path)
    return path


def test_profiles_are_named_by_group_and_keep_donor_covariates(tmp_path, capsys):
    path = _write_donor_screen(tmp_path)
    result = cx.pb.aggregate(
        path, groupby=["donor", "subclass"], method="sum", min_cells=1,
        output_path=tmp_path / "pb.h5ad", verbose=True,
    )
    obs = result.backed.obs.copy()
    result.close()
    assert obs.index.tolist() == ["d1|L2/3 IT", "d1|L4 IT", "d2|L2/3 IT", "d2|L4 IT"]
    assert obs["sex"].astype(str).tolist() == ["F", "F", "M", "M"]
    assert obs["disease"].tolist() == ["AD", "AD", "control", "control"]
    assert "pct_mito" not in obs
    assert "not carried over: ['pct_mito']" in capsys.readouterr().out
    # n_cells over two grouping columns is their composition table.
    table = obs.pivot(index="donor", columns="subclass", values="n_cells")
    assert table.loc["d1", "L2/3 IT"] == 6 and table.loc["d2", "L4 IT"] == 3


def test_nullable_and_missing_covariates_are_carried_over(tmp_path):
    """Donor-level columns with missing values (as concat_h5ad makes for a
    column only some files have) keep their dtype; NaN counts as a value."""
    path = _write_donor_screen(tmp_path)
    adata = ad.read_h5ad(path)
    is_d1 = (adata.obs["donor"] == "d1").to_numpy()
    adata.obs["flag"] = pd.array(np.where(is_d1, True, None), dtype="boolean")
    adata.obs["age"] = pd.array(np.where(is_d1, 70, None), dtype="Int64")
    adata.obs["score"] = np.where(is_d1, 0.5, np.nan)
    adata.write(path)
    result = cx.pb.aggregate(path, groupby="donor", method="sum", min_cells=1,
                             output_path=tmp_path / "pb.h5ad", verbose=False)
    obs = result.backed.obs.copy()
    result.close()
    assert obs["flag"].dtype == "boolean" and obs["flag"].tolist() == [True, pd.NA]
    assert obs["age"].dtype == "Int64" and obs["age"].tolist() == [70, pd.NA]
    assert obs["score"].iloc[0] == 0.5 and np.isnan(obs["score"].iloc[1])
    assert "pct_mito" not in obs


def test_constant_covariates_treat_unhashable_values_as_varying():
    from crispyx.pseudobulk import _constant_covariates

    obs = pd.DataFrame({"tags": [["a"], ["a"], ["b"]], "donor": ["d1", "d1", "d2"]})
    first, varying = _constant_covariates(obs, np.array([0, 0, 1]), 2, exclude=[])
    assert varying == ["tags"] and first["donor"].tolist() == ["d1", "d2"]


def test_constant_covariates_compare_large_integers_exactly():
    """2**53 + 1 and 2**53 are the same float64; as IDs they differ."""
    from crispyx.pseudobulk import _constant_covariates

    obs = pd.DataFrame({
        "cell_id": np.array([2**53 + 1, 2**53, 7], dtype=np.int64),
        "flag": pd.array([True, None, False], dtype="boolean"),
        "score": [np.nan, np.nan, 1.0],
    })
    first, varying = _constant_covariates(obs, np.array([0, 0, 1]), 2, exclude=[])
    assert varying == ["cell_id", "flag"] and np.isnan(first["score"].iloc[0])


def test_one_grouping_column_names_profiles_by_its_value(tmp_path):
    path = _write_donor_screen(tmp_path)
    result = cx.pb.aggregate(
        path, groupby="donor", method="sum", min_cells=1, output_path=tmp_path / "pb.h5ad",
    )
    assert result.backed.obs_names.tolist() == ["d1", "d2"]
    result.close()


def test_ambiguous_profile_names_raise(tmp_path):
    """("d1", "a|b") and ("d1|a", "b") would both be named "d1|a|b"."""
    obs = pd.DataFrame(
        {"donor": ["d1"] * 3 + ["d1|a"] * 3, "subclass": ["a|b"] * 3 + ["b"] * 3},
        index=[f"c{i}" for i in range(6)],
    )
    path = tmp_path / "clash.h5ad"
    ad.AnnData(sp.csr_matrix(np.ones((6, 2), dtype=np.float32)), obs=obs).write(path)
    with pytest.raises(ValueError, match="ambiguous"):
        cx.pb.aggregate(
            path, groupby=["donor", "subclass"], method="sum", min_cells=1,
            output_path=tmp_path / "pb.h5ad",
        )


def test_method_is_required():
    with pytest.raises(TypeError, match="method"):
        cx.aggregate_pseudobulk("unused.h5ad", groupby="donor")


def test_gene_name_column_unlike_var_names_is_written(tmp_path):
    """Profiles are named by the symbols; the symbol column itself is kept."""
    path, _, _ = _write_screen(tmp_path)
    source = ad.read_h5ad(path)
    source.var = pd.DataFrame({"gene_name": ["A1", "B2", "C3"]}, index=["ENSG01", "ENSG02", "ENSG03"])
    source.write(path)
    result = cx.pb.aggregate(path, groupby=["perturbation", "batch"], method="sum", min_cells=1,
                             gene_name_column="gene_name", output_path=tmp_path / "profiles.h5ad")
    var = result.backed.var
    assert list(var.index) == ["A1", "B2", "C3"]
    assert list(var["gene_name"]) == ["A1", "B2", "C3"]
