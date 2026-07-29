"""Tests for absolute batch-level pseudo-bulk profiles and effects."""

from __future__ import annotations

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
        min_cells=4,
        output_path=output,
    )
    first.close()
    mtime = output.stat().st_mtime_ns
    second = cx.pb.aggregate(
        path,
        groupby=["perturbation", "batch"],
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
            perturbations=["missing"],
            min_cells=1,
            output_path=tmp_path / "missing.h5ad",
            force=True,
        )
