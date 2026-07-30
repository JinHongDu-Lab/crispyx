"""Tests for the unified normalising effect estimator, ``compute_expression_effects``."""

from __future__ import annotations

import inspect
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _write_screen(tmp_path: Path) -> tuple[Path, np.ndarray, pd.DataFrame]:
    """Two batches with deliberately unbalanced group sizes and a batch offset."""
    rng = np.random.default_rng(11)
    rows: list[np.ndarray] = []
    records: list[dict[str, str]] = []
    sizes = {("control", "b1"): 9, ("control", "b2"): 5,
             ("KO_A", "b1"): 8, ("KO_A", "b2"): 4,
             ("KO_B", "b1"): 6, ("KO_B", "b2"): 6}
    for (group, batch), n_cells in sizes.items():
        rate = 12 if batch == "b1" else 4
        rows.append(rng.poisson(rate, size=(n_cells, 6)).astype(float))
        records.extend({"perturbation": group, "batch": batch} for _ in range(n_cells))
    counts = np.vstack(rows)
    obs = pd.DataFrame(records, index=[f"cell_{i}" for i in range(len(records))])
    path = tmp_path / "counts.h5ad"
    ad.AnnData(
        sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    ).write(path)
    return path, counts, obs


def _materialise(result: cx.AnnData):
    backed = result.backed
    return (
        np.asarray(backed.X[:]),
        backed.obs.copy(),
        {k: np.asarray(v[:]) for k, v in backed.layers.items()},
        dict(backed.uns),
    )


@pytest.mark.parametrize("method", ["mean_log1p", "log_mean"])
@pytest.mark.parametrize("batch_column", [None, "batch"])
def test_effects_shape_layers_and_metadata(tmp_path, method, batch_column):
    path, _, _ = _write_screen(tmp_path)
    result = cx.pb.expression_effects(
        path,
        groupby="perturbation",
        reference="control",
        method=method,
        batch_column=batch_column,
        output_path=tmp_path / f"{method}_{batch_column}.h5ad",
    )
    values, obs, layers, uns = _materialise(result)
    result.close()

    # The reference is contrasted against, never returned as a row.
    assert sorted(obs["perturbation"].astype(str)) == ["KO_A", "KO_B"]
    assert values.shape == (2, 6)
    assert uns["method"] == method
    assert "perturbation_profile" in layers
    if method == "log_mean":
        assert "baseline_count" in uns

    if batch_column is None:
        assert "control_profile_matched" not in layers
        # Pooled: the effect is the difference of the two pooled profiles.
        np.testing.assert_allclose(
            values, layers["perturbation_profile"] - uns["control_profile"], atol=1e-12
        )
    else:
        # Batch-corrected: the reference is re-averaged with each group's own
        # weights, so the identity must hold against the matched layer.
        assert uns["batch_column"] == "batch"
        np.testing.assert_allclose(
            values,
            layers["perturbation_profile"] - layers["control_profile_matched"],
            atol=1e-12,
        )


def test_mean_of_logs_and_log_of_mean_are_different_estimators(tmp_path):
    """Averaging before or after the log gives different answers (Jensen gap)."""
    path, _, _ = _write_screen(tmp_path)
    common = dict(groupby="perturbation", reference="control")
    mean_log = np.asarray(
        cx.pb.expression_effects(
            path, method="mean_log1p", output_path=tmp_path / "a.h5ad", **common
        ).backed.X[:]
    )
    log_mean = np.asarray(
        cx.pb.expression_effects(
            path, method="log_mean", output_path=tmp_path / "b.h5ad", **common
        ).backed.X[:]
    )
    assert not np.allclose(mean_log, log_mean)


def test_batch_correction_differs_from_pooling(tmp_path):
    """Unbalanced batches mean the corrected effect must not equal the pooled one."""
    path, _, _ = _write_screen(tmp_path)
    common = dict(groupby="perturbation", reference="control", method="mean_log1p")
    pooled = np.asarray(
        cx.pb.expression_effects(
            path, output_path=tmp_path / "pooled.h5ad", **common
        ).backed.X[:]
    )
    corrected = np.asarray(
        cx.pb.expression_effects(
            path, batch_column="batch", output_path=tmp_path / "corr.h5ad", **common
        ).backed.X[:]
    )
    assert not np.allclose(pooled, corrected)


def test_matches_a_direct_harmonic_weighted_calculation(tmp_path):
    """The batch-corrected effect equals an independent per-batch computation."""
    path, counts, obs = _write_screen(tmp_path)
    result = cx.pb.expression_effects(
        path,
        groupby="perturbation",
        reference="control",
        method="mean_log1p",
        batch_column="batch",
        output_path=tmp_path / "harmonic.h5ad",
    )
    values, out_obs, _, _ = _materialise(result)
    result.close()

    # Reproduce normalize_total_block: scale each cell to target_sum=1e4, then log1p.
    library = counts.sum(axis=1, keepdims=True)
    transformed = np.log1p(counts / library * 1e4)
    labels = obs["perturbation"].to_numpy()
    batches = obs["batch"].to_numpy()

    for row, group in enumerate(out_obs["perturbation"].astype(str)):
        contrasts, weights = [], []
        for batch in ("b1", "b2"):
            group_mask = (labels == group) & (batches == batch)
            ref_mask = (labels == "control") & (batches == batch)
            n_g, n_r = int(group_mask.sum()), int(ref_mask.sum())
            contrasts.append(
                transformed[group_mask].mean(axis=0) - transformed[ref_mask].mean(axis=0)
            )
            weights.append(n_g * n_r / (n_g + n_r))
        expected = np.average(contrasts, axis=0, weights=np.asarray(weights, float))
        np.testing.assert_allclose(values[row], expected, rtol=1e-8, atol=1e-8)


def test_empty_selection_still_records_metadata(tmp_path):
    """An empty result must not omit uns keys a caller reads unconditionally."""
    path, _, _ = _write_screen(tmp_path)
    result = cx.pb.expression_effects(
        path,
        groupby="perturbation",
        reference="control",
        method="log_mean",
        perturbations=[],
        output_path=tmp_path / "empty.h5ad",
    )
    values, _, _, uns = _materialise(result)
    result.close()
    assert values.shape[0] == 0
    assert uns["method"] == "log_mean"
    assert "baseline_count" in uns
    assert "control_profile" in uns


def test_argument_guards(tmp_path):
    path, _, _ = _write_screen(tmp_path)
    common = dict(output_path=tmp_path / "bad.h5ad")

    with pytest.raises(ValueError, match="method must be 'mean_log1p' or 'log_mean'"):
        cx.pb.expression_effects(path, groupby="perturbation", method="sum", **common)

    with pytest.raises(ValueError, match="baseline_count must be positive"):
        cx.pb.expression_effects(
            path, groupby="perturbation", method="log_mean", baseline_count=0, **common
        )

    with pytest.raises(TypeError, match="aliases for the same parameter"):
        cx.pb.expression_effects(
            path, groupby="perturbation", perturbation_column="perturbation", **common
        )

    with pytest.raises(TypeError, match="aliases for the same parameter"):
        cx.pb.expression_effects(
            path, groupby="perturbation", reference="control",
            control_label="control", **common,
        )


def test_layout_is_not_public_api():
    """The output-naming switch is an implementation detail of the aliases."""
    parameters = inspect.signature(cx.compute_expression_effects).parameters
    assert "layout" not in parameters
    assert "_layout" not in parameters


@pytest.mark.parametrize(
    "deprecated,method,profile_layer,matched_layer,reference_uns",
    [
        ("compute_average_log_expression", "mean_log1p",
         "perturbation_mean", "control_mean_matched", "control_mean"),
        ("compute_pseudobulk_expression", "log_mean",
         "perturbation_bulk", "control_bulk_matched", "control_bulk"),
    ],
)
def test_deprecated_aliases_warn_and_keep_their_original_names(
    tmp_path, deprecated, method, profile_layer, matched_layer, reference_uns
):
    path, _, _ = _write_screen(tmp_path)
    with pytest.warns(DeprecationWarning, match=f"method='{method}'"):
        legacy = getattr(cx, deprecated)(
            path,
            perturbation_column="perturbation",
            control_label="control",
            batch_column="batch",
            output_path=tmp_path / f"{deprecated}.h5ad",
        )
    legacy_values, _, legacy_layers, legacy_uns = _materialise(legacy)
    legacy.close()

    # Historical output naming is part of the compatibility contract.
    assert set(legacy_layers) == {profile_layer, matched_layer}
    assert reference_uns in legacy_uns

    # The values must be identical to the replacement call.
    current = cx.pb.expression_effects(
        path,
        groupby="perturbation",
        reference="control",
        method=method,
        batch_column="batch",
        output_path=tmp_path / f"{deprecated}_new.h5ad",
    )
    current_values, _, current_layers, _ = _materialise(current)
    current.close()
    np.testing.assert_allclose(legacy_values, current_values, atol=1e-12)
    np.testing.assert_allclose(
        legacy_layers[profile_layer], current_layers["perturbation_profile"], atol=1e-12
    )
