"""QC: every strategy gives the same result, and it matches Scanpy."""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from crispyx.qc import _qc_column_oriented, _qc_in_memory, _qc_row_oriented, quality_control_summary

ADAMSON = {
    "path": Path(__file__).resolve().parents[1] / "data" / "Adamson_subset.h5ad",
    "perturbation_column": "perturbation",
}

# QC parameters used on the real dataset
QC_PARAMS = {
    "min_genes": 100,
    "min_cells_per_perturbation": 50,
    "min_cells_per_gene": 100,
}


def _make_uneven_screen(path: Path, fmt: str, seed: int = 0) -> Path:
    """Perturbation groups of 5-40 cells and genes of varying density, so each
    QC step (cells, perturbations, genes) removes something."""
    rng = np.random.default_rng(seed)
    sizes = [100] + list(range(5, 45, 4))
    labels = np.repeat(["NTC"] + [f"P{i}" for i in range(len(sizes) - 1)], sizes)
    rng.shuffle(labels)
    n_cells, n_genes = labels.size, 300  # enough non-zeros to flush the writer's buffer
    density = np.linspace(0.02, 0.5, n_genes)
    X = (rng.random((n_cells, n_genes)) < density) * rng.integers(1, 10, (n_cells, n_genes))
    X[: n_cells // 10] *= rng.random((n_cells // 10, n_genes)) < 0.1  # some near-empty cells
    X = X.astype(np.float32)
    obs = pd.DataFrame({"perturbation": pd.Categorical(labels)}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    store = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](X)
    ad.AnnData(X=store, obs=obs, var=var).write_h5ad(path)
    return path


_STRATEGIES = {
    "column": lambda path, out, **kw: _qc_column_oriented(path, output_path=out, chunk_size=7, **kw),
    "row-memmap-delta": lambda path, out, **kw: _qc_row_oriented(
        path, output_path=out, chunk_size=7, cache_mode="memmap", delta_threshold=1e9, **kw),
    "row-memory-recompute": lambda path, out, **kw: _qc_row_oriented(
        path, output_path=out, chunk_size=7, cache_mode="memory", delta_threshold=0.0, **kw),
    "row-no-cache": lambda path, out, **kw: _qc_row_oriented(
        path, output_path=out, chunk_size=7, cache_mode="none", **kw),
}


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
@pytest.mark.parametrize("strategy", list(_STRATEGIES))
def test_streaming_strategies_match_in_memory_qc(tmp_path, fmt, strategy):
    path = _make_uneven_screen(tmp_path / f"{fmt}.h5ad", fmt)
    kw = dict(perturbation_column="perturbation", control_label="NTC", gene_name_column=None,
              min_genes=4, min_cells_per_perturbation=15, min_cells_per_gene=20)
    expected = _qc_in_memory(path, output_path=tmp_path / "memory.h5ad", **kw)
    # The fixture must exercise every filter, or the comparison proves little.
    assert 0 < expected.cell_mask.sum() < expected.cell_mask.size
    assert 0 < expected.gene_mask.sum() < expected.gene_mask.size
    all_groups = set(ad.read_h5ad(path, backed="r").obs["perturbation"].cat.categories)
    assert {g for g, keep in expected.perturbation_keep.items() if keep} < all_groups
    assert sp.csr_matrix(ad.read_h5ad(tmp_path / "memory.h5ad").X).nnz > 2 * 8192

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # slow-axis streaming on purpose
        actual = _STRATEGIES[strategy](path, tmp_path / "streamed.h5ad", **kw)
    np.testing.assert_array_equal(actual.cell_mask, expected.cell_mask)
    np.testing.assert_array_equal(actual.gene_mask, expected.gene_mask)
    assert actual.perturbation_keep == expected.perturbation_keep
    source = ad.read_h5ad(path)
    written = ad.read_h5ad(tmp_path / "streamed.h5ad")
    reference = source[expected.cell_mask][:, expected.gene_mask]
    np.testing.assert_array_equal(np.asarray(sp.csr_matrix(written.X).todense()),
                                  np.asarray(sp.csr_matrix(reference.X).todense()))
    pd.testing.assert_index_equal(written.obs_names, reference.obs_names)
    pd.testing.assert_index_equal(written.var_names, reference.var_names)


def test_quality_control_summary_dispatch(tmp_path):
    """Test that quality_control_summary correctly dispatches based on data size."""
    
    dataset = ADAMSON
    if not dataset["path"].exists():
        pytest.skip(f"data/{dataset['path'].name} not found")
    
    # Test with force_streaming=False (should use in-memory for small data)
    result1 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_path,
        data_name="test1",
        force_streaming=False,
        **QC_PARAMS,
    )
    
    # Test with force_streaming=True (should use streaming)
    result2 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_path,
        data_name="test2",
        force_streaming=True,
        **QC_PARAMS,
    )
    
    # Results should be identical
    assert np.array_equal(result1.cell_mask, result2.cell_mask), (
        f"Cell mask mismatch between dispatch modes: "
        f"{result1.cell_mask.sum()} vs {result2.cell_mask.sum()}"
    )
    assert np.array_equal(result1.gene_mask, result2.gene_mask), (
        f"Gene mask mismatch between dispatch modes: "
        f"{result1.gene_mask.sum()} vs {result2.gene_mask.sum()}"
    )


def test_qc_against_scanpy(tmp_path):
    """Compare crispyx QC results against Scanpy QC as ground truth."""
    import scanpy as sc
    
    dataset = ADAMSON
    if not dataset["path"].exists():
        pytest.skip(f"data/{dataset['path'].name} not found")
    
    from crispyx.data import resolve_control_label, read_backed
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    # Run crispyx QC
    crispyx_result = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_path,
        data_name="crispyx",
        **QC_PARAMS,
    )
    
    # Run Scanpy QC
    adata = ad.read_h5ad(dataset["path"])
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=QC_PARAMS["min_genes"])
    
    # Filter perturbations
    labels = adata.obs[dataset["perturbation_column"]].astype(str)
    counts = labels.value_counts()
    keep = labels.eq(control_label) | counts.loc[labels].ge(QC_PARAMS["min_cells_per_perturbation"]).to_numpy()
    adata = adata[keep].copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=QC_PARAMS["min_cells_per_gene"])
    
    # Compare results
    assert crispyx_result.cell_mask.sum() == adata.n_obs, (
        f"Cell count mismatch: crispyx={crispyx_result.cell_mask.sum()}, scanpy={adata.n_obs}"
    )
    assert crispyx_result.gene_mask.sum() == adata.n_vars, (
        f"Gene count mismatch: crispyx={crispyx_result.gene_mask.sum()}, scanpy={adata.n_vars}"
    )


def _make_synthetic_h5ad(dir_path, fmt, seed=0):
    """Write a small synthetic h5ad in the requested storage format ('csr'/'csc')."""

    rng = np.random.default_rng(seed)
    n, g = 400, 60
    X = sp.random(n, g, density=0.15, random_state=seed,
                  data_rvs=lambda s: rng.integers(1, 10, s)).tocsr()
    X.data = X.data.astype(np.float32)
    obs_pert = np.array(["NTC"] * 80 + list(np.repeat([f"P{i}" for i in range(16)], 20)))
    rng.shuffle(obs_pert)
    obs = pd.DataFrame({"perturbation": pd.Categorical(obs_pert)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    Xf = X.tocsr() if fmt == "csr" else X.tocsc()
    path = Path(dir_path) / f"{fmt}.h5ad"
    ad.AnnData(X=Xf, obs=obs, var=var).write_h5ad(path)
    return path


def test_masks_only_csc_matches_csr(tmp_path):
    """Masks-only QC (output_dir=None) must give identical results for CSC and CSR.

    Regression test for the CSC row-slicing performance fix: the masks-only
    path now uses column-oriented counting for CSC inputs, and must remain
    numerically identical to the CSR row-oriented path.
    """
    from crispyx.data import get_matrix_storage_format

    csr_p = _make_synthetic_h5ad(tmp_path, "csr")
    csc_p = _make_synthetic_h5ad(tmp_path, "csc")
    assert get_matrix_storage_format(csr_p) == "csr"
    assert get_matrix_storage_format(csc_p) == "csc"

    kw = dict(perturbation_column="perturbation", control_label="NTC",
              min_genes=3, min_cells_per_perturbation=10, min_cells_per_gene=5,
              output_dir=None)
    r_csr = quality_control_summary(csr_p, **kw)
    r_csc = quality_control_summary(csc_p, **kw)

    assert np.array_equal(r_csr.cell_mask, r_csc.cell_mask)
    assert np.array_equal(r_csr.gene_mask, r_csc.gene_mask)
    assert np.array_equal(r_csr.cell_gene_counts, r_csc.cell_gene_counts)
    assert np.array_equal(r_csr.gene_cell_counts, r_csc.gene_cell_counts)
    assert r_csr.perturbation_keep == r_csc.perturbation_keep


def test_iter_matrix_chunks_slow_axis_warns_once(tmp_path, caplog):
    """Streaming a backed CSC matrix by rows should warn exactly once."""
    import logging

    import crispyx.data as cxd
    from crispyx.data import iter_matrix_chunks, read_backed

    csc_p = _make_synthetic_h5ad(tmp_path, "csc")
    cxd._SLOW_AXIS_WARNED.clear()

    backed = read_backed(csc_p)
    try:
        with caplog.at_level(logging.WARNING, logger="crispyx.data"):
            for _ in iter_matrix_chunks(backed, axis=0, chunk_size=64, convert_to_dense=False):
                pass
            for _ in iter_matrix_chunks(backed, axis=0, chunk_size=64, convert_to_dense=False):
                pass
    finally:
        backed.file.close()

    slow_warnings = [r for r in caplog.records if "slower" in r.getMessage()]
    assert len(slow_warnings) == 1, f"expected exactly one slow-axis warning, got {len(slow_warnings)}"


def test_verbose_prefix_matches_current_function_and_namespace_names(tmp_path, capsys):
    """Regression test: quality_control_summary's print prefix must track its
    own name (and cx.pp.qc_summary's), not a name it was renamed from.

    quality_control_summary was previously named quality_control, and its
    verbose output said "[cx] qc.quality_control: ..." long after the
    rename -- three different names for one function. Guard against that
    drifting again.
    """
    quality_control_summary(
        _make_synthetic_h5ad(tmp_path, "csr"),
        perturbation_column="perturbation",
        control_label="NTC",
        min_genes=3,
        min_cells_per_perturbation=10,
        min_cells_per_gene=5,
        output_dir=tmp_path,
        data_name="verbose_prefix_test",
        verbose=1,
    )
    out = capsys.readouterr().out
    assert "[cx] pp.qc_summary:" in out
    assert "qc.quality_control" not in out
    assert "perturbations kept" in out


def _make_dataset_for_filtering(tmp_path, n=200, g=30, seed=0):
    """A dataset where roughly a third of cells/genes are near-empty, so a
    strict threshold drops a controllable majority."""

    rng = np.random.default_rng(seed)
    dense = rng.poisson(3, size=(n, g)).astype(np.float32)
    # Zero out most rows/columns entirely so a min_genes/min_cells threshold
    # of a handful reliably drops the majority.
    dense[: int(n * 0.8), :] = 0
    dense[:, : int(g * 0.8)] = 0
    obs = pd.DataFrame({
        "perturbation": pd.Categorical(
            ["NTC"] * (n // 2) + [f"P{i}" for i in range(n // 2)]
        )
    })
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    path = Path(tmp_path) / "filter_test.h5ad"
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var).write_h5ad(path)
    return path


class TestFilteringMessaging:
    def test_filter_cells_reports_kept_count(self, tmp_path, capsys):
        from crispyx.qc import filter_cells_by_gene_count

        path = _make_dataset_for_filtering(tmp_path)
        filter_cells_by_gene_count(path, min_genes=1)
        out = capsys.readouterr().out
        assert "[cx] pp.filter_cells: Done" in out
        assert "cells kept" in out

    def test_filter_cells_warns_when_most_dropped(self, tmp_path):
        from crispyx.qc import filter_cells_by_gene_count

        path = _make_dataset_for_filtering(tmp_path)
        with pytest.warns(UserWarning, match=r"pp\.filter_cells: only \d+/\d+ cells"):
            filter_cells_by_gene_count(path, min_genes=1)

    def test_filter_genes_warns_when_most_dropped(self, tmp_path):
        from crispyx.qc import filter_genes_by_cell_count

        path = _make_dataset_for_filtering(tmp_path)
        with pytest.warns(UserWarning, match=r"pp\.filter_genes: only \d+/\d+ genes"):
            filter_genes_by_cell_count(path, min_cells=1)

    def test_filter_perturbations_warns_when_most_dropped(self, tmp_path):
        from crispyx.qc import filter_perturbations_by_cell_count

        path = _make_dataset_for_filtering(tmp_path)
        with pytest.warns(UserWarning, match=r"pp\.filter_perturbations: only \d+/\d+ perturbations"):
            filter_perturbations_by_cell_count(
                path, perturbation_column="perturbation", control_label="NTC", min_cells=10_000,
            )

    def test_no_warning_when_most_pass(self, tmp_path):
        from crispyx.qc import filter_cells_by_gene_count

        path = _make_dataset_for_filtering(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            filter_cells_by_gene_count(path, min_genes=0)

    def test_verbose_false_silences_print_but_not_warning(self, tmp_path, capsys):
        from crispyx.qc import filter_cells_by_gene_count

        path = _make_dataset_for_filtering(tmp_path)
        with pytest.warns(UserWarning):
            filter_cells_by_gene_count(path, min_genes=1, verbose=False)
        assert capsys.readouterr().out == ""
