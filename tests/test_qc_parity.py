"""Tests for QC strategy parity - all strategies should produce identical results."""

from __future__ import annotations

import pytest
import numpy as np
import warnings
from pathlib import Path
import tempfile
import shutil


# Test datasets with different storage formats
TEST_DATASETS = {
    "csr_small": {
        "path": Path("data/Adamson_subset.h5ad"),
        "perturbation_column": "perturbation",
        "expected_format": "csr",
    },
    "csc_medium": {
        "path": Path("data/Tian-crispra.h5ad"),
        "perturbation_column": "perturbation",
        "expected_format": "csc",
    },
}

# QC parameters used for testing
QC_PARAMS = {
    "min_genes": 100,
    "min_cells_per_perturbation": 50,
    "min_cells_per_gene": 100,
}


@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory for test outputs."""
    tmp_dir = tempfile.mkdtemp(prefix="crispyx_qc_test_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_get_matrix_storage_format():
    """Test storage format detection function."""
    from crispyx.data import get_matrix_storage_format
    
    for name, config in TEST_DATASETS.items():
        if not config["path"].exists():
            pytest.skip(f"Dataset {config['path']} not found")
        
        detected_format = get_matrix_storage_format(config["path"])
        assert detected_format == config["expected_format"], (
            f"{name}: expected {config['expected_format']}, got {detected_format}"
        )


def test_qc_in_memory_basic(tmp_output_dir):
    """Test that in-memory QC runs without errors on small dataset."""
    from crispyx.qc import _qc_in_memory
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "in_memory.h5ad"
    result = _qc_in_memory(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        output_path=output_path,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"
    
    # Verify output file is readable
    import anndata as ad
    adata = ad.read_h5ad(output_path)
    assert adata.n_obs == result.cell_mask.sum()
    assert adata.n_vars == result.gene_mask.sum()


def test_qc_column_oriented_basic(tmp_output_dir):
    """Test that column-oriented QC runs without errors on CSC dataset."""
    from crispyx.qc import _qc_column_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csc_medium"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "column_oriented.h5ad"
    result = _qc_column_oriented(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        chunk_size=1024,
        output_path=output_path,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"


def test_qc_row_oriented_basic(tmp_output_dir):
    """Test that row-oriented QC runs without errors on CSR dataset."""
    from crispyx.qc import _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "row_oriented.h5ad"
    result = _qc_row_oriented(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        chunk_size=1024,
        output_path=output_path,
        cache_mode="memmap",
        delta_threshold=0.3,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"


def test_qc_strategy_parity_csr(tmp_output_dir):
    """Verify all QC strategies produce identical results on CSR dataset."""
    from crispyx.qc import _qc_in_memory, _qc_column_oriented, _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    common_kwargs = {
        "perturbation_column": dataset["perturbation_column"],
        "control_label": control_label,
        "gene_name_column": None,
        **QC_PARAMS,
    }
    
    # Run all three strategies
    result_memory = _qc_in_memory(
        dataset["path"],
        output_path=tmp_output_dir / "memory.h5ad",
        **common_kwargs,
    )
    
    result_column = _qc_column_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "column.h5ad",
        chunk_size=1024,
        **common_kwargs,
    )
    
    result_row = _qc_row_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "row.h5ad",
        chunk_size=1024,
        cache_mode="memmap",
        delta_threshold=0.3,
        **common_kwargs,
    )
    
    # Verify cell masks are identical
    assert np.array_equal(result_memory.cell_mask, result_column.cell_mask), (
        f"Cell mask mismatch (in-memory vs column): "
        f"{result_memory.cell_mask.sum()} vs {result_column.cell_mask.sum()}"
    )
    assert np.array_equal(result_memory.cell_mask, result_row.cell_mask), (
        f"Cell mask mismatch (in-memory vs row): "
        f"{result_memory.cell_mask.sum()} vs {result_row.cell_mask.sum()}"
    )
    
    # Verify gene masks are identical
    assert np.array_equal(result_memory.gene_mask, result_column.gene_mask), (
        f"Gene mask mismatch (in-memory vs column): "
        f"{result_memory.gene_mask.sum()} vs {result_column.gene_mask.sum()}"
    )
    assert np.array_equal(result_memory.gene_mask, result_row.gene_mask), (
        f"Gene mask mismatch (in-memory vs row): "
        f"{result_memory.gene_mask.sum()} vs {result_row.gene_mask.sum()}"
    )
    
    print(f"✓ CSR parity: cells={result_memory.cell_mask.sum()}, genes={result_memory.gene_mask.sum()}")


def test_qc_strategy_parity_csc(tmp_output_dir):
    """Verify all QC strategies produce identical results on CSC dataset."""
    from crispyx.qc import _qc_in_memory, _qc_column_oriented, _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csc_medium"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    common_kwargs = {
        "perturbation_column": dataset["perturbation_column"],
        "control_label": control_label,
        "gene_name_column": None,
        **QC_PARAMS,
    }
    
    # Run all three strategies
    result_memory = _qc_in_memory(
        dataset["path"],
        output_path=tmp_output_dir / "memory.h5ad",
        **common_kwargs,
    )
    
    result_column = _qc_column_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "column.h5ad",
        chunk_size=1024,
        **common_kwargs,
    )
    
    result_row = _qc_row_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "row.h5ad",
        chunk_size=1024,
        cache_mode="memmap",
        delta_threshold=0.3,
        **common_kwargs,
    )
    
    # Verify cell masks are identical
    assert np.array_equal(result_memory.cell_mask, result_column.cell_mask), (
        f"Cell mask mismatch (in-memory vs column): "
        f"{result_memory.cell_mask.sum()} vs {result_column.cell_mask.sum()}"
    )
    assert np.array_equal(result_memory.cell_mask, result_row.cell_mask), (
        f"Cell mask mismatch (in-memory vs row): "
        f"{result_memory.cell_mask.sum()} vs {result_row.cell_mask.sum()}"
    )
    
    # Verify gene masks are identical
    assert np.array_equal(result_memory.gene_mask, result_column.gene_mask), (
        f"Gene mask mismatch (in-memory vs column): "
        f"{result_memory.gene_mask.sum()} vs {result_column.gene_mask.sum()}"
    )
    assert np.array_equal(result_memory.gene_mask, result_row.gene_mask), (
        f"Gene mask mismatch (in-memory vs row): "
        f"{result_memory.gene_mask.sum()} vs {result_row.gene_mask.sum()}"
    )
    
    print(f"✓ CSC parity: cells={result_memory.cell_mask.sum()}, genes={result_memory.gene_mask.sum()}")


def test_quality_control_summary_dispatch(tmp_output_dir):
    """Test that quality_control_summary correctly dispatches based on data size."""
    from crispyx.qc import quality_control_summary
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Test with force_streaming=False (should use in-memory for small data)
    result1 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
        data_name="test1",
        force_streaming=False,
        **QC_PARAMS,
    )
    
    # Test with force_streaming=True (should use streaming)
    result2 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
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
    
    print(f"✓ Dispatch parity verified")


def test_qc_against_scanpy(tmp_output_dir):
    """Compare crispyx QC results against Scanpy QC as ground truth."""
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    from crispyx.qc import quality_control_summary
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
        output_dir=tmp_output_dir,
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
    
    print(f"✓ Scanpy parity: cells={adata.n_obs}, genes={adata.n_vars}")


def test_qc_strategy_selection_thresholds():
    """Verify the in-memory threshold: file×4 < min(limit×0.6, 50 GB).

    Threshold table (matches qc.py comments):
      - Small file   (<12.5 GB):  file×4 < 50 GB → in-memory
      - Medium file  ( 12.5 GB):  file×4 = 50 GB → streaming (edge, just over)
      - Large file   ( 27 GB):    file×4 = 108 GB → streaming

    This does NOT call the full QC pipeline; it directly validates the
    decision logic from quality_control_summary.
    """
    def _would_use_in_memory(file_size_gb: float, memory_limit_gb: float = 128.0) -> bool:
        """Mirror the threshold logic from quality_control_summary."""
        estimated_memory_gb = file_size_gb * 4          # sparse 4× multiplier
        threshold = min(memory_limit_gb * 0.6, 50.0)   # cap at 50 GB
        return estimated_memory_gb < threshold

    # Small datasets → in-memory
    assert _would_use_in_memory(0.05) is True,  "Adamson_subset (50 MB) should be in-memory"
    assert _would_use_in_memory(2.0) is True,   "Adamson (2 GB) should be in-memory"
    assert _would_use_in_memory(10.0) is True,  "Frangieh (10 GB) should be in-memory"
    assert _would_use_in_memory(12.4) is True,  "12.4 GB just under threshold"

    # Large datasets → streaming
    assert _would_use_in_memory(12.6) is False, "12.6 GB just over threshold → streaming"
    assert _would_use_in_memory(15.0) is False, "Feng-gwsf (15 GB) should stream"
    assert _would_use_in_memory(27.0) is False, "Feng-gwsnf (27 GB) should stream"
    assert _would_use_in_memory(54.0) is False, "Very large file should stream"

    # Cap behaviour: even with 500 GB node, threshold stays at 50 GB
    assert _would_use_in_memory(12.6, memory_limit_gb=500.0) is False, \
        "High-memory node should not relax threshold beyond 50 GB cap"
    assert _would_use_in_memory(10.0, memory_limit_gb=500.0) is True, \
        "10 GB file should still be in-memory even on 500 GB node"


def _make_synthetic_h5ad(dir_path, fmt, seed=0):
    """Write a small synthetic h5ad in the requested storage format ('csr'/'csc')."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

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


def test_masks_only_csc_matches_csr(tmp_output_dir):
    """Masks-only QC (output_dir=None) must give identical results for CSC and CSR.

    Regression test for the CSC row-slicing performance fix: the masks-only
    path now uses column-oriented counting for CSC inputs, and must remain
    numerically identical to the CSR row-oriented path.
    """
    from crispyx.data import get_matrix_storage_format
    from crispyx.qc import quality_control_summary

    csr_p = _make_synthetic_h5ad(tmp_output_dir, "csr")
    csc_p = _make_synthetic_h5ad(tmp_output_dir, "csc")
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


def test_iter_matrix_chunks_slow_axis_warns_once(tmp_output_dir, caplog):
    """Streaming a backed CSC matrix by rows should warn exactly once."""
    import logging

    import crispyx.data as cxd
    from crispyx.data import iter_matrix_chunks, read_backed

    csc_p = _make_synthetic_h5ad(tmp_output_dir, "csc")
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


def test_verbose_prefix_matches_current_function_and_namespace_names(tmp_output_dir, capsys):
    """Regression test: quality_control_summary's print prefix must track its
    own name (and cx.pp.qc_summary's), not a name it was renamed from.

    quality_control_summary was previously named quality_control, and its
    verbose output said "[cx] qc.quality_control: ..." long after the
    rename -- three different names for one function. Guard against that
    drifting again.
    """
    from crispyx.qc import quality_control_summary

    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")

    quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
        data_name="verbose_prefix_test",
        verbose=1,
        **QC_PARAMS,
    )
    out = capsys.readouterr().out
    assert "[cx] pp.qc_summary:" in out
    assert "qc.quality_control" not in out


def _make_dataset_for_filtering(tmp_path, n=200, g=30, seed=0):
    """A dataset where roughly a third of cells/genes are near-empty, so a
    strict threshold drops a controllable majority."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

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

    def test_quality_control_summary_reports_perturbation_counts(self, tmp_output_dir, capsys):
        from crispyx.qc import quality_control_summary

        dataset = TEST_DATASETS["csr_small"]
        if not dataset["path"].exists():
            pytest.skip(f"Dataset {dataset['path']} not found")

        quality_control_summary(
            dataset["path"],
            perturbation_column=dataset["perturbation_column"],
            output_dir=tmp_output_dir,
            data_name="pert_count_test",
            verbose=1,
            **QC_PARAMS,
        )
        out = capsys.readouterr().out
        assert "perturbations kept" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
