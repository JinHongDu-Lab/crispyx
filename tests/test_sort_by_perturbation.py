"""sort_by_perturbation writes the source's rows in perturbation order."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx.data
from crispyx.data import perturbation_sort_order, sort_by_perturbation


def _screen(tmp_path, fmt: str):
    rng = np.random.default_rng(3)
    n_cells, n_genes = 500, 40
    labels = rng.choice(["ctrl", "A", "B", "C", "D"], n_cells)
    counts = sp.random(n_cells, n_genes, density=0.2, format="csr", random_state=4, dtype=np.float32)
    counts.data = np.ceil(counts.data * 9)
    counts[7] = 0  # an empty row
    counts.eliminate_zeros()
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    path = tmp_path / f"{fmt}.h5ad"
    matrix = counts.toarray() if fmt == "dense" else counts.asformat(fmt)
    ad.AnnData(matrix, obs=obs, var=var).write_h5ad(path)
    return path, counts, labels


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
@pytest.mark.parametrize("bands", ["one", "several", "spilled"])
def test_sorted_rows_match_the_reference(tmp_path, monkeypatch, fmt, bands):
    path, counts, labels = _screen(tmp_path, fmt)
    if bands != "one":  # several row bands: one pass over the source each, or one pass spilling the rest
        monkeypatch.setattr(crispyx.data, "_conversion_buffer_budget_bytes", lambda _limit: 2_000)
    if bands == "spilled":
        monkeypatch.setattr(crispyx.data, "_page_cache_is_charged", lambda: True)  # as in a Slurm job
        monkeypatch.setattr(crispyx.data, "_spill_dir", lambda *args: tmp_path)
        passes = []
        real = crispyx.data._source_chunks
        monkeypatch.setattr(crispyx.data, "_source_chunks",
                            lambda *args: (passes.append(args), real(*args))[1])
    out = sort_by_perturbation(path, "perturbation", "ctrl", output_path=tmp_path / "sorted.h5ad",
                               chunk_size=64, verbose=False)
    order = perturbation_sort_order(labels, "ctrl")
    result = ad.read_h5ad(out)
    if fmt == "dense":
        assert isinstance(result.X, np.ndarray)
    else:
        assert sp.isspmatrix_csr(result.X) or isinstance(result.X, sp.csr_array)
        assert result.X.has_sorted_indices
    if bands == "spilled" and fmt == "csr":
        assert len(passes) == 1                       # the source read once
    got = result.X if isinstance(result.X, np.ndarray) else result.X.toarray()
    np.testing.assert_array_equal(got, counts[order].toarray())
    assert list(result.obs_names) == [f"c{i}" for i in order]
    assert result.uns["sorting_metadata"]["sorted_by"] == "perturbation"


def test_contiguous_ranges_keeps_only_single_runs():
    from crispyx._grouping import contiguous_ranges

    assert contiguous_ranges(np.array([], dtype=str)) == {}
    assert contiguous_ranges(np.array(["c", "c", "a", "b", "b"])) == {
        "c": slice(0, 2), "a": slice(2, 3), "b": slice(3, 5)}
    # "a" split over two runs is left out; the rest are still ranges.
    assert contiguous_ranges(np.array(["a", "b", "a", "c"])) == {"b": slice(1, 2), "c": slice(3, 4)}


def test_sorted_file_stores_no_per_label_map(tmp_path):
    """One HDF5 dataset per label in uns made every backed open of a
    genome-wide sorted file take over a second; the ranges come from obs."""
    import h5py

    path, _, _ = _screen(tmp_path, "csr")
    out = sort_by_perturbation(path, "perturbation", "ctrl", output_path=tmp_path / "sorted.h5ad", verbose=False)
    with h5py.File(out, "r") as handle:
        assert set(handle["uns/sorting_metadata"]) == {"original_path", "sorted_by", "control_label", "timestamp"}


def test_sorted_copy_of_an_older_layout_is_rewritten(tmp_path):
    import h5py

    path, _, _ = _screen(tmp_path, "csr")
    out = sort_by_perturbation(path, "perturbation", "ctrl", output_path=tmp_path / "sorted.h5ad", verbose=False)
    with h5py.File(out, "r+") as handle:  # as a 0.2.6 copy: sorted layout 1
        handle["uns/crispyx/schema"][()] = 1
    before = out.stat().st_mtime_ns
    sort_by_perturbation(path, "perturbation", "ctrl", output_path=out, verbose=False)
    assert out.stat().st_mtime_ns != before


def test_chunk_starts_follow_the_selected_runs():
    from crispyx.data import chunk_starts

    assert chunk_starts(10, 4, selected=np.ones(10, dtype=bool)) == [0, 4, 8]  # all selected: the grid
    selected = np.zeros(20, dtype=bool)
    selected[[2, 3, 4, 5, 6, 15, 16]] = True
    assert chunk_starts(20, 4, selected=selected) == [2, 6, 15]  # gaps skipped, chunks at run starts
    assert chunk_starts(20, 4, selected=np.zeros(20, dtype=bool)) == []


def _log_screen(tmp_path):
    rng = np.random.default_rng(5)
    labels = rng.choice(["ctrl"] * 4 + [f"G{i}" for i in range(12)], 600)
    counts = rng.poisson(3, (labels.size, 15)).astype(np.float32)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(labels.size)])
    path = tmp_path / "scattered.h5ad"
    ad.AnnData(sp.csr_matrix(np.log1p(counts)), obs=obs).write_h5ad(path)
    return path, labels


def test_t_test_on_a_subset_reads_only_its_rows_of_a_sorted_file(tmp_path, monkeypatch):
    import crispyx as cx
    import crispyx.de

    path, labels = _log_screen(tmp_path)
    sorted_path = sort_by_perturbation(path, "perturbation", "ctrl", output_path=tmp_path / "s.h5ad", verbose=False)
    real = crispyx.de._add_group_moments
    rows_read = []
    monkeypatch.setattr(crispyx.de, "_add_group_moments",
                        lambda block, *args: (rows_read.append(block.shape[0]), real(block, *args))[1])
    kw = dict(perturbation_column="perturbation", control_label="ctrl", perturbations=["G3", "G7"],
              cell_chunk_size=32, verbose=False, min_pct_ctrl=0.0, min_pct_pert=0.0, min_mean_ctrl=0.0,
              min_mean_pert=0.0)
    on_sorted = cx.t_test(sorted_path, output_path=tmp_path / "t_sorted.h5ad", **kw)
    wanted = int(np.isin(labels, ["ctrl", "G3", "G7"]).sum())
    assert wanted <= sum(rows_read) < wanted + 3 * 32  # each range's last chunk may run past it
    rows_read.clear()
    on_scattered = cx.t_test(path, output_path=tmp_path / "t_scattered.h5ad", **kw)
    full = cx.t_test(path, output_path=tmp_path / "t_full.h5ad", **{**kw, "perturbations": None})
    for label in ("G3", "G7"):
        for result in (on_scattered, full):
            np.testing.assert_allclose(on_sorted[label].statistic, result[label].statistic, rtol=1e-10)
            np.testing.assert_allclose(on_sorted[label].pvalue, result[label].pvalue, rtol=1e-10)
