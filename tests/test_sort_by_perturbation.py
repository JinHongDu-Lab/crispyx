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

