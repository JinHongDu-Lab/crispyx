"""Temporary files must not outlive a call that fails partway."""

from __future__ import annotations

from pathlib import Path

import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx.data
from crispyx import qc
from crispyx.data import sort_by_perturbation


def _write_counts(path: Path, *, n_cells: int = 60, n_genes: int = 12) -> Path:
    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(rng.poisson(1.0, (n_cells, n_genes)).astype(np.float32))
    labels = np.array(["ctrl", "A", "B"] * (n_cells // 3))
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    ad.AnnData(counts, obs=obs, var=var).write(path)
    return path


def test_qc_gene_filter_cache_is_removed_when_filling_it_fails(tmp_path, monkeypatch):
    path = _write_counts(tmp_path / "counts.h5ad")
    cache_root = tmp_path / "tmpdir"
    cache_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(cache_root))

    def boom(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(qc, "_ensure_csr", boom)
    with pytest.raises(RuntimeError, match="injected"):
        qc._filter_genes_with_cache(
            path,
            min_cells=1,
            cell_mask=np.ones(60, dtype=bool),
            gene_cell_counts=np.full(12, 10),
            output_path=tmp_path / "out.h5ad",
            cache_mode="disk",
            verbose=False,
        )
    assert not list(cache_root.glob("crispyx_qc_cache_*"))


def test_a_failed_sort_leaves_no_file(tmp_path, monkeypatch):
    path = _write_counts(tmp_path / "counts.h5ad")
    output = tmp_path / "sorted.h5ad"

    def failing_write(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(crispyx.data, "_write_band", failing_write)
    with pytest.raises(RuntimeError, match="injected"):
        sort_by_perturbation(path, "perturbation", "ctrl", output_path=output, force=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["counts.h5ad"]