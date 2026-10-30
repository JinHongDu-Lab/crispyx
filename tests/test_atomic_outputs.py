"""An output file either holds a complete result or is what it was before.

Every writer puts its result in a hidden partial file beside the output and
renames it into place once the write completes and its sparse layout checks
out, so a run that fails or is killed midway never leaves a file a later call
could take for a result.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx import data as cxdata
from crispyx.data import _replace_on_success, check_sparse_layout


def _write_counts(path: Path, *, n_cells: int = 60, n_genes: int = 12, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    counts = sp.csr_matrix(rng.poisson(1.0, (n_cells, n_genes)).astype(np.float32))
    labels = np.array(["ctrl", "A", "B"] * (n_cells // 3))
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    ad.AnnData(counts, obs=obs, var=var).write(path)
    return path


def _partials(directory: Path) -> list[Path]:
    return sorted(directory.glob(".*.partial"))


# --- _replace_on_success ------------------------------------------------------


def test_partial_is_renamed_into_place_on_success(tmp_path):
    output = tmp_path / "out.h5ad"
    with _replace_on_success(output) as partial:
        assert partial.parent == output.parent
        assert partial.name.startswith(".out.h5ad.")
        assert f".{os.getpid()}-" in partial.name  # one name per writing process
        _write_counts(partial)
    assert output.exists()
    assert not _partials(tmp_path)


def test_failure_removes_the_partial_and_keeps_the_previous_output(tmp_path):
    output = _write_counts(tmp_path / "out.h5ad", seed=1)
    before = output.read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        with _replace_on_success(output) as partial:
            _write_counts(partial, seed=2)
            raise RuntimeError("injected")
    assert output.read_bytes() == before
    assert not _partials(tmp_path)


def test_truncated_sparse_write_is_refused_before_the_rename(tmp_path):
    output = _write_counts(tmp_path / "out.h5ad", seed=1)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="not written completely"):
        with _replace_on_success(output) as partial:
            _write_counts(partial, seed=2)
            with h5py.File(partial, "r+") as f:
                indptr = f["X/indptr"][...]
                del f["X/indptr"]
                f["X"].create_dataset("indptr", data=indptr[:-5])  # stopped short
    assert output.read_bytes() == before
    assert not _partials(tmp_path)


# --- check_sparse_layout ------------------------------------------------------


def test_check_sparse_layout_passes_complete_csr_csc_and_dense(tmp_path):
    csr = _write_counts(tmp_path / "csr.h5ad")
    check_sparse_layout(csr)
    a = ad.read_h5ad(csr)
    a.X = sp.csc_matrix(a.X)
    a.write(tmp_path / "csc.h5ad")
    check_sparse_layout(tmp_path / "csc.h5ad")
    a.X = a.X.toarray()
    a.write(tmp_path / "dense.h5ad")
    check_sparse_layout(tmp_path / "dense.h5ad")
    ad.AnnData(obs=a.obs, var=a.var).write(tmp_path / "no_x.h5ad")
    check_sparse_layout(tmp_path / "no_x.h5ad")


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda ip, nnz: ip[:-1], "entries, expected"),
        (lambda ip, nnz: ip + 1, "starts at 1"),
        (lambda ip, nnz: np.concatenate([ip[:3], ip[1:2], ip[4:]]), "decreases"),
        (lambda ip, nnz: np.concatenate([ip[:-1], [nnz - 1]]), "but data has"),
    ],
)
def test_check_sparse_layout_names_the_inconsistency(tmp_path, corrupt, message):
    path = _write_counts(tmp_path / "x.h5ad")
    with h5py.File(path, "r+") as f:
        indptr = f["X/indptr"][...].astype(np.int64)
        nnz = f["X/data"].shape[0]
        del f["X/indptr"]
        f["X"].create_dataset("indptr", data=corrupt(indptr, nnz))
    with pytest.raises(ValueError, match=message):
        check_sparse_layout(path)


# --- the streaming writers that used to fill their output in place ------------


class _FailingEvict:
    """Stands in for data._EvictEvery: raises on the first write of the
    streamed matrix, i.e. after the placeholder file exists."""

    def __init__(self, *paths):
        pass

    def __call__(self, n_bytes):
        raise RuntimeError("injected")


@pytest.mark.parametrize(
    "run",
    [
        lambda src, out: cx.pp.normalize_total_log1p(src, output_path=out, verbose=False),
        lambda src, out: cx.pp.downsample_counts(src, counts_per_cell=5, output_path=out, verbose=False),
    ],
    ids=["normalize_total_log1p", "downsample_counts"],
)
def test_streaming_writer_killed_midway_leaves_previous_output_intact(tmp_path, monkeypatch, run):
    src = _write_counts(tmp_path / "counts.h5ad")
    output = _write_counts(tmp_path / "out.h5ad", seed=3)
    before = output.read_bytes()
    monkeypatch.setattr(cxdata, "_EvictEvery", _FailingEvict)
    with pytest.raises(RuntimeError, match="injected"):
        run(src, output)
    assert output.read_bytes() == before
    assert not _partials(tmp_path)
    monkeypatch.undo()
    run(src, output)  # the same call then succeeds and replaces the old output
    assert output.read_bytes() != before
    assert not _partials(tmp_path)


def test_qc_summary_and_normalized_effects_write_through_a_partial(tmp_path):
    src = _write_counts(tmp_path / "counts.h5ad")
    qc_out = tmp_path / "qc.h5ad"
    cx.pp.qc_summary(
        src, perturbation_column="perturbation", control_label="ctrl",
        min_genes=1, min_cells_per_perturbation=1, min_cells_per_gene=1,
        output_path=qc_out, verbose=False,
    )
    assert qc_out.exists()
    fx_out = tmp_path / "effects.h5ad"
    cx.pb.normalized_effects(
        src, perturbation_column="perturbation", control_label="ctrl",
        output_path=fx_out, verbose=False,
    )
    assert fx_out.exists()
    assert not _partials(tmp_path)
