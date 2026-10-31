"""crispyx.data.convert_to_csc / convert_to_csr, tested symmetrically."""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
import crispyx.data as cxd
from crispyx.data import (
    _contiguous_bands,
    _conversion_buffer_budget_bytes,
    convert_to_csc,
    convert_to_csr,
    get_matrix_storage_format,
)

CONVERT = {"csc": convert_to_csc, "csr": convert_to_csr}
OTHER = {"csc": "csr", "csr": "csc"}
targets = pytest.mark.parametrize("target", ["csc", "csr"])


def _dense(dtype=np.float64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return ((rng.random((50, 30)) < 0.3) * rng.integers(1, 9, (50, 30))).astype(dtype)


def _write(path: Path, dense: np.ndarray, fmt: str) -> Path:
    """Write ``dense`` as an h5ad whose X is stored as ``fmt`` ('csr', 'csc' or 'dense')."""
    n_obs, n_vars = dense.shape
    obs = pd.DataFrame(
        {"perturbation": np.resize(["ctrl", "A", "B"], n_obs)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"gene_symbols": [f"G{j}" for j in range(n_vars)]}, index=[f"g{j}" for j in range(n_vars)])
    X = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[fmt](dense)
    ad.AnnData(X, obs=obs, var=var).write(path)
    return path


def _assert_canonical(path: Path, dense: np.ndarray, fmt: str) -> None:
    """The stored arrays equal scipy's canonical (sorted-index) matrix, byte for byte."""
    ref = sp.csc_matrix(dense) if fmt == "csc" else sp.csr_matrix(dense)
    ref.sort_indices()
    with h5py.File(path, "r") as f:
        encoding = f["X"].attrs["encoding-type"]
        assert (encoding.decode() if isinstance(encoding, bytes) else encoding) == f"{fmt}_matrix"
        assert list(f["X"].attrs["shape"]) == list(dense.shape)
        np.testing.assert_array_equal(f["X/indptr"][:], ref.indptr)
        np.testing.assert_array_equal(f["X/indices"][:], ref.indices)
        np.testing.assert_array_equal(f["X/data"][:], ref.data)
        assert f["X/data"].dtype == dense.dtype
    assert get_matrix_storage_format(path) == fmt


@targets
@pytest.mark.parametrize("source_fmt", ["other", "dense"])
def test_conversion_is_exact_and_keeps_metadata(tmp_path, target, source_fmt):
    dense = _dense()
    fmt = OTHER[target] if source_fmt == "other" else "dense"
    src = _write(tmp_path / "src.h5ad", dense, fmt)
    assert get_matrix_storage_format(src) == fmt
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)
    original, converted = ad.read_h5ad(src), ad.read_h5ad(out)
    pd.testing.assert_frame_equal(converted.obs, original.obs)
    pd.testing.assert_frame_equal(converted.var, original.var)


@targets
def test_chunking_and_memory_bands_do_not_change_the_file(tmp_path, target):
    """chunk_size=1 and a tiny memory budget (several bands) give the same file."""
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    nnz_per_band_axis = np.count_nonzero(dense, axis=0 if target == "csc" else 1)
    assert len(_contiguous_bands(nnz_per_band_axis, per_item_bytes=12,
                                 budget_bytes=_conversion_buffer_budget_bytes(1e-6))) > 1
    for name, kw in [("one", dict(chunk_size=1)), ("seven", dict(chunk_size=7)),
                     ("banded", dict(chunk_size=7, memory_limit_gb=1e-6))]:
        out = tmp_path / f"{name}.h5ad"
        CONVERT[target](src, output_path=out, verbose=False, **kw).close()
        _assert_canonical(out, dense, target)


@targets
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_conversion_preserves_value_dtype(tmp_path, target, dtype):
    dense = _dense(dtype)
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
@pytest.mark.parametrize("source_fmt", ["other", "dense"])
def test_conversion_evicts_its_output_as_it_writes(tmp_path, monkeypatch, target, source_fmt):
    """The converted file's writes are reported for page-cache eviction in
    slices, so a large output is not left in a job's page cache."""
    reported = []

    class Recorder:
        def __init__(self, *paths, nbytes=0):
            self.names = [Path(p).name for p in paths]

        def __call__(self, moved):
            reported.extend((name, moved) for name in self.names)

    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target] if source_fmt == "other" else "dense")
    monkeypatch.setattr(cxd, "_EvictEvery", Recorder)
    monkeypatch.setattr(cxd, "_BAND_WRITE_BYTES", 64)
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, chunk_size=7, memory_limit_gb=1e-6, verbose=False).close()
    _assert_canonical(out, dense, target)
    written = [
        moved for name, moved in reported
        if name.startswith(".out.h5ad.") and name.endswith(".partial")
    ]
    assert len(written) > 1
    assert sum(written) == np.count_nonzero(dense) * (dense.itemsize + 4)


@targets
def test_source_already_in_target_format_is_returned(tmp_path, target):
    src = _write(tmp_path / "src.h5ad", _dense(), target)
    out = tmp_path / "should_not_exist.h5ad"
    result = CONVERT[target](src, output_path=out, verbose=False)
    assert not out.exists()
    assert Path(result.filename).resolve() == src.resolve()
    result.file.close()


@targets
def test_default_output_path_is_next_to_the_source(tmp_path, target):
    src = _write(tmp_path / "screen.h5ad", _dense(), OTHER[target])
    CONVERT[target](src, verbose=False).close()
    assert get_matrix_storage_format(tmp_path / f"screen_cx_{target}.h5ad") == target


@targets
def test_empty_matrix(tmp_path, target):
    dense = np.zeros((3, 2), dtype=np.float32)
    src = _write(tmp_path / "empty.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
def test_warns_when_disk_space_low(tmp_path, target, monkeypatch):
    """A near-full destination filesystem warns but does not block the conversion."""
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(total=1000, used=999, free=1))
    with pytest.warns(UserWarning, match=f"pp.convert_to_{target}"):
        CONVERT[target](src, output_path=out, verbose=False).close()
    _assert_canonical(out, dense, target)


@targets
def test_interrupted_conversion_leaves_no_output_file(tmp_path, target, monkeypatch):
    """A conversion killed while filling the pre-sized output must not leave a
    structurally valid file with zero-filled bands at output_path."""
    src = _write(tmp_path / "src.h5ad", _dense(), OTHER[target])
    out = tmp_path / "out.h5ad"

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cxd, "_scatter_band", boom)
    with pytest.raises(KeyboardInterrupt):
        CONVERT[target](src, output_path=out, verbose=False)
    assert not out.exists()
    assert list(tmp_path.glob(".*partial")) == []


@targets
def test_pp_namespace_forwards_to_the_function(tmp_path, target):
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = tmp_path / "out.h5ad"
    getattr(cx.pp, f"convert_to_{target}")(src, output_path=out, memory_limit_gb=1e-6, verbose=False).close()
    _assert_canonical(out, dense, target)


def test_contiguous_bands_respects_budget_and_covers_everything():
    counts = np.array([5, 1, 9, 2, 2, 2, 40, 1])
    bands = _contiguous_bands(counts, per_item_bytes=1, budget_bytes=10)
    assert bands[0][0] == 0 and bands[-1][1] == len(counts)
    assert all(b[1] == nb[0] for b, nb in zip(bands, bands[1:]))  # contiguous
    for start, stop in bands:
        # Within budget unless a single index alone exceeds it (index 6: 40).
        assert counts[start:stop].sum() <= 10 or stop - start == 1
    assert (6, 7) in bands
    assert _contiguous_bands(counts, per_item_bytes=1, budget_bytes=1000) == [(0, len(counts))]
    assert _contiguous_bands(np.zeros(0, dtype=np.int64), per_item_bytes=1, budget_bytes=1) == [(0, 0)]


@targets
@pytest.mark.parametrize("spill", ["tmp", "beside_output", "none"])
def test_banded_conversion_reads_once_or_per_band_to_the_same_bytes(tmp_path, monkeypatch, target, spill):
    """Several bands spill to the temporary directory (or beside the output)
    and read the source once; with no room to spill, each band rereads it.
    The files are identical and no spill files are left behind."""
    import tempfile

    scratch_root = tmp_path / "tmp"
    scratch_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    chosen = {"tmp": scratch_root, "beside_output": out_dir, "none": None}[spill]
    monkeypatch.setattr(cxd, "_page_cache_is_charged", lambda: True)  # as in a Slurm job
    monkeypatch.setattr(cxd, "_spill_dir", lambda *args: chosen)
    monkeypatch.setattr(cxd, "_SPILL_READ_ITEMS", 5)  # several reads per band
    reads = []
    real_chunks = cxd._source_chunks
    monkeypatch.setattr(cxd, "_source_chunks", lambda *a: (reads.append(1), real_chunks(*a))[1])
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    out = out_dir / "out.h5ad"
    CONVERT[target](src, output_path=out, chunk_size=7, memory_limit_gb=1e-6, verbose=False).close()
    _assert_canonical(out, dense, target)
    n_bands = len(_contiguous_bands(np.count_nonzero(dense, axis=0 if target == "csc" else 1),
                                    per_item_bytes=12, budget_bytes=_conversion_buffer_budget_bytes(1e-6)))
    assert len(reads) == (n_bands if spill == "none" else 1)
    assert list(scratch_root.iterdir()) == []
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.h5ad"]


@targets
def test_spill_files_are_removed_when_a_conversion_fails(tmp_path, monkeypatch, target):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(cxd, "_page_cache_is_charged", lambda: True)
    monkeypatch.setattr(cxd, "_spill_dir", lambda *args: tmp_path / "tmp")

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cxd, "_write_band", boom)  # after the spill files are written
    src = _write(tmp_path / "src.h5ad", _dense(), OTHER[target])
    with pytest.raises(KeyboardInterrupt):
        CONVERT[target](src, output_path=tmp_path / "out.h5ad", memory_limit_gb=1e-6, verbose=False)
    assert list((tmp_path / "tmp").iterdir()) == []


def test_spill_goes_where_there_is_room(tmp_path, monkeypatch):
    """The temporary directory first, else the output's directory (which must
    also hold the output still to be written), else nowhere; never a
    RAM-backed directory, nor one on the source's filesystem where writing
    and reading back the spill costs more than re-reading the source."""
    import tempfile

    tmp, out_dir = tmp_path / "tmp", tmp_path / "out"
    tmp.mkdir()
    out_dir.mkdir()
    source = tmp_path / "src.h5ad"
    source.touch()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    free = {}
    monkeypatch.setattr(cxd, "_free_bytes_at", lambda p: free[Path(p)])
    output = out_dir / "x.h5ad"
    free.update({tmp: 2000.0, out_dir: 2000.0})
    assert cxd._spill_dir(1000, output, 500, source, 1e9) == tmp
    # tmp and out share a filesystem here, so each needs 1.1 x spill + output = 1600
    free.update({tmp: 1500.0, out_dir: 1500.0})
    assert cxd._spill_dir(1000, output, 500, source, 1e9) is None
    free.update({out_dir: 1700.0})
    assert cxd._spill_dir(1000, output, 500, source, 1e9) == out_dir
    free.update({tmp: None, out_dir: None})  # unknown free space: do not spill
    assert cxd._spill_dir(1000, output, 500, source, 1e9) is None
    free.update({tmp: 2000.0, out_dir: 2000.0})
    # Same filesystem as the source: 2 x 1000 B of spill I/O vs 1500 B of re-reads.
    assert cxd._spill_dir(1000, output, 500, source, 1500) is None
    monkeypatch.setattr(cxd, "_is_ram_backed", lambda p: Path(p) == tmp)
    assert cxd._spill_dir(1000, output, 500, source, 1e9) == out_dir


def test_ram_backed_filesystems_are_read_from_the_mount_table(tmp_path, monkeypatch):
    import builtins
    import io

    from crispyx import _memory

    table = f"/dev/sda1 / ext4 rw 0 0\ntmpfs {tmp_path} tmpfs rw 0 0\n"
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: io.StringIO(table)
                        if path == "/proc/self/mounts" else real_open(path, *a, **k))
    monkeypatch.setattr(_memory.os.path, "realpath", lambda p: str(p))
    assert _memory._is_ram_backed(tmp_path / "x")
    assert not _memory._is_ram_backed(Path(str(tmp_path) + "-sibling"))
    assert not _memory._is_ram_backed(Path("/home/x"))


def test_spilling_reserves_one_open_file_per_band(monkeypatch):
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert cxd._reserve_open_files(10)
    if hard != resource.RLIM_INFINITY:
        assert not cxd._reserve_open_files(hard + 1)
    assert resource.getrlimit(resource.RLIMIT_NOFILE)[1] == hard


@targets
def test_bands_reread_the_source_where_page_cache_is_free(tmp_path, monkeypatch, target):
    """Without a memory limit that charges page cache, a band's re-read is
    served from cache and spilling would only add disk traffic."""
    monkeypatch.setattr(cxd, "_page_cache_is_charged", lambda: False)
    monkeypatch.setattr(cxd, "_spill_dir", lambda *args: pytest.fail("spilled"))
    dense = _dense()
    src = _write(tmp_path / "src.h5ad", dense, OTHER[target])
    CONVERT[target](src, output_path=tmp_path / "out.h5ad", memory_limit_gb=1e-6, verbose=False).close()
    _assert_canonical(tmp_path / "out.h5ad", dense, target)
