"""Tests for crispyx.compress_h5ad and the shuffle+deflate codec behind it."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx import _h5codec
from crispyx.data import compress_h5ad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_cell(n_obs: int = 3000, n_vars: int = 400, *, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(0.3, 1.0, (n_obs, n_vars))).astype(np.float32)
    x = sp.csr_matrix(np.log1p(counts * 1e4 / np.maximum(counts.sum(1, keepdims=True), 1)))
    obs = pd.DataFrame(
        {
            "perturbation": pd.Categorical(rng.choice(["ctrl", "A", "B", "C"], n_obs)),
            "n_counts": counts.sum(1),
            "flag": rng.random(n_obs) > 0.5,
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"gene_symbols": [f"G{i}" for i in range(n_vars)]}, index=[f"g{i}" for i in range(n_vars)])
    adata = ad.AnnData(x, obs=obs, var=var)
    adata.layers["counts"] = sp.csr_matrix(counts)
    adata.obsm["X_pca"] = rng.standard_normal((n_obs, 20)).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_vars, 20))
    adata.obsp["connectivities"] = sp.random(n_obs, n_obs, density=0.001, format="csr", random_state=1)
    adata.uns["nested"] = {"a": np.arange(5000), "b": {"c": "text", "d": np.array(["x", "y"])}}
    return adata


def _write_exotic(path: Path) -> None:
    """Every HDF5 feature a future .h5ad could plausibly carry."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["empty_attr"] = h5py.Empty("f4")
        f.attrs["fixed_bytes"] = np.bytes_("csr_matrix")
        f.attrs["array_attr"] = np.arange(3, dtype=np.int16)
        g = f.create_group("things")
        g.create_dataset("float64", data=rng.standard_normal((700, 90)))
        g.create_dataset("float16", data=rng.standard_normal(5000).astype(np.float16))
        g.create_dataset("big_endian", data=rng.standard_normal(5000).astype(">f4"))
        g.create_dataset("bool", data=rng.random(5000) > 0.5)
        g.create_dataset("complex", data=rng.standard_normal(3000) + 1j * rng.standard_normal(3000))
        g.create_dataset("uint16", data=rng.integers(0, 60000, 5000, dtype=np.uint16))
        g.create_dataset("int8_3d", data=rng.integers(-5, 5, (30, 20, 10), dtype=np.int8))
        nan_signed = np.array([np.nan, -0.0, 0.0, np.inf, -np.inf] * 400)
        g.create_dataset("specials", data=nan_signed)
        g.create_dataset("fixed_str", data=np.array([f"gene_{i}".encode() for i in range(3000)], dtype="S12"))
        g.create_dataset(
            "vlen_str", data=np.array([f"cell_{i}" for i in range(3000)], dtype=object),
            dtype=h5py.string_dtype(),
        )
        rec = np.zeros(2000, dtype=[("names", "S8"), ("scores", "<f4"), ("pad", "<f8")])
        rec["scores"] = rng.standard_normal(2000)
        g.create_dataset("recarray", data=rec)
        g.create_dataset(
            "resizable", data=np.arange(4000, dtype=np.int64), maxshape=(None,), chunks=(512,),
        )
        g.create_dataset("lzf_source", data=rng.standard_normal(20000), compression="lzf")
        g.create_dataset("scalar", data=3.5)
        g.create_dataset("empty", shape=(0,), dtype=np.float32)
        g.create_dataset("tiny", data=np.arange(4))
        g.create_dataset("h5empty", data=h5py.Empty("i8"))
        # A row wider than one chunk forces chunks that split trailing axes,
        # with edge chunks on both axes.
        g.create_dataset("wide", data=rng.standard_normal((3, 140_001)))
        g["soft"] = h5py.SoftLink("/things/float64")
        g["external"] = h5py.ExternalLink("elsewhere.h5", "/data")
        g["float64"].attrs["note"] = "keeps attrs"


def _datasets(path: Path) -> dict[str, h5py.Dataset]:
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = (obj.compression, obj.shuffle, obj.chunks)
        f.visititems(visit)
    return out


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_])
def test_direct_chunks_decode_through_hdf5s_own_filter(tmp_path, dtype):
    """Chunks encoded here must be exactly what HDF5's shuffle+deflate reads."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal(700_000) * 100).astype(dtype)
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.create_dataset("d", data=data)
    compress_h5ad(src, tmp_path / "dst.h5", verify=False)
    with h5py.File(tmp_path / "dst.h5", "r") as f:
        ds = f["d"]
        assert ds.compression == "gzip" and ds.shuffle
        assert np.array_equal(ds[:], data)
        # and our decoder agrees with HDF5 on a raw chunk
        _, blob = ds.id.read_direct_chunk((0,))
        n = ds.chunks[0]
        assert np.array_equal(_h5codec.decode_chunk(blob, ds.dtype, (n,)), data[:n])


def test_chunk_shape_targets_one_mib():
    assert _h5codec.chunk_shape((10**8,), 4) == (2**18,)
    assert _h5codec.chunk_shape((5000, 8000), 8) == (16, 8000)
    assert _h5codec.chunk_shape((3, 10**6), 8) == (1, 2**17)
    assert _h5codec.chunk_shape((10,), 4) == (10,)


# ---------------------------------------------------------------------------
# compress_h5ad
# ---------------------------------------------------------------------------


def test_anndata_round_trip_is_exact(tmp_path):
    adata = _single_cell()
    src, dst = tmp_path / "sc.h5ad", tmp_path / "sc.gz.h5ad"
    adata.write(src)
    result = compress_h5ad(src, dst)

    assert result["verified"] and result["dst_bytes"] < result["src_bytes"]
    a, b = ad.read_h5ad(src), ad.read_h5ad(dst)
    assert (a.X != b.X).nnz == 0 and a.X.dtype == b.X.dtype
    assert a.X.indices.dtype == b.X.indices.dtype and a.X.indptr.dtype == b.X.indptr.dtype
    assert (a.layers["counts"] != b.layers["counts"]).nnz == 0
    pd.testing.assert_frame_equal(a.obs, b.obs)
    pd.testing.assert_frame_equal(a.var, b.var)
    assert np.array_equal(a.obsm["X_pca"], b.obsm["X_pca"])
    assert np.array_equal(a.varm["PCs"], b.varm["PCs"])
    assert (a.obsp["connectivities"] != b.obsp["connectivities"]).nnz == 0
    assert np.array_equal(a.uns["nested"]["a"], b.uns["nested"]["a"])
    assert b.uns["nested"]["b"]["c"] == "text"

    backed = ad.read_h5ad(dst, backed="r")
    assert (backed.X[100:200] != a.X[100:200]).nnz == 0
    backed.file.close()

    layout = _datasets(dst)
    assert layout["X/data"][:2] == ("gzip", True)
    assert layout["X/indices"][:2] == ("gzip", True)


def test_every_hdf5_feature_survives(tmp_path):
    src, dst = tmp_path / "exotic.h5", tmp_path / "exotic.gz.h5"
    _write_exotic(src)
    compress_h5ad(src, dst)  # verify=True compares every byte and attribute

    with h5py.File(src, "r") as fs, h5py.File(dst, "r") as fd:
        assert fd["/"].id.get_create_plist().get_link_creation_order()  # track_order kept
        assert list(fs["things"]) == list(fd["things"])
        assert isinstance(fd["things"].get("soft", getlink=True), h5py.SoftLink)
        assert isinstance(fd["things"].get("external", getlink=True), h5py.ExternalLink)
        assert fd["things/resizable"].maxshape == (None,)
        assert fd["things/wide"].chunks == (1, 2**17)
        assert fd["things/big_endian"].dtype == np.dtype(">f4")
        assert fd["things/float64"].attrs["note"] == "keeps attrs"
        assert np.array_equal(fs["things/specials"][:], fd["things/specials"][:], equal_nan=True)
        assert fd["things/vlen_str"].compression is None  # vlen copied as is
        assert fd["things/fixed_str"].compression == "gzip"
        assert fd["things/recarray"].compression == "gzip"
        assert fd["things/lzf_source"].compression == "gzip"  # re-encoded
        assert fd["things/tiny"].compression is None


def test_crispyx_streaming_matches_on_compressed_input(tmp_path):
    src, dst = tmp_path / "sc.h5ad", tmp_path / "sc.gz.h5ad"
    _single_cell().write(src)
    compress_h5ad(src, dst)
    raw = cx.aggregate_pseudobulk(src, groupby="perturbation", output_dir=tmp_path / "raw", verbose=False)
    gz = cx.aggregate_pseudobulk(dst, groupby="perturbation", output_dir=tmp_path / "gz", verbose=False)
    assert np.array_equal(np.asarray(raw.to_memory().X), np.asarray(gz.to_memory().X))
    pd.testing.assert_frame_equal(cx.load_obs(src), cx.load_obs(dst))


def test_output_is_independent_of_thread_count(tmp_path):
    src = tmp_path / "sc.h5ad"
    _single_cell(n_obs=1500).write(src)
    compress_h5ad(src, tmp_path / "one.h5ad", n_jobs=1)
    compress_h5ad(src, tmp_path / "four.h5ad", n_jobs=4)
    assert (tmp_path / "one.h5ad").read_bytes() == (tmp_path / "four.h5ad").read_bytes()


def test_in_place_replacement(tmp_path):
    path = tmp_path / "sc.h5ad"
    adata = _single_cell(n_obs=1500)
    adata.write(path)
    before = path.stat().st_size
    with pytest.raises(FileExistsError):
        compress_h5ad(path, path)
    result = compress_h5ad(path, path, overwrite=True)
    assert result["dst_bytes"] == path.stat().st_size < before
    assert (ad.read_h5ad(path).X != adata.X).nnz == 0
    assert not list(tmp_path.glob("*.partial"))


def test_failure_leaves_no_partial_and_keeps_source(tmp_path, monkeypatch):
    src, dst = tmp_path / "sc.h5ad", tmp_path / "out.h5ad"
    _single_cell(n_obs=1500).write(src)
    original = src.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(_h5codec, "verify_tree", boom)
    with pytest.raises(RuntimeError, match="injected"):
        compress_h5ad(src, dst)
    assert not dst.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert src.read_bytes() == original


def test_reclaims_space_left_by_in_place_rewrites(tmp_path):
    path = tmp_path / "sc.h5ad"
    _single_cell(n_obs=1500).write(path)
    rng = np.random.default_rng(0)
    for _ in range(5):  # e.g. rerunning pp.pca on the same file
        cx.data.write_obsm_to_h5ad(path, "X_pca", rng.standard_normal((1500, 50)).astype(np.float32))
    bloated = path.stat().st_size
    compress_h5ad(path, tmp_path / "clean.h5ad")
    assert (tmp_path / "clean.h5ad").stat().st_size < bloated


# ---------------------------------------------------------------------------
# Verifier negative controls
# ---------------------------------------------------------------------------


def _copy_then(tmp_path, mutate) -> tuple[Path, Path]:
    src, dst = tmp_path / "sc.h5ad", tmp_path / "sc.gz.h5ad"
    _single_cell(n_obs=1500).write(src)
    compress_h5ad(src, dst, verify=False)
    with h5py.File(dst, "r+") as f:
        mutate(f)
    return src, dst


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda f: f["X/data"].__setitem__(7, np.nextafter(f["X/data"][7], np.float32(np.inf))),
            id="one-ulp",
        ),
        pytest.param(lambda f: f["obs"].attrs.__setitem__("column-order", ["flag"]), id="attr-value"),
        pytest.param(lambda f: f["uns/nested/b"].attrs.__setitem__("extra", 1), id="extra-attr"),
        pytest.param(lambda f: f["uns/nested/b"].create_dataset("new", data=1), id="extra-member"),
    ],
)
def test_verifier_detects_differences(tmp_path, mutate):
    src, dst = _copy_then(tmp_path, mutate)
    with h5py.File(src, "r") as fs, h5py.File(dst, "r") as fd, pytest.raises(ValueError):
        _h5codec.verify_tree(fs, fd)


def test_rejects_bad_arguments(tmp_path):
    src = tmp_path / "sc.h5ad"
    _single_cell(n_obs=100).write(src)
    with pytest.raises(ValueError):
        compress_h5ad(src, tmp_path / "o.h5ad", level=0)
    with pytest.raises(ValueError):
        compress_h5ad(src, tmp_path / "o.h5ad", n_jobs=0)
    with pytest.raises(FileNotFoundError):
        compress_h5ad(tmp_path / "missing.h5ad", tmp_path / "o.h5ad")


# ---------------------------------------------------------------------------
# Streaming reads of compressed files (threaded decode in iter_matrix_chunks)
# ---------------------------------------------------------------------------


def _blocks(path, *, axis, chunk_size, layer=None):
    from crispyx.data import iter_matrix_chunks, read_backed

    backed = read_backed(path)
    try:
        matrix = backed.layers[layer] if layer else None
        return [
            block
            for _slc, block in iter_matrix_chunks(
                backed, axis=axis, chunk_size=chunk_size, convert_to_dense=False,
                matrix=matrix, warn_slow_axis=False,
            )
        ]
    finally:
        backed.file.close()


def _assert_same_blocks(expected, actual):
    assert len(expected) == len(actual)
    for a, b in zip(expected, actual):
        assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
        if sp.issparse(a):
            for attr in ("data", "indices", "indptr"):
                x, y = getattr(a, attr), getattr(b, attr)
                assert x.dtype == y.dtype and np.array_equal(x, y)
        else:
            assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize(
    "fmt, axis",
    [("csr", 0), ("csc", 1), ("dense", 0)],
)
def test_streaming_a_compressed_file_yields_identical_blocks(tmp_path, fmt, axis):
    adata = _single_cell(n_obs=2500, n_vars=300)
    if fmt == "csc":
        adata.X = sp.csc_matrix(adata.X)
    elif fmt == "dense":
        adata.X = adata.X.toarray()
    src, dst = tmp_path / "raw.h5ad", tmp_path / "gz.h5ad"
    adata.write(src)
    compress_h5ad(src, dst)
    for chunk_size in (1, 97, 4096):
        _assert_same_blocks(
            _blocks(src, axis=axis, chunk_size=chunk_size),
            _blocks(dst, axis=axis, chunk_size=chunk_size),
        )


def test_streaming_a_compressed_layer_and_deflate_only_file(tmp_path):
    """Public .h5ad files are often written by anndata with gzip but no
    shuffle; that pipeline is decoded in parallel too."""
    adata = _single_cell(n_obs=2000, n_vars=200)
    src, gz = tmp_path / "raw.h5ad", tmp_path / "anndata_gzip.h5ad"
    adata.write(src)
    adata.write(gz, compression="gzip")
    with h5py.File(gz, "r") as f:
        assert f["X/data"].compression == "gzip" and not f["X/data"].shuffle
        assert _h5codec.deflate_shuffle(f["X/data"]) is False
    _assert_same_blocks(_blocks(src, axis=0, chunk_size=300), _blocks(gz, axis=0, chunk_size=300))
    _assert_same_blocks(
        _blocks(src, axis=0, chunk_size=300, layer="counts"),
        _blocks(gz, axis=0, chunk_size=300, layer="counts"),
    )


def test_decoder_falls_back_for_unwritten_chunks(tmp_path):
    path = tmp_path / "partial.h5"
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "d", shape=(1000, 50), dtype=np.float64, chunks=(100, 50),
            compression="gzip", shuffle=True, fillvalue=np.nan,
        )
        ds[:300] = np.arange(300 * 50, dtype=np.float64).reshape(300, 50)  # chunks 3.. never written
    with h5py.File(path, "r") as f, _h5codec.ChunkDecoder(4) as decoder:
        ds = f["d"]
        assert decoder.supports(ds)
        for start, stop in ((0, 250), (250, 450), (990, 1000), (0, 1000)):
            assert np.array_equal(decoder.read(ds, start, stop), ds[start:stop], equal_nan=True)


def test_uncompressed_files_keep_the_plain_slicing_path(tmp_path):
    from crispyx.data import _decoded_block_reader, read_backed

    path = tmp_path / "raw.h5ad"
    _single_cell(n_obs=500).write(path)
    backed = read_backed(path)
    try:
        assert _decoded_block_reader(backed.X, "csr", 0, backed.n_obs, backed.n_vars) == (None, None)
    finally:
        backed.file.close()
