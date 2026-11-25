"""cx.read_mtx / cx.read_10x_mtx: MatrixMarket count matrices streamed into
h5ad, read as anndata and Scanpy read them in memory."""

from __future__ import annotations

import gzip
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

import crispyx as cx
import crispyx.data as cx_data

N_GENES, N_CELLS = 30, 23


def _counts(seed=0, n_genes=N_GENES, n_cells=N_CELLS):
    """Genes x cells counts with an all-zero gene and all-zero last cells."""
    counts = np.random.default_rng(seed).poisson(0.5, (n_genes, n_cells))
    counts[4], counts[:, -2:] = 0, 0
    return sp.coo_matrix(counts)


def _write_mtx(path, matrix, *, order="column", field="integer", entries=None, shape=None):
    """``matrix`` as a coordinate file with its entries sorted by column, by
    row, or shuffled."""
    index = {
        "column": np.lexsort((matrix.row, matrix.col)),
        "row": np.lexsort((matrix.col, matrix.row)),
        "shuffled": np.random.default_rng(1).permutation(matrix.nnz),
    }[order]
    n_rows, n_cols = shape or matrix.shape
    lines = [f"%%MatrixMarket matrix coordinate {field} general", "%a comment", "",
             f"{n_rows} {n_cols} {matrix.nnz if entries is None else entries}"]
    lines += [f"{matrix.row[i] + 1} {matrix.col[i] + 1} {matrix.data[i]}" for i in index]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _write_folder(folder, matrix, *, gz=True, prefix="", feature_columns=3, feature_file="features.tsv", **mtx):
    """A 10x-style folder: genes 3 and 7 share a symbol, the last five rows
    are guide capture."""
    folder.mkdir(exist_ok=True)
    suffix = ".gz" if gz else ""
    opener = gzip.open if gz else open
    _write_mtx(folder / f"{prefix}matrix.mtx{suffix}", matrix, **mtx)
    n_genes, n_cells = matrix.shape
    rows = [(f"ENSG{g:04d}", "DUP" if g in (3, 7) else f"G{g}",
             "Gene Expression" if g < n_genes - 5 else "CRISPR Guide Capture") for g in range(n_genes)]
    columns = {1: [1], 2: [0, 1], 3: [0, 1, 2]}[feature_columns]
    with opener(folder / f"{prefix}{feature_file}{suffix}", "wt") as f:
        f.write("".join("\t".join(row[c] for c in columns) + "\n" for row in rows))
    with opener(folder / f"{prefix}barcodes.tsv{suffix}", "wt") as f:
        f.write("".join(f"BC{c:03d}-1\n" for c in range(n_cells)))
    return folder


def _x(path):
    x = ad.read_h5ad(path).X
    assert x.format == "csr" and x.has_sorted_indices and x.dtype == np.float32
    return x.toarray()


def _no_scratch(folder):
    return not any(p.name.startswith(".") for p in folder.iterdir())


@pytest.mark.parametrize("gz", [True, False])
def test_read_mtx_matches_anndata(tmp_path, gz):
    source = tmp_path / ("m.mtx.gz" if gz else "m.mtx")
    _write_mtx(source, _counts())
    cx.read_mtx(source, output_path=tmp_path / "m.h5ad", verbose=False)
    got, expected = ad.read_h5ad(tmp_path / "m.h5ad"), ad.io.read_mtx(source)
    np.testing.assert_array_equal(got.X.toarray(), expected.X.toarray())
    assert list(got.obs_names) == list(expected.obs_names)
    assert list(got.var_names) == list(expected.var_names)
    cx.read_mtx(source, output_path=tmp_path / "t.h5ad", transpose=True, verbose=False)
    np.testing.assert_array_equal(_x(tmp_path / "t.h5ad"), expected.X.toarray().T)
    # another stored type: wider spill records, the same matrix
    cx.read_mtx(source, output_path=tmp_path / "f8.h5ad", dtype="float64", verbose=False)
    wide = ad.read_h5ad(tmp_path / "f8.h5ad").X
    assert wide.dtype == np.float64
    np.testing.assert_array_equal(wide.toarray(), expected.X.toarray())


@pytest.mark.parametrize("var_names", ["gene_symbols", "gene_ids"])
@pytest.mark.parametrize("gex_only", [True, False])
def test_read_10x_mtx_matches_scanpy(tmp_path, var_names, gex_only):
    folder = _write_folder(tmp_path / "mex", _counts())
    cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", var_names=var_names, gex_only=gex_only, verbose=False)
    got = ad.read_h5ad(tmp_path / "x.h5ad")
    expected = sc.read_10x_mtx(folder, var_names=var_names, gex_only=gex_only)
    np.testing.assert_array_equal(got.X.toarray(), expected.X.toarray())
    assert list(got.obs_names) == list(expected.obs_names)
    assert list(got.var_names) == list(expected.var_names)
    assert list(got.var.columns) == list(expected.var.columns)
    for column in expected.var.columns:
        assert list(got.var[column]) == list(expected.var[column])
    assert got.n_vars == (N_GENES - 5 if gex_only else N_GENES)
    # make_unique=False keeps the repeated symbol
    cx.read_10x_mtx(folder, output_path=tmp_path / "d.h5ad", var_names=var_names, make_unique=False, verbose=False)
    assert ("DUP" in cx.load_var(tmp_path / "d.h5ad").index) == (var_names == "gene_symbols")
    assert _no_scratch(tmp_path)


@pytest.mark.parametrize("order", ["column", "row", "shuffled"])
@pytest.mark.parametrize("memory_limit_gb", [None, 1e-7])
def test_any_entry_order_blocks_and_bands_give_the_same_matrix(tmp_path, monkeypatch, order, memory_limit_gb):
    monkeypatch.setattr(cx_data, "_MTX_BLOCK_ENTRIES", 11)  # several blocks
    matrix = _counts(2)
    folder = _write_folder(tmp_path / "mex", matrix, order=order)
    # at 1e-7 GB a band holds two non-zeros: one band per cell, most of them split
    cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", gex_only=False, memory_limit_gb=memory_limit_gb,
                    verbose=False)
    np.testing.assert_array_equal(_x(tmp_path / "x.h5ad"), matrix.toarray().T)


def test_a_band_over_budget_is_split_by_its_cells(tmp_path, monkeypatch):
    """Cells ordered by size put most non-zeros in the first band of equal
    width; it is then built in parts."""
    counts = np.zeros((N_GENES, 40), dtype=int)
    counts[:, :4] = 1 + np.arange(N_GENES)[:, None]  # four full cells, then nearly empty ones
    counts[0, 4:] = 1
    matrix = sp.coo_matrix(counts)
    _write_mtx(tmp_path / "m.mtx", matrix, order="row")
    blocks = []
    append = cx_data._append_sparse_block
    monkeypatch.setattr(cx_data, "_append_sparse_block", lambda h5, block: blocks.append(block.shape[0]) or append(h5, block))
    monkeypatch.setattr(cx_data, "_conversion_buffer_budget_bytes", lambda limit: 40 * 2 * 12)  # 40 non-zeros a band
    cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "m.h5ad", transpose=True, verbose=False)
    np.testing.assert_array_equal(_x(tmp_path / "m.h5ad"), counts.T)
    # four bands of ten cells; the first is built in parts of at most 40 non-zeros (a full cell has 30)
    assert blocks == [1, 1, 1, 7, 10, 10, 10]


def test_one_and_two_column_features_and_legacy_names(tmp_path):
    matrix = _counts(3)
    symbols = ad.utils.make_index_unique(pd.Index(["DUP" if g in (3, 7) else f"G{g}" for g in range(N_GENES)]))
    # one column: the names, whichever var_names is asked for, and no other column
    one = _write_folder(tmp_path / "one", matrix, feature_columns=1)
    for var_names in ("gene_symbols", "gene_ids"):
        cx.read_10x_mtx(one, output_path=tmp_path / "one.h5ad", var_names=var_names, verbose=False)
        var = cx.load_var(tmp_path / "one.h5ad")
        assert list(var.index) == list(symbols) and list(var.columns) == []
    # two columns in an uncompressed genes.tsv (Cell Ranger 2), as Scanpy reads it
    legacy = _write_folder(tmp_path / "legacy", matrix, gz=False, feature_columns=2, feature_file="genes.tsv")
    cx.read_10x_mtx(legacy, output_path=tmp_path / "legacy.h5ad", verbose=False)
    got, expected = ad.read_h5ad(tmp_path / "legacy.h5ad"), sc.read_10x_mtx(legacy)
    np.testing.assert_array_equal(got.X.toarray(), expected.X.toarray())
    assert list(got.var_names) == list(expected.var_names) and list(got.var.columns) == ["gene_ids"]
    # two columns under the current name, and a prefix
    two = _write_folder(tmp_path / "two", matrix, feature_columns=2, prefix="s1_")
    cx.read_10x_mtx(two, output_path=tmp_path / "two.h5ad", var_names="gene_ids", prefix="s1_", verbose=False)
    var = cx.load_var(tmp_path / "two.h5ad")
    assert var.index[0] == "ENSG0000" and list(var.columns) == ["gene_symbols"]
    np.testing.assert_array_equal(_x(tmp_path / "two.h5ad"), matrix.toarray().T)


def test_names_are_taken_as_written(tmp_path):
    """Missing-value words, quotes and an empty line are names like any other."""
    folder = tmp_path / "mex"
    folder.mkdir()
    (folder / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n4 3 2\n1 1 1\n2 2 3\n")
    (folder / "features.tsv").write_text('NA\n"A\n\n"B"\n')
    (folder / "barcodes.tsv").write_text("NA\nnan\nA\n")
    cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", make_unique=False, verbose=False)
    assert list(cx.load_var(tmp_path / "x.h5ad").index) == ["NA", '"A', "", '"B"']
    assert list(cx.load_obs(tmp_path / "x.h5ad").index) == ["NA", "nan", "A"]
    # empty lines at the end of a table are not names
    (folder / "barcodes.tsv").write_text("NA\nnan\nA\n\n\n")
    cx.read_10x_mtx(folder, output_path=tmp_path / "t.h5ad", make_unique=False, verbose=False)
    assert list(cx.load_obs(tmp_path / "t.h5ad").index) == ["NA", "nan", "A"]


def test_a_feature_table_with_lines_of_unequal_length_is_refused(tmp_path):
    """pandas pads a short line, and the gene whose type is then missing
    would be dropped by gex_only with its counts."""
    folder = tmp_path / "mex"
    folder.mkdir()
    (folder / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n3 2 3\n1 1 1\n2 1 5\n3 2 2\n")
    (folder / "barcodes.tsv").write_text("c1\nc2\n")
    for table, match in (
        ("E1\tA\tGene Expression\nE2\tB\nE3\tC\tGene Expression\n", "line 2 of .*features.tsv has 2 tab-separated fields; the first line has 3"),
        ("a\nb\tc\td\ne\n", "line 2 of .*features.tsv has 3 tab-separated fields; the first line has 1"),
        ("E1\tA\tGene Expression\n\nE3\tC\tGene Expression\n", "line 2 of .*features.tsv has 1 tab-separated fields"),
        ("", ".*features.tsv is empty"),
    ):
        (folder / "features.tsv").write_text(table)
        with pytest.raises(ValueError, match="read_10x_mtx: " + match):
            cx.read_10x_mtx(folder, output_path=tmp_path / "y.h5ad", verbose=False)
    assert not (tmp_path / "y.h5ad").exists()


def test_integer_dtype_is_stored_or_refused_never_wrapped(tmp_path):
    (tmp_path / "m.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 300\n2 2 7\n")
    cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "i.h5ad", dtype="int32", verbose=False)
    got = ad.read_h5ad(tmp_path / "i.h5ad").X
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got.toarray(), [[300, 0], [0, 7]])
    with pytest.raises(ValueError, match="does not fit dtype uint8"):
        cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "u.h5ad", dtype="uint8", verbose=False)
    (tmp_path / "neg.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 -1\n")
    with pytest.raises(ValueError, match="does not fit dtype uint32"):
        cx.read_mtx(tmp_path / "neg.mtx", output_path=tmp_path / "u.h5ad", dtype="uint32", verbose=False)
    (tmp_path / "real.mtx").write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1.5\n")
    with pytest.raises(ValueError, match="read_mtx"):
        cx.read_mtx(tmp_path / "real.mtx", output_path=tmp_path / "u.h5ad", dtype="int32", verbose=False)
    assert not (tmp_path / "u.h5ad").exists()


@pytest.mark.parametrize("value", ["1e40", "inf", "-inf"])
def test_a_real_value_float32_cannot_hold_is_refused(tmp_path, value):
    (tmp_path / "m.mtx").write_text(f"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 {value}\n2 2 3\n")
    with pytest.raises(ValueError, match="does not fit dtype float32"), np.errstate(over="ignore"):
        cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "m.h5ad", verbose=False)
    assert not (tmp_path / "m.h5ad").exists()
    if value == "1e40":  # float64 holds it
        cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "m.h5ad", dtype="float64", verbose=False)
        assert ad.read_h5ad(tmp_path / "m.h5ad").X[0, 0] == 1e40


def test_duplicates_are_summed_and_zeros_dropped(tmp_path):
    (tmp_path / "m.mtx").write_text(
        "%%MatrixMarket matrix coordinate real general\n3 2 5\n1 1 1.5\n3 2 2\n1 1 2\n2 2 0\n2 1 4\n"
    )
    cx.read_mtx(tmp_path / "m.mtx", output_path=tmp_path / "m.h5ad", verbose=False)
    got = ad.read_h5ad(tmp_path / "m.h5ad").X
    np.testing.assert_array_equal(got.toarray(), [[3.5, 0], [4, 0], [0, 2]])
    assert got.nnz == 3
    np.testing.assert_array_equal(got.toarray(), ad.io.read_mtx(tmp_path / "m.mtx").X.toarray())
    (tmp_path / "empty.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n3 4 0\n")
    cx.read_mtx(tmp_path / "empty.mtx", output_path=tmp_path / "empty.h5ad", verbose=False)
    assert ad.read_h5ad(tmp_path / "empty.h5ad").shape == (3, 4)


@pytest.mark.parametrize("text, match", [
    ("1 2 3\n", "header"),
    ("%%MatrixMarket matrix array real general\n2 2\n1\n2\n3\n4\n", "header"),
    ("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 1\n", "pattern general"),
    ("%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 1 1\n", "real symmetric"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2\n1 1 1\n", "rows columns entries"),
    ("%%MatrixMarket matrix coordinate integer general\n% only comments\n", "rows columns entries"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n3 1 1\n", "outside its declared 2 x 2"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n0 1 1\n", "outside its declared"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 3\n1 1 1\n2 1 1\n", "declares 3 entries and has 2"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 1\n2 1 1\n", "declares 1 entries and has 2"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n2 1\n", "without a value"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n2 x 1\n", "read_mtx"),
    ("%%MatrixMarket matrix coordinate integer general\n3 3 2\n3 1 2 7\n2 2 3 9\n", "more than three fields"),
    ("%%MatrixMarket matrix coordinate integer general\n3 3 2\n1 1 1\n3 1 2 7 9\n", "read_mtx"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 0\n1 1 5\n2 2 7\n", "declares 0 entries and has 2"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n", "declares 2 entries and has 0"),
    ("%%MatrixMarket matrix coordinate integer general\n2 2 2\n\n\n", "declares 2 entries and has 0"),
])
def test_malformed_files_raise_and_leave_nothing(tmp_path, text, match):
    (tmp_path / "bad.mtx").write_text(text)
    with pytest.raises(ValueError, match=match):
        cx.read_mtx(tmp_path / "bad.mtx", output_path=tmp_path / "bad.h5ad", verbose=False)
    assert [p.name for p in tmp_path.iterdir()] == ["bad.mtx"]


def test_folder_errors(tmp_path):
    matrix = _counts(4)
    folder = _write_folder(tmp_path / "mex", matrix)
    with pytest.raises(ValueError, match="var_names"):
        cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", var_names="symbols", verbose=False)
    with gzip.open(folder / "barcodes.tsv.gz", "at") as f:
        f.write("EXTRA-1\n")
    with pytest.raises(ValueError, match=f"{N_CELLS} cells but {N_CELLS + 1} names"):
        cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", verbose=False)
    (folder / "features.tsv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="features.tsv or genes.tsv"):
        cx.read_10x_mtx(folder, output_path=tmp_path / "x.h5ad", verbose=False)
    assert not (tmp_path / "x.h5ad").exists() and _no_scratch(tmp_path)


def test_the_output_is_reused_until_a_file_or_the_call_changes(tmp_path, capsys, monkeypatch):
    folder = _write_folder(tmp_path / "mex", _counts(5))
    out = tmp_path / "x.h5ad"

    def reused(**kwargs):
        capsys.readouterr()
        cx.read_10x_mtx(folder, output_path=out, **kwargs)
        return "Loading existing result" in capsys.readouterr().out

    assert not reused()
    assert reused()
    assert not reused(gex_only=False) and ad.read_h5ad(out).n_vars == N_GENES  # another call
    assert reused(gex_only=False)
    for file in ("barcodes.tsv.gz", "matrix.mtx.gz"):  # a file changed
        stat = os.stat(folder / file)
        os.utime(folder / file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert not reused(gex_only=False)
    assert reused(gex_only=False) and not reused(gex_only=False, force=True)
    # a reused output is returned without reading the barcode and feature tables again
    monkeypatch.setattr(cx_data, "_read_name_table", lambda *args: pytest.fail("the tables were read"))
    assert reused(gex_only=False)


def test_a_truncated_gzip_raises_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cx_data, "_MTX_BLOCK_ENTRIES", 50)
    _write_mtx(tmp_path / "m.mtx.gz", _counts(6, n_genes=200, n_cells=60))
    whole = (tmp_path / "m.mtx.gz").read_bytes()
    (tmp_path / "m.mtx.gz").write_bytes(whole[: len(whole) // 2])  # a download that stopped halfway
    with pytest.raises((EOFError, ValueError)):
        cx.read_mtx(tmp_path / "m.mtx.gz", output_path=tmp_path / "m.h5ad", verbose=False)
    assert [p.name for p in tmp_path.iterdir()] == ["m.mtx.gz"]
