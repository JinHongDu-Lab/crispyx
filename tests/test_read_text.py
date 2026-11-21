"""cx.read_text / cx.read_csv: text count tables streamed into h5ad, read as
anndata reads them in memory."""

from __future__ import annotations

import gzip
import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import crispyx as cx
import crispyx.data as cx_data


def _table(rng, n_rows=40, n_cols=25):
    counts = rng.poisson(0.4, (n_rows, n_cols)).astype(np.float32)
    counts[3] = 0  # an all-zero row
    return counts


def _write(path, counts, *, sep=",", header=True, row_names=True, corner=False, comment=None):
    rows = [f"r{i}" for i in range(counts.shape[0])]
    cols = [f"c{j}" for j in range(counts.shape[1])]
    lines = [] if comment is None else [comment]
    if header:
        lines.append(sep.join((["corner"] if corner else []) + cols))
    for name, row in zip(rows, counts):
        values = [f"{v:g}" for v in row]
        lines.append(sep.join(([f'"{name}"'] if row_names else []) + values))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _as_dense(path):
    adata = ad.read_h5ad(path)
    assert sp.isspmatrix_csr(adata.X) or isinstance(adata.X, sp.csr_array)
    return adata, adata.X.toarray()


@pytest.mark.parametrize("compressed", [False, True])
def test_read_csv_matches_anndata(tmp_path, compressed, monkeypatch):
    monkeypatch.setattr(cx_data, "_TEXT_BLOCK_LINES", 7)  # several blocks
    counts = _table(np.random.default_rng(0))
    source = tmp_path / ("t.csv.gz" if compressed else "t.csv")
    _write(source, counts)
    cx.read_csv(source, output_path=tmp_path / "t.h5ad", verbose=False)
    got, x = _as_dense(tmp_path / "t.h5ad")
    expected = ad.io.read_csv(source)
    np.testing.assert_array_equal(x, expected.X)
    assert list(got.obs_names) == list(expected.obs_names) == [f"r{i}" for i in range(counts.shape[0])]
    assert list(got.var_names) == list(expected.var_names)


def test_transpose_takes_rows_as_genes(tmp_path, monkeypatch):
    monkeypatch.setattr(cx_data, "_TEXT_BLOCK_LINES", 4)
    counts = _table(np.random.default_rng(1), n_rows=30, n_cols=50)  # 30 genes x 50 cells
    _write(tmp_path / "g.csv.gz", counts)
    cx.read_csv(tmp_path / "g.csv.gz", output_path=tmp_path / "g.h5ad", transpose=True, verbose=False)
    got, x = _as_dense(tmp_path / "g.h5ad")
    np.testing.assert_array_equal(x, counts.T)
    assert list(got.obs_names) == [f"c{j}" for j in range(50)]
    assert list(got.var_names) == [f"r{i}" for i in range(30)]
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())  # no scratch file left


def test_read_text_whitespace_comments_and_corner_label(tmp_path):
    counts = _table(np.random.default_rng(2))
    # whitespace, no header, no row names: both numbered
    _write(tmp_path / "plain.txt", counts, sep="  ", header=False, row_names=False)
    got, x = _as_dense(cx.read_text(tmp_path / "plain.txt", output_path=tmp_path / "plain.h5ad", verbose=False).path)
    np.testing.assert_array_equal(x, ad.io.read_text(tmp_path / "plain.txt").X)
    assert list(got.var_names) == [str(j) for j in range(counts.shape[1])]
    # names from the last comment line
    names = " ".join(f"g{j}" for j in range(counts.shape[1]))
    _write(tmp_path / "comment.tsv", counts, sep="\t", header=False, row_names=False, comment=f"# {names}")
    got = ad.read_h5ad(cx.read_text(tmp_path / "comment.tsv", output_path=tmp_path / "c.h5ad", delimiter="\t",
                                    verbose=False).path)
    assert got.var_names[0] == "g0"
    # a header with a corner label over the row names
    _write(tmp_path / "corner.csv", counts, corner=True)
    got = ad.read_h5ad(cx.read_csv(tmp_path / "corner.csv", output_path=tmp_path / "k.h5ad", verbose=False).path)
    assert list(got.var_names) == [f"c{j}" for j in range(counts.shape[1])]
    assert got.obs_names[0] == "r0"


def test_ragged_lines_and_missing_delimiter_raise(tmp_path):
    (tmp_path / "ragged.csv").write_text("a,b,c\nr0,1,2,3\nr1,1,2\n")
    with pytest.raises(ValueError, match="line 3"):
        cx.read_csv(tmp_path / "ragged.csv", output_path=tmp_path / "r.h5ad", verbose=False)
    assert not (tmp_path / "r.h5ad").exists()
    (tmp_path / "tabs.tsv").write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="delimiter"):
        cx.read_csv(tmp_path / "tabs.tsv", output_path=tmp_path / "t.h5ad", verbose=False)
    # a value that does not parse, as anndata refuses it (and named for the function called)
    (tmp_path / "bad.csv").write_text("a,b,c\nr0,1,2,3abc\nr1,4,5,6\n")
    with pytest.raises(ValueError, match="read_csv: line 2 .*3abc"):
        cx.read_csv(tmp_path / "bad.csv", output_path=tmp_path / "b.h5ad", verbose=False)
    with pytest.raises(ValueError):
        ad.io.read_csv(tmp_path / "bad.csv")
    # a row name with no values
    (tmp_path / "short.csv").write_text("a,b\nr0\nr1,1,2\n")
    with pytest.raises(ValueError, match="line 2"):
        cx.read_csv(tmp_path / "short.csv", output_path=tmp_path / "s.h5ad", first_column_names=True, verbose=False)


def test_header_rules_follow_anndata(tmp_path):
    cases = {
        "empty_comment.txt": ("# g0 g1 g2\n#\n1 2 3\n4 5 6\n", None),
        "integer_names.csv": ("1,2,3\n10,5,6,7\n11,8,9,0\n", ","),
        "header_and_names.csv": ("x,y\nr0,1,2\nr1,3,4\n", ","),
    }
    for file, (text, delimiter) in cases.items():
        (tmp_path / file).write_text(text)
        expected = ad.io.read_text(tmp_path / file, delimiter=delimiter)
        got = ad.read_h5ad(cx.read_text(tmp_path / file, output_path=tmp_path / f"{file}.h5ad", delimiter=delimiter,
                                        verbose=False).path)
        np.testing.assert_array_equal(got.X.toarray(), expected.X, err_msg=file)
        assert list(got.obs_names) == list(expected.obs_names), file
        assert list(got.var_names) == list(expected.var_names), file


def test_the_output_is_reused_until_the_table_or_call_changes(tmp_path, capsys):
    counts = _table(np.random.default_rng(3))
    _write(tmp_path / "t.csv", counts)
    out = tmp_path / "t.h5ad"
    cx.read_csv(tmp_path / "t.csv", output_path=out)
    capsys.readouterr()
    cx.read_csv(tmp_path / "t.csv", output_path=out)
    assert "Loading existing result" in capsys.readouterr().out
    cx.read_csv(tmp_path / "t.csv", output_path=out, transpose=True)  # another call
    assert ad.read_h5ad(out).shape == counts.T.shape
    stat = os.stat(tmp_path / "t.csv")
    os.utime(tmp_path / "t.csv", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # the table changed
    cx.read_csv(tmp_path / "t.csv", output_path=out)
    assert "Loading existing result" not in capsys.readouterr().out
    assert ad.read_h5ad(out).shape == counts.shape
