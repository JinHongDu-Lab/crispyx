"""tl.infercnv and pp.genomic_position_from_gtf: infercnvpy's method, streamed."""

from __future__ import annotations

import gzip
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx


def _reference_infercnv(adata, *, reference_key=None, reference_cat=None, lfc_clip=3, window_size=100, step=10,
                        dynamic_threshold=1.5, exclude_chromosomes=("chrX", "chrY")):
    """infercnvpy 0.6.1's ``tl.infercnv`` on one chunk holding every cell,
    transcribed: dense, a row at a time, with ``np.convolve``."""
    var_mask = adata.var["chromosome"].isnull()
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)
    X = adata.X.toarray()
    if reference_key is None:
        reference = X.mean(axis=0)[None, :]
    else:
        cats = [reference_cat] if isinstance(reference_cat, str) else reference_cat
        reference = np.vstack([X[adata.obs[reference_key].to_numpy() == c].mean(axis=0) for c in cats])
    reference = reference[:, ~var_mask.to_numpy()]
    x = X[:, ~var_mask.to_numpy()]
    var = adata.var.loc[~var_mask, ["chromosome", "start", "end"]]
    if reference.shape[0] == 1:
        centered = x - reference[0]
    else:
        lo, hi = reference.min(axis=0), reference.max(axis=0)
        centered = np.zeros_like(x)
        centered[x > hi] = (x - hi)[x > hi]
        centered[x < lo] = (x - lo)[x < lo]
    clipped = np.clip(centered, -lfc_clip, lfc_clip)

    def natural(values):
        return sorted(values, key=lambda t: [int(p) if p.isdigit() else p.lower() for p in re.split("([0-9]+)", t)])

    chromosomes = natural([c for c in var["chromosome"].unique() if c.startswith("chr") and c != "chrM"])
    means, chr_pos, start = [], {}, 0
    for chromosome in chromosomes:
        genes = var.loc[var["chromosome"] == chromosome].sort_values("start").index.values
        sub = clipped[:, var.index.get_indexer(genes)]
        n = window_size
        if n < sub.shape[1]:
            r = np.arange(1, n + 1)
            pyramid = np.minimum(r, r[::-1])
            smoothed = np.apply_along_axis(lambda row: np.convolve(row, pyramid, mode="valid"), 1, sub) / pyramid.sum()
            smoothed = smoothed[:, np.arange(0, smoothed.shape[1], step)]
        else:
            n = sub.shape[1]
            smoothed = np.apply_along_axis(lambda row: np.convolve(row, np.ones(n), mode="valid"), 1, sub) / n
        chr_pos[chromosome] = start
        start += smoothed.shape[1]
        means.append(smoothed)
    smoothed = np.hstack(means)
    res = smoothed - np.median(smoothed, axis=1)[:, None]
    if dynamic_threshold is not None:
        res[np.abs(res) < dynamic_threshold * np.std(res)] = 0
    return chr_pos, res


def _tumour(tmp_path):
    """Cells with genes on chr1, chr2, chr3 (fewer genes than a window),
    chr10, chrX, chrM, an unplaced contig and none; a chr2 gain in 100."""
    rng = np.random.default_rng(0)
    n, g = 400, 1500
    chromosome = rng.choice(["chr1", "chr2", "chr10", "chr3", "chrX", "chrM", "GL000194.1", None], g,
                            p=[.32, .3, .2, .04, .05, .02, .02, .05])
    var = pd.DataFrame({"chromosome": chromosome, "start": rng.permutation(g) * 1000, "end": 0},
                       index=[f"g{j}" for j in range(g)])
    var["end"] = var["start"] + 500
    X = np.log1p(rng.poisson(rng.gamma(0.6, 1.0, g), (n, g)) * rng.uniform(0.5, 1.5, (n, 1)))
    X[:100, (var["chromosome"] == "chr2").to_numpy()] += 0.8
    obs = pd.DataFrame({"type": np.where(np.arange(n) < 100, "tumour", np.where(np.arange(n) < 250, "T", "B"))},
                       index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(sp.csr_matrix(X), obs=obs, var=var)
    path = tmp_path / "tumour.h5ad"
    adata.write_h5ad(path)
    return path, adata


@pytest.mark.parametrize("kwargs", [
    {},
    {"reference_key": "type", "reference_cat": ["T", "B"]},
    {"reference_key": "type", "reference_cat": "T", "dynamic_threshold": None, "window_size": 50, "step": 1},
])
def test_profiles_match_infercnvpy(tmp_path, kwargs):
    path, adata = _tumour(tmp_path)
    chr_pos, expected = _reference_infercnv(adata, **kwargs)
    uns, got = cx.tl.infercnv(path, inplace=False, chunk_size=73, verbose=False, **kwargs)
    assert uns["chr_pos"] == chr_pos and list(chr_pos) == ["chr1", "chr2", "chr3", "chr10"]
    one = cx.tl.infercnv(path, inplace=False, verbose=False, exclude_chromosomes="chrX", **kwargs)[0]
    assert "chrX" not in one["chr_pos"] and one["chr_pos"] == _reference_infercnv(
        adata, exclude_chromosomes=("chrX",), **kwargs)[0]
    np.testing.assert_allclose(got.toarray(), expected, atol=1e-6)
    assert got.nnz == np.count_nonzero(expected)


def test_profiles_are_written_to_the_file(tmp_path):
    path, adata = _tumour(tmp_path)
    cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], chunk_size=50, verbose=False)
    stored = ad.read_h5ad(path)
    chr_pos, expected = _reference_infercnv(adata, reference_key="type", reference_cat=["T", "B"])
    assert sp.issparse(stored.obsm["X_cnv"]) and dict(stored.uns["cnv"]["chr_pos"]) == chr_pos
    np.testing.assert_allclose(stored.obsm["X_cnv"].toarray(), expected, atol=1e-6)
    # The gain shows in the tumour cells' chr2 windows only.
    windows = stored.obsm["X_cnv"].toarray()
    chr2 = slice(chr_pos["chr2"], chr_pos["chr3"])
    assert windows[:100, chr2].mean() > 0.1 > abs(windows[100:, chr2].mean())
    assert not list(tmp_path.glob(".*partial"))
    pd.testing.assert_frame_equal(stored.uns["cnv"]["windows"].reset_index(drop=True),
                                  cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], inplace=False,
                                                 verbose=False)[0]["windows"])


def test_the_file_is_checked_before_the_passes_and_scratch_is_removed(tmp_path, monkeypatch):
    """A file open elsewhere is an error before any pass; the profiles kept
    for the threshold go to the scratch root and are removed after the run."""
    import h5py
    from crispyx import cnv

    path, _ = _tumour(tmp_path)
    monkeypatch.setattr(cnv, "read_backed", lambda *a: pytest.fail("a pass ran"))
    with h5py.File(path, "r"):
        with pytest.raises(OSError, match="open elsewhere.*inplace=False"):
            cx.tl.infercnv(path, verbose=False)
    monkeypatch.undo()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(cnv, "_scratch_root", lambda size, fallback: scratch)
    created = []
    real = cnv.tempfile.mkstemp
    monkeypatch.setattr(cnv.tempfile, "mkstemp", lambda **kw: created.append(real(**kw)) or created[-1])
    cx.tl.infercnv(path, verbose=False)
    assert len(created) == 1 and Path(created[0][1]).parent == scratch and not list(scratch.iterdir())
    assert not list(tmp_path.glob(".*partial"))


def test_windows_table(tmp_path):
    """Each window's chromosome, first and last gene, and its genes' span."""
    path, adata = _tumour(tmp_path)
    uns, got = cx.tl.infercnv(path, inplace=False, verbose=False)
    windows, var = uns["windows"], adata.var
    assert len(windows) == got.shape[1]
    assert [windows["chromosome"].tolist().index(c) for c in uns["chr_pos"]] == list(uns["chr_pos"].values())
    chr1 = var[var["chromosome"] == "chr1"].sort_values("start")
    second = chr1.iloc[10:110]  # step 10: the second window
    row = windows.iloc[uns["chr_pos"]["chr1"] + 1]
    assert (row["first_gene"], row["last_gene"]) == (second.index[0], second.index[-1])
    assert (row["start"], row["end"]) == (second["start"].iloc[0], second["end"].max())
    chr3 = var[var["chromosome"] == "chr3"].sort_values("start")  # fewer genes than a window: one
    row = windows.iloc[uns["chr_pos"]["chr3"]]
    assert (row["first_gene"], row["last_gene"], row["end"]) == (chr3.index[0], chr3.index[-1], chr3["end"].max())


def test_profiles_do_not_change_with_the_chunk_size(tmp_path, monkeypatch):
    """The sums run over fixed blocks of cells, so any chunk size gives the
    same profiles to the bit."""
    from crispyx import cnv

    monkeypatch.setattr(cnv, "_SUM_ROWS", 25)
    path, _ = _tumour(tmp_path)
    runs = [cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], chunk_size=size, inplace=False,
                           verbose=False)[1] for size in (25, 60, 150, None)]
    for other in runs[1:]:
        assert (runs[0] != other).nnz == 0


def _batches(adata):
    """Two batches: s1 holds every B cell and about half the rest, s2 the
    other half and no B cell."""
    rng = np.random.default_rng(1)
    batch = np.where((adata.obs["type"] == "B").to_numpy() | (rng.random(adata.n_obs) < 0.5), "s1", "s2")
    adata.obs["sample"] = batch
    return batch


def test_each_batch_against_its_own_reference(tmp_path):
    """batch_key: a batch is profiled as if alone; a reference group it
    lacks (B in s2) is left out of its range."""
    path, adata = _tumour(tmp_path)
    batch = _batches(adata)
    adata.write_h5ad(path)
    uns, got = cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], batch_key="sample",
                              min_reference_cells=50, chunk_size=100, inplace=False, verbose=False)
    for name, cats in [("s1", ["T", "B"]), ("s2", "T")]:
        rows = batch == name
        chr_pos, expected = _reference_infercnv(adata[rows], reference_key="type", reference_cat=cats)
        assert uns["chr_pos"] == chr_pos
        np.testing.assert_allclose(got[rows].toarray(), expected, atol=1e-6)
    # A chunk holding s1, then s2, then s1 again.
    adata.obs["sample"] = np.where((np.arange(adata.n_obs) % 256 < 30) | (np.arange(adata.n_obs) % 256 > 220)
                                   | (adata.obs["type"] == "B").to_numpy(), "s1", "s2")
    adata.write_h5ad(path)
    _, got = cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], batch_key="sample",
                            min_reference_cells=20, inplace=False, verbose=False)
    for name, cats in [("s1", ["T", "B"]), ("s2", "T")]:
        rows = (adata.obs["sample"] == name).to_numpy()
        _, expected = _reference_infercnv(adata[rows], reference_key="type", reference_cat=cats)
        np.testing.assert_allclose(got[rows].toarray(), expected, atol=1e-6)
    with pytest.raises(ValueError, match=r"fewer than min_reference_cells=150 reference cells in s2 \(121\)$"):
        cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], batch_key="sample",
                       min_reference_cells=150, inplace=False, verbose=False)


def test_a_group_summary_per_chromosome(tmp_path):
    """tl.cnv_summary: cells, mean |profile| and each chromosome's mean,
    per group and per batch and group; in memory or from the file."""
    path, adata = _tumour(tmp_path)
    _batches(adata)
    adata.write_h5ad(path)
    cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], verbose=False)
    stored = ad.read_h5ad(path)
    X = stored.obsm["X_cnv"].toarray()
    chr_pos = dict(sorted(stored.uns["cnv"]["chr_pos"].items(), key=lambda t: t[1]))  # h5 orders keys by name
    bounds = dict(zip(chr_pos, list(chr_pos.values())[1:] + [X.shape[1]]))
    table = cx.tl.cnv_summary(path, "type", batch_key="sample")
    assert list(table.columns) == ["sample", "type", "cells", "score", "chr1", "chr2", "chr3", "chr10"]
    assert len(table) == 5  # no B in s2
    for row in table.itertuples():
        cells = X[((stored.obs["sample"] == row.sample) & (stored.obs["type"] == row.type)).to_numpy()]
        assert row.cells == len(cells)
        np.testing.assert_allclose(row.score, np.abs(cells).mean())
        for chromosome, start in chr_pos.items():
            np.testing.assert_allclose(getattr(row, chromosome), cells[:, start:bounds[chromosome]].mean())
    stored.obs["unlabelled"] = stored.obs["type"].astype(object).where(stored.obs["type"] != "B")
    assert cx.tl.cnv_summary(stored, "unlabelled")["cells"].tolist() == [150, 100]  # T, tumour; cells without one left out
    by_type = cx.tl.cnv_summary(stored, "type", min_cells=120)
    assert by_type["type"].tolist() == ["B", "T"]  # 100 tumour cells fall under min_cells
    pd.testing.assert_frame_equal(by_type, cx.tl.cnv_summary(path, "type", min_cells=120))
    assert by_type.set_index("type").loc["T", "chr2"] < 0.1 < cx.tl.cnv_summary(path, "type").set_index("type").loc[
        "tumour", "chr2"]


def test_chromosome_heatmap(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path, adata = _tumour(tmp_path)
    _batches(adata)
    adata.uns["type_colors"] = ["#ff0000", "#00ff00", "#0000ff"]  # B, T, tumour
    adata.write_h5ad(path)
    cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "B"], verbose=False)
    axes = cx.pl.chromosome_heatmap(path, "type", batch="s2", batch_key="sample", max_cells=40)
    image = axes["heatmap_ax"].images[0]
    assert image.get_array().shape[0] == 80  # 40 tumour and 40 T cells; no B in s2
    assert image.norm.vcenter == 0 and image.norm.vmin == -image.norm.vmax
    groups = axes["groupby_ax"].images[0].cmap.colors
    assert [matplotlib.colors.to_hex(c) for c in groups] == ["#00ff00", "#0000ff"]
    plt.close("all")
    # Saved with the chromosome lines; cells without a group are left out,
    # with or without max_cells.
    stored = ad.read_h5ad(path)
    stored.obs["type"] = stored.obs["type"].astype(object).where(stored.obs["type"] != "B")
    for max_cells in (None, 1000):
        axes = cx.pl.chromosome_heatmap(stored, "type", max_cells=max_cells, save=tmp_path / "cnv.png")
        assert axes["heatmap_ax"].images[0].get_array().shape[0] == 250
        lines = [sorted(seg[0][0] for seg in c.get_segments()) for c in axes["heatmap_ax"].collections]
        assert sorted(stored.uns["cnv"]["chr_pos"].values())[1:] in lines  # the chromosome boundaries
        assert (tmp_path / "cnv.png").stat().st_size > 0
        plt.close("all")
    # A range on one side of 0 (gains only) is drawn as it is.
    axes = cx.pl.chromosome_heatmap(path, "type", vmin=0, vmax=0.3)
    assert (axes["heatmap_ax"].images[0].norm.vmin, axes["heatmap_ax"].images[0].norm.vmax) == (0, 0.3)
    plt.close("all")
    # swap_axes: windows on y, so the chromosome lines are horizontal.
    axes = cx.pl.chromosome_heatmap(path, "type", swap_axes=True)
    lines = [sorted(seg[0][1] for seg in c.get_segments()) for c in axes["heatmap_ax"].collections]
    assert sorted(dict(ad.read_h5ad(path).uns["cnv"]["chr_pos"]).values())[1:] in lines
    plt.close("all")
    with pytest.raises(ValueError, match="batch needs batch_key"):
        cx.pl.chromosome_heatmap(path, "type", batch="s2")


def test_a_reference_array_and_a_layer(tmp_path):
    """reference= gives the groups' means directly; layer= reads another matrix."""
    import h5py
    from anndata.io import write_elem

    path, adata = _tumour(tmp_path)
    with h5py.File(path, "a") as h5:
        write_elem(h5.require_group("layers"), "lognorm", adata.X)
    X = adata.X.toarray()
    groups = np.vstack([X[(adata.obs["type"] == c).to_numpy()].mean(axis=0) for c in ("T", "B")])
    _, expected = _reference_infercnv(adata, reference_key="type", reference_cat=["T", "B"])
    _, got = cx.tl.infercnv(path, reference=groups, layer="lognorm", inplace=False, verbose=False)
    np.testing.assert_allclose(got.toarray(), expected, atol=1e-6)
    with pytest.raises(ValueError, match="reference has 10 genes, the file 1500"):
        cx.tl.infercnv(path, reference=np.zeros(10), inplace=False, verbose=False)


def test_missing_positions_and_reference_are_errors(tmp_path):
    path, adata = _tumour(tmp_path)
    with pytest.raises(ValueError, match=r"obs\['type'\] has no \['NK'\]"):
        cx.tl.infercnv(path, reference_key="type", reference_cat=["T", "NK"], verbose=False)
    few = tmp_path / "few.h5ad"
    adata[:50].write_h5ad(few)
    with pytest.raises(ValueError, match=r"fewer than min_reference_cells=100 reference cells in all cells \(50\)"):
        cx.tl.infercnv(few, verbose=False)
    with pytest.raises(ValueError, match="pass reference, or reference_key / batch_key, not both"):
        cx.tl.infercnv(path, reference=np.zeros(adata.n_vars), batch_key="type", verbose=False)
    cx.write_var(path, cx.load_var(path).drop(columns="start"))
    with pytest.raises(ValueError, match=r"var has no \['start'\]: add the genes' positions"):
        cx.tl.infercnv(path, verbose=False)


GTF = """\
#!genome-build GRCh38
1\tensembl\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_name "DDX11L1"; gene_biotype "lncRNA";
1\tensembl\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; transcript_id "ENST1"; gene_name "DDX11L1";
2\tensembl\tgene\t100\t900\t.\t-\t.\tgene_id "ENSG00000000002"; gene_name "TWO";
X\tensembl\tgene\t5000\t6000\t.\t+\t.\tgene_id "ENSG00000000003"; gene_name "DUP";
Y\tensembl\tgene\t7000\t8000\t.\t+\t.\tgene_id "ENSG00000000004"; gene_name "DUP";
MT\tensembl\tgene\t3307\t4262\t.\t+\t.\tgene_id "ENSG00000198888"; gene_name "MT-ND1";
KI270728.1\tensembl\tgene\t10\t20\t.\t+\t.\tgene_id "ENSG00000000005"; gene_name "SCAFFOLD";
"""

#: NCBI RefSeq: accessions for chromosomes, symbols for gene_id, no gene_name.
REFSEQ = """\
#gtf-version 2.2
NC_000001.11\tBestRefSeq\tgene\t11874\t14409\t.\t+\t.\tgene_id "DDX11L1"; transcript_id ""; db_xref "GeneID:100287102"; gbkey "Gene"; gene "DDX11L1"; gene_biotype "transcribed_pseudogene";
NC_000001.11\tBestRefSeq\texon\t11874\t12227\t.\t+\t.\tgene_id "DDX11L1"; transcript_id "NR_046018.2"; gene "DDX11L1";
NC_000023.11\tBestRefSeq\tgene\t5000\t6000\t.\t+\t.\tgene_id "DUP"; transcript_id ""; description "copy #1"; gene "DUP";
NC_000024.10\tBestRefSeq\tgene\t7000\t8000\t.\t+\t.\tgene_id "DUP_1"; transcript_id ""; gene "DUP";
NT_187361.1\tBestRefSeq\tgene\t10\t20\t.\t+\t.\tgene_id "ALT"; transcript_id ""; gene "ALT";
NC_012920.1\tRefSeq\tgene\t3307\t4262\t.\t+\t.\tgene_id "ND1"; transcript_id ""; gene "ND1";
###
"""


def _var_file(tmp_path, names, **columns):
    path = tmp_path / "x.h5ad"
    var = pd.DataFrame(columns, index=names)
    ad.AnnData(sp.csr_matrix((2, len(names)), dtype=np.float32), var=var).write_h5ad(path)
    return path


@pytest.mark.parametrize("compressed", [False, True])
def test_positions_from_an_ensembl_gtf(tmp_path, compressed):
    gtf = tmp_path / ("genes.gtf.gz" if compressed else "genes.gtf")
    with (gzip.open(gtf, "wt") if compressed else open(gtf, "w")) as handle:
        handle.write(GTF)
    path = _var_file(tmp_path, ["DDX11L1", "TWO", "DUP", "MT-ND1", "SCAFFOLD", "ABSENT"],
                     gene_ids=["ENSG00000223972", "ENSG00000000002.3", "ENSG00000000003", "ENSG00000198888",
                               "ENSG00000000005", "ENSG00000999999"])
    with pytest.warns(UserWarning, match="3 of 6 genes have no position on chr1-22, X or Y"), \
            pytest.warns(UserWarning, match="1 gene_names are on more than one GTF record; each takes its first"):
        cx.pp.genomic_position_from_gtf(path, gtf, verbose=False)
    placed = cx.load_var(path)
    assert placed["chromosome"].tolist()[:3] == ["chr1", "chr2", "chrX"]  # DUP: its first record
    assert placed["chromosome"].iloc[3:].isna().all()  # chrM, a scaffold and a gene not in the GTF
    assert placed["start"].tolist()[:3] == [11869, 100, 5000] and placed["end"].iloc[0] == 14409
    with pytest.warns(UserWarning, match="3 of 6 genes"):
        by_id = cx.pp.genomic_position_from_gtf(path, gtf, gtf_gene_id="gene_id", adata_gene_id="gene_ids",
                                                inplace=False, verbose=False)
    assert by_id["chromosome"].tolist()[:3] == ["chr1", "chr2", "chrX"] and by_id["chromosome"].iloc[3:].isna().all()


def test_positions_from_a_refseq_or_ucsc_gtf(tmp_path):
    """RefSeq's accessions and UCSC's chr names; RefSeq's gene attribute is
    the symbol."""
    gtf = tmp_path / "refseq.gtf"
    gtf.write_text(REFSEQ)
    path = _var_file(tmp_path, ["DDX11L1", "DUP", "ALT", "ND1"])
    with pytest.warns(UserWarning, match="2 of 4 genes"), pytest.warns(UserWarning, match="1 gene_names are on more"):
        placed = cx.pp.genomic_position_from_gtf(path, gtf, inplace=False, verbose=False)
    assert placed["chromosome"].tolist()[:2] == ["chr1", "chrX"] and placed["chromosome"].iloc[2:].isna().all()
    assert placed["start"].tolist()[:2] == [11874, 5000]
    with pytest.warns(UserWarning, match="2 of 4 genes"):  # RefSeq's gene_ids are unique (DUP_1)
        by_id = cx.pp.genomic_position_from_gtf(path, gtf, gtf_gene_id="gene_id", inplace=False, verbose=False)
    assert by_id["chromosome"].tolist()[:2] == ["chr1", "chrX"]
    ucsc = tmp_path / "ucsc.gtf"
    ucsc.write_text(GTF.replace("\n1\t", "\nchr1\t").replace("\n2\t", "\nchr2_KI270706v1_random\t"))
    with pytest.warns(UserWarning):
        placed = cx.pp.genomic_position_from_gtf(path, ucsc, inplace=False, verbose=False)
    assert placed.loc["DDX11L1", "chromosome"] == "chr1"
