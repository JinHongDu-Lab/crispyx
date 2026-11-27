"""tl.infercnv and pp.genomic_position_from_gtf: infercnvpy's method, streamed."""

from __future__ import annotations

import gzip
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
    got_pos, got = cx.tl.infercnv(path, inplace=False, chunk_size=73, verbose=False, **kwargs)
    assert got_pos == chr_pos and list(got_pos) == ["chr1", "chr2", "chr3", "chr10"]
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
    cx.write_var(path, cx.load_var(path).drop(columns="start"))
    with pytest.raises(ValueError, match=r"var has no \['start'\]: add the genes' positions"):
        cx.tl.infercnv(path, verbose=False)


GTF = """\
#!genome-build GRCh38
1\tensembl\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_name "DDX11L1"; gene_biotype "lncRNA";
1\tensembl\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; transcript_id "ENST1"; gene_name "DDX11L1";
2\tensembl\tgene\t100\t900\t.\t-\t.\tgene_id "ENSG00000000002"; gene_name "TWO";
X\tensembl\tgene\t5000\t6000\t.\t+\t.\tgene_id "ENSG00000000003"; gene_name "DUP";
Y\tensembl\tgene\t5000\t6000\t.\t+\t.\tgene_id "ENSG00000000004"; gene_name "DUP";
MT\tensembl\tgene\t3307\t4262\t.\t+\t.\tgene_id "ENSG00000198888"; gene_name "MT-ND1";
"""


@pytest.mark.parametrize("compressed", [False, True])
def test_positions_from_a_gtf(tmp_path, compressed):
    gtf = tmp_path / ("genes.gtf.gz" if compressed else "genes.gtf")
    with (gzip.open(gtf, "wt") if compressed else open(gtf, "w")) as handle:
        handle.write(GTF)
    var = pd.DataFrame({"gene_ids": ["ENSG00000223972", "ENSG00000000002.3", "ENSG00000000003", "ENSG00000198888",
                                     "ENSG00000999999"]},
                       index=["DDX11L1", "TWO", "DUP", "MT-ND1", "ABSENT"])
    path = tmp_path / "x.h5ad"
    ad.AnnData(sp.csr_matrix((2, 5), dtype=np.float32), var=var).write_h5ad(path)
    with pytest.warns(UserWarning, match="1 gene_names are on more than one GTF record"):
        cx.pp.genomic_position_from_gtf(path, gtf, verbose=False)
    placed = cx.load_var(path)
    assert placed["chromosome"].tolist()[:2] == ["chr1", "chr2"] and placed["chromosome"].iloc[3] == "chrM"
    assert placed["chromosome"].iloc[[2, 4]].isna().all()
    assert placed["start"].tolist()[:2] == [11869, 100] and placed["end"].iloc[0] == 14409
    by_id = cx.pp.genomic_position_from_gtf(path, gtf, gtf_gene_id="gene_id", adata_gene_id="gene_ids",
                                            inplace=False, verbose=False)
    assert by_id["chromosome"].tolist()[:4] == ["chr1", "chr2", "chrX", "chrM"] and pd.isna(by_id["chromosome"].iloc[4])
