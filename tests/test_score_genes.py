"""tl.score_genes / tl.score_genes_cell_cycle: Scanpy's scores, streamed."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata.io import write_elem

import crispyx as cx

S_GENES = [f"g{j}" for j in range(0, 30, 3)]
G2M_GENES = [f"g{j}" for j in range(100, 130)]


def _lognorm(tmp_path, fmt: str = "csr"):
    """Log-normalised float64 expression without ties between gene means
    (cells of continuous depth): bins then do not depend on summation order,
    which differs between a chunked and a one-pass sum."""
    rng = np.random.default_rng(0)
    n, g = 600, 800
    counts = rng.poisson(rng.gamma(0.5, 1.0, g), (n, g))
    X = np.log1p(counts * rng.uniform(0.5, 1.5, (n, 1)))
    adata = ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(g)]))
    stored = adata.copy()
    if fmt != "csr":
        stored.X = X if fmt == "dense" else sp.csc_matrix(X)
    path = tmp_path / f"lognorm_{fmt}.h5ad"
    stored.write_h5ad(path)
    return path, adata


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
@pytest.mark.parametrize("ctrl_as_ref", [True, False])
def test_score_genes_matches_scanpy(tmp_path, fmt, ctrl_as_ref):
    path, adata = _lognorm(tmp_path, fmt)
    genes = S_GENES + ["absent"]
    expected = adata.copy()
    sc.tl.score_genes(expected, genes, ctrl_size=30, ctrl_as_ref=ctrl_as_ref, random_state=3)
    with pytest.warns(UserWarning, match="1 genes of 'score' are not in var_names"):
        cx.tl.score_genes(path, genes, ctrl_size=30, ctrl_as_ref=ctrl_as_ref, random_state=3, chunk_size=97,
                          verbose=False)
    np.testing.assert_allclose(cx.load_obs(path)["score"], expected.obs["score"], rtol=1e-12, atol=1e-12)


def test_score_genes_with_a_gene_pool_and_a_layer(tmp_path):
    path, adata = _lognorm(tmp_path)
    with h5py.File(path, "a") as h5:
        write_elem(h5.require_group("layers"), "lognorm", adata.X)
    pool = [f"g{j}" for j in range(0, 800, 3)]
    expected = adata.copy()
    sc.tl.score_genes(expected, S_GENES, gene_pool=pool, n_bins=10, score_name="s")
    got = cx.tl.score_genes(path, S_GENES, gene_pool=pool, n_bins=10, score_name="s", layer="lognorm",
                            inplace=False, verbose=False)
    np.testing.assert_allclose(got, expected.obs["s"], rtol=1e-12, atol=1e-12)
    assert "s" not in cx.load_obs(path)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_cell_cycle_matches_scanpy(tmp_path, fmt):
    path, adata = _lognorm(tmp_path, fmt)
    expected = adata.copy()
    sc.tl.score_genes_cell_cycle(expected, s_genes=S_GENES, g2m_genes=G2M_GENES)
    cx.tl.score_genes_cell_cycle(path, s_genes=S_GENES, g2m_genes=G2M_GENES, chunk_size=50, verbose=False)
    obs = cx.load_obs(path)
    np.testing.assert_allclose(obs[["S_score", "G2M_score"]], expected.obs[["S_score", "G2M_score"]], rtol=1e-12,
                               atol=1e-12)
    assert (obs["phase"].astype(str).to_numpy() == expected.obs["phase"].to_numpy()).all()
    assert set(obs["phase"]) == {"G1", "S", "G2M"}


def test_csc_reads_only_the_gene_blocks_with_scored_genes(tmp_path, monkeypatch):
    from crispyx import score

    path, _ = _lognorm(tmp_path, "csc")
    seen = []
    real = score.iter_matrix_chunks

    def spy(*args, selected=None, **kwargs):
        for slc, block in real(*args, selected=selected, **kwargs):
            seen.append((selected is not None, slc))
            yield slc, block

    monkeypatch.setattr(score, "iter_matrix_chunks", spy)
    cx.tl.score_genes(path, S_GENES, ctrl_size=5, n_bins=5, chunk_size=10, inplace=False, verbose=False)
    first, second = [s for sel, s in seen if not sel], [s for sel, s in seen if sel]
    assert len(first) == 80 and 0 < len(second) < 80


def test_no_listed_gene_is_an_error(tmp_path):
    path, _ = _lognorm(tmp_path)
    with pytest.raises(ValueError, match="no gene of 'score' is in var_names"), pytest.warns(UserWarning):
        cx.tl.score_genes(path, ["absent"], verbose=False)


def test_a_listed_gene_outside_the_pool_is_an_error(tmp_path):
    path, _ = _lognorm(tmp_path)
    with pytest.raises(ValueError, match="listed genes must be in gene_pool; not there: g3, g9, g15"):
        cx.tl.score_genes(path, S_GENES, gene_pool=[f"g{j}" for j in range(0, 800, 2)], verbose=False)


def test_raw_is_scored_when_the_file_has_one(tmp_path):
    """Scanpy scores raw by default (use_raw=None) when there is one; a file
    as Scanpy leaves it (raw log-normalised, X scaled) gives its scores."""
    path, adata = _lognorm(tmp_path)
    adata.raw = adata
    adata.X = adata.X.toarray() * 2.0 - 1.0  # X no longer the log-normalised values
    adata.write_h5ad(path)
    expected = adata.copy()
    sc.tl.score_genes(expected, S_GENES, ctrl_size=30, random_state=3)
    got = cx.tl.score_genes(path, S_GENES, ctrl_size=30, random_state=3, inplace=False, verbose=False)
    np.testing.assert_allclose(got.to_numpy(), expected.obs["score"].to_numpy(), atol=1e-12)
    on_x = adata.copy()
    sc.tl.score_genes(on_x, S_GENES, ctrl_size=30, random_state=3, use_raw=False)
    got = cx.tl.score_genes(path, S_GENES, ctrl_size=30, random_state=3, use_raw=False, inplace=False, verbose=False)
    np.testing.assert_allclose(got.to_numpy(), on_x.obs["score"].to_numpy(), atol=1e-12)
