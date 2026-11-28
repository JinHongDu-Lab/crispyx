"""Gene-set scores of cells, streamed: Seurat's method as Scanpy's
:func:`scanpy.tl.score_genes` implements it, without loading ``X``."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _messages
from ._checkpoint import _create_progress_context
from .data import (
    AnnData,
    _detect_backed_sparse_format,
    _expression_source,
    _open_matrix,
    calculate_optimal_chunk_size,
    calculate_optimal_gene_chunk_size,
    chunk_starts,
    iter_matrix_chunks,
    load_obs,
    resolve_data_path,
    resolve_path_to_update,
    write_obs,
)


def _control_genes(
    gene_list, gene_pool, means: pd.Series, *, ctrl_as_ref: bool, ctrl_size: int, n_bins: int, random_state,
) -> pd.Index:
    """Scanpy's reference genes for ``gene_list`` (its ``_score_genes_bins``):
    the pool binned by mean expression, ``ctrl_size`` genes drawn from each bin
    that holds a listed gene. ``random_state`` seeds the draw as Scanpy's
    ``np.random.seed`` does, without touching NumPy's global state."""
    rng = None if random_state is None else np.random.RandomState(random_state)
    obs_avg = means.loc[gene_pool]
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    outside = gene_list.difference(obs_cut.index, sort=False)
    if len(outside):
        raise ValueError(f"tl.score_genes: listed genes must be in gene_pool; not there: {', '.join(map(str, outside[:5]))}"
                         + (", ..." if len(outside) > 5 else ""))
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gene_list)
    control = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if len(r_genes) == 0:
            _messages.warn("tl.score_genes", f"no control genes for bin {cut}: the gene pool ({len(gene_pool)} genes) is too small")
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size, random_state=rng).index
        if ctrl_as_ref:
            r_genes = r_genes.difference(gene_list)
        control = control.union(r_genes)
    if len(control) == 0:
        raise RuntimeError(
            "tl.score_genes: no control genes found in any bin" + (": try ctrl_as_ref=False" if ctrl_as_ref else "")
        )
    return control


def _gene_set_scores(
    data, sets: dict[str, tuple[Sequence[str], int]], *, ctrl_as_ref: bool, gene_pool, n_bins: int, random_state,
    use_raw: bool | None, layer: str | None, chunk_size: int | None, name: str, verbose: int | bool, inplace: bool,
) -> tuple[Path, pd.DataFrame]:
    """Scores of every ``{score_name: (gene_list, ctrl_size)}`` in two passes
    over the matrix, whatever the number of sets: the genes' means, then each
    cell's mean over every set's genes and reference genes."""
    path = resolve_path_to_update(data, "obs") if inplace else resolve_data_path(data)
    _messages.print_reading(verbose, name, path)
    key, var = _expression_source(path, use_raw=use_raw, layer=layer, name=name)
    var_names = pd.Index(var.index.astype(str))
    if var_names.has_duplicates:
        raise ValueError(f"{name}: var_names are not unique; pp.collapse_genes or var_names_make_unique first")
    pool = var_names.astype("string") if gene_pool is None else pd.Index(gene_pool, dtype="string").intersection(var_names)
    if len(pool) == 0:
        raise ValueError(f"{name}: no gene of gene_pool is in var_names")
    lists = {}
    for score_name, (gene_list, _) in sets.items():
        genes = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
        missing = genes.difference(var_names, sort=False)
        genes = genes.intersection(var_names)
        if len(missing):
            _messages.warn(name, f"{len(missing)} genes of {score_name!r} are not in var_names and are ignored: "
                                 f"{', '.join(map(str, missing[:5]))}{', ...' if len(missing) > 5 else ''}")
        if len(genes) == 0:
            raise ValueError(f"{name}: no gene of {score_name!r} is in var_names")
        lists[score_name] = genes

    with _open_matrix(path, key) as matrix:
        n_obs, n_vars = matrix.shape
        by_gene = _detect_backed_sparse_format(matrix) == "csc"
        if chunk_size is None:
            chunk_size = (calculate_optimal_gene_chunk_size if by_gene else calculate_optimal_chunk_size)(n_obs, n_vars)

        def blocks(label: str, selected: np.ndarray | None = None):
            n_chunks = len(chunk_starts(n_vars if by_gene else n_obs, chunk_size, selected=selected))
            with _create_progress_context(n_chunks, label, verbose, unit="chunk") as bar:
                for slc, block in iter_matrix_chunks(
                    None, axis=int(by_gene), matrix=matrix, chunk_size=chunk_size, convert_to_dense=False,
                    selected=selected,
                ):
                    bar.update(1)
                    yield ((slice(None), slc) if by_gene else (slc, slice(None))), sp.csr_matrix(block, dtype=np.float64)

        totals = np.zeros(n_vars)
        for (_, genes), block in blocks("gene means"):
            totals[genes] += np.asarray(block.sum(axis=0)).ravel()
        means = pd.Series(totals / n_obs, index=var_names.astype("string"))

        # One column per set and per reference: each holds 1/size on its genes.
        columns, weights = [], []
        for score_name, (_, ctrl_size) in sets.items():
            control = _control_genes(
                lists[score_name], pool, means, ctrl_as_ref=ctrl_as_ref, ctrl_size=ctrl_size, n_bins=n_bins,
                random_state=random_state,
            )
            for genes in (lists[score_name], control):
                column = np.zeros(n_vars)
                column[var_names.get_indexer(genes)] = 1.0 / len(genes)
                weights.append(column)
            columns.append(score_name)
        weights = np.column_stack(weights)
        # On CSC, only the gene blocks that hold a scored gene are read.
        selected = (weights != 0).any(axis=1) if by_gene else None
        cell_means = np.zeros((n_obs, weights.shape[1]))
        for (cells, genes), block in blocks("cell means", selected):
            cell_means[cells] += block @ weights[genes]
    scores = pd.DataFrame(cell_means[:, 0::2] - cell_means[:, 1::2], columns=columns, index=load_obs(path).index)
    return path, scores


def score_genes(
    data: str | Path | AnnData | ad.AnnData,
    gene_list: Sequence[str],
    *,
    ctrl_as_ref: bool = True,
    ctrl_size: int = 50,
    gene_pool: Sequence[str] | None = None,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int | None = 0,
    use_raw: bool | None = None,
    layer: str | None = None,
    chunk_size: int | None = None,
    inplace: bool = True,
    verbose: int | bool = True,
) -> pd.Series | None:
    """Score each cell on a gene set: its mean expression of the set less its
    mean over reference genes of matched expression (Seurat's method, as
    :func:`scanpy.tl.score_genes`).

    The pool's genes are binned by mean expression into ``n_bins`` bins, and
    ``ctrl_size`` reference genes are drawn from each bin that holds a listed
    gene. Same arguments, defaults and scores as Scanpy's, read in two passes
    over the matrix in its stored layout: the genes' means, then each cell's
    mean over the listed and reference genes (on CSC, only the gene blocks
    that hold them). Memory holds a chunk and O(cells + genes) arrays.

    The bins come from a ranking of the genes' means, which crispyx sums in
    float64 chunk by chunk. Scanpy sums float32 data in float32, in one pass,
    so on float32 data, or with genes whose means agree to rounding error, a
    gene at a bin's edge can fall in the neighbouring bin there and the
    reference genes drawn after it differ.

    Parameters
    ----------
    data
        Path to an h5ad file of log-normalised expression, or a crispyx
        AnnData / backed anndata.AnnData (``inplace=False`` only).
    gene_list
        Genes of the set, as in ``var_names``; genes not there are ignored
        with a warning.
    ctrl_as_ref
        Allow listed genes to be drawn as reference genes, as Scanpy does by
        default (they are then removed from the reference).
    ctrl_size
        Reference genes drawn from each bin.
    gene_pool
        Genes to draw reference genes from; all genes when ``None``.
    n_bins
        Number of expression bins.
    score_name
        ``obs`` column for the score.
    random_state
        Seed of the draw, as in Scanpy (which seeds NumPy's global state;
        crispyx leaves it untouched).
    use_raw
        Score ``raw`` (Scanpy's default when the file has one) instead of
        ``X``; ``None`` uses ``raw`` when there is one and no ``layer``.
    layer
        Use ``layers[layer]`` instead of ``X``.
    chunk_size
        Cells (CSR, dense) or genes (CSC) per chunk; from the file's shape
        when ``None``.
    inplace
        Write the score to the file's ``obs`` and return ``None``; ``False``
        returns it.
    verbose
        Print progress.

    Returns
    -------
    pd.Series or None
        The score, indexed like ``obs``, when ``inplace=False``.
    """
    path, scores = _gene_set_scores(
        data, {score_name: (gene_list, ctrl_size)}, ctrl_as_ref=ctrl_as_ref, gene_pool=gene_pool, n_bins=n_bins,
        random_state=random_state, use_raw=use_raw, layer=layer, chunk_size=chunk_size, name="tl.score_genes",
        verbose=verbose, inplace=inplace,
    )
    _messages.print_done(verbose, "tl.score_genes", f"obs[{score_name!r}] for {len(scores):,} cells")
    if not inplace:
        return scores[score_name]
    write_obs(path, load_obs(path).assign(**{score_name: scores[score_name]}))
    return None


def score_genes_cell_cycle(
    data: str | Path | AnnData | ad.AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    ctrl_as_ref: bool = True,
    gene_pool: Sequence[str] | None = None,
    n_bins: int = 25,
    random_state: int | None = 0,
    use_raw: bool | None = None,
    layer: str | None = None,
    chunk_size: int | None = None,
    inplace: bool = True,
    verbose: int | bool = True,
) -> pd.DataFrame | None:
    """Cell-cycle scores and phase, as :func:`scanpy.tl.score_genes_cell_cycle`.

    ``S_score`` and ``G2M_score`` are :func:`score_genes` scores of the two
    sets, each with ``min(len(s_genes), len(g2m_genes))`` reference genes per
    bin and the draw seeded by ``random_state``, as Scanpy scores them; both
    are computed in the same two passes over the matrix. ``phase`` is
    ``G2M`` where that score is the higher, ``S`` otherwise, and ``G1`` where
    both are negative.

    Parameters
    ----------
    data
        Path to an h5ad file of log-normalised expression, or a crispyx
        AnnData / backed anndata.AnnData (``inplace=False`` only).
    s_genes, g2m_genes
        Genes of the S and G2/M phases (e.g. Tirosh et al.'s lists).
    ctrl_as_ref, gene_pool, n_bins, random_state, use_raw, layer, chunk_size, verbose
        As in :func:`score_genes`.
    inplace
        Write ``S_score``, ``G2M_score`` and ``phase`` to the file's ``obs``
        and return ``None``; ``False`` returns them.

    Returns
    -------
    pd.DataFrame or None
        ``S_score``, ``G2M_score`` and ``phase``, indexed like ``obs``, when
        ``inplace=False``.
    """
    ctrl_size = min(len(s_genes), len(g2m_genes))
    path, scores = _gene_set_scores(
        data, {"S_score": (s_genes, ctrl_size), "G2M_score": (g2m_genes, ctrl_size)}, ctrl_as_ref=ctrl_as_ref,
        gene_pool=gene_pool, n_bins=n_bins, random_state=random_state, use_raw=use_raw, layer=layer,
        chunk_size=chunk_size, name="tl.score_genes_cell_cycle", verbose=verbose, inplace=inplace,
    )
    phase = pd.Series("S", index=scores.index)
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"
    phase[(scores < 0).all(axis=1)] = "G1"
    scores["phase"] = phase
    _messages.print_done(
        verbose, "tl.score_genes_cell_cycle",
        ", ".join(f"{count:,} {label}" for label, count in phase.value_counts().sort_index().items()),
    )
    if not inplace:
        return scores
    write_obs(path, load_obs(path).assign(**scores))
    return None
