"""Copy-number profiles inferred from expression along the genome, streamed:
infercnvpy's method (:func:`infercnvpy.tl.infercnv`) and the gene positions
it needs, read from a GTF file."""

from __future__ import annotations

import csv
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _messages
from ._checkpoint import _create_progress_context
from ._memory import _resolve_memory_limit_bytes
from .data import (
    AnnData,
    _append_sparse_block,
    _close_wrappers_on,
    iter_matrix_chunks,
    load_obs,
    load_var,
    read_backed,
    resolve_data_path,
    resolve_path_to_update,
    write_uns_dict_to_h5ad,
    write_var,
)

#: GTF lines parsed per block.
_GTF_BLOCK_LINES = 200_000


def _gtf_genes(gtf_file: str | Path) -> pd.DataFrame:
    """The ``gene`` records of a GTF file (plain or compressed, by its
    suffix), read a block of lines at a time: ``chromosome``, ``start``,
    ``end``, ``gene_id`` and ``gene_name`` (NA when a record has none)."""
    reader = pd.read_csv(
        gtf_file, sep="\t", header=None, comment="#", usecols=[0, 2, 3, 4, 8],
        names=["chromosome", "feature", "start", "end", "attributes"], dtype=str, quoting=csv.QUOTE_NONE,
        chunksize=_GTF_BLOCK_LINES,
    )
    parts = []
    for block in reader:
        block = block[block["feature"] == "gene"]
        attributes = block["attributes"]
        parts.append(pd.DataFrame({
            "chromosome": block["chromosome"],
            "start": block["start"].astype(np.int64),
            "end": block["end"].astype(np.int64),
            "gene_id": attributes.str.extract(r'gene_id "([^"]*)"', expand=False),
            "gene_name": attributes.str.extract(r'gene_name "([^"]*)"', expand=False),
        }))
    genes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["chromosome", "start", "end", "gene_id", "gene_name"]
    )
    if genes.empty:
        raise ValueError(f"pp.genomic_position_from_gtf: {gtf_file} has no 'gene' records")
    return genes


def genomic_position_from_gtf(
    data: str | Path | AnnData | ad.AnnData,
    gtf_file: str | Path,
    *,
    gtf_gene_id: Literal["gene_id", "gene_name"] = "gene_name",
    adata_gene_id: str | None = None,
    inplace: bool = True,
    verbose: int | bool = True,
) -> pd.DataFrame | None:
    """Each gene's ``chromosome``, ``start`` and ``end`` from a GTF file, as
    :func:`infercnvpy.io.genomic_position_from_gtf` adds them to ``var``.

    The GTF's ``gene`` records are read a block of lines at a time (a
    GENCODE file need not fit in memory, and may be gzipped) and matched to
    ``var`` on ``gtf_gene_id``; only ``var`` is read from the h5ad file. A
    key on several records (a symbol on two loci) is left unplaced, as
    infercnvpy leaves it; Ensembl versions (``.12``) are ignored on both
    sides when matching IDs. Chromosomes are named ``chr1``, ... (``chr``
    is prefixed when the GTF has none, and Ensembl's ``MT`` becomes
    ``chrM``), the names :func:`infercnv` reads.

    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData
        (``inplace=False`` only).
    gtf_file
        GTF annotation of the genome the counts were aligned to.
    gtf_gene_id
        GTF attribute matched to ``var``: ``"gene_name"`` (symbols) or
        ``"gene_id"`` (Ensembl IDs).
    adata_gene_id
        ``var`` column matched to it; ``var_names`` when ``None``.
    inplace
        Write the three columns to the file's ``var`` and return ``None``;
        ``False`` returns them.
    verbose
        Print how many genes were placed.

    Returns
    -------
    pd.DataFrame or None
        ``chromosome``, ``start`` and ``end`` indexed like ``var`` (NA for a
        gene not placed), when ``inplace=False``.
    """
    name = "pp.genomic_position_from_gtf"
    path = resolve_path_to_update(data, "var") if inplace else resolve_data_path(data)
    _messages.print_reading(verbose, name, path)
    var = load_var(path)
    keys = pd.Index((var.index if adata_gene_id is None else var[adata_gene_id]).astype(str))
    genes = _gtf_genes(gtf_file).drop_duplicates()
    if gtf_gene_id == "gene_id":
        keys = keys.str.replace(r"\.\d+$", "", regex=True)
        genes["gene_id"] = genes["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    repeated = genes[gtf_gene_id].duplicated(keep=False)
    if repeated.any():
        _messages.warn(name, f"{genes.loc[repeated, gtf_gene_id].nunique():,} {gtf_gene_id}s are on more than one "
                             "GTF record and are left unplaced")
    table = genes.loc[~repeated & genes[gtf_gene_id].notna()].set_index(gtf_gene_id)
    if not table["chromosome"].str.startswith("chr").any():
        table["chromosome"] = ("chr" + table["chromosome"]).replace("chrMT", "chrM")
    placed = table.reindex(keys)
    positions = pd.DataFrame({
        "chromosome": placed["chromosome"].to_numpy(dtype=object),
        "start": pd.array(placed["start"].to_numpy(), dtype="Int64"),
        "end": pd.array(placed["end"].to_numpy(), dtype="Int64"),
    }, index=var.index)
    n_placed = int(positions["chromosome"].notna().sum())
    if n_placed < len(var):
        _messages.warn(name, f"{len(var) - n_placed:,} of {len(var):,} genes have no position in {gtf_file}")
    _messages.print_done(verbose, name, f"{n_placed:,}/{len(var):,} genes placed")
    if not inplace:
        return positions
    write_var(path, var.assign(**positions))
    return None


def _natural_sort(values) -> list[str]:
    """``chr2`` before ``chr10``, as infercnvpy orders chromosomes."""
    def key(text):
        return [int(part) if part.isdigit() else part.lower() for part in re.split("([0-9]+)", text)]
    return sorted(values, key=key)


def _windows(var: pd.DataFrame, window_size: int, step: int) -> tuple[np.ndarray, sp.csr_matrix, dict[str, int]]:
    """infercnvpy's running means as one sparse product. Returns the rows of
    ``var`` chromosome by chromosome in genomic order; a genes (in that order)
    x windows matrix of pyramid weights, each window ``window_size`` genes and
    every ``step``-th of a chromosome's windows kept (one flat window over a
    chromosome with no more genes than that); each chromosome's first window.
    Only chromosomes named ``chr*`` other than ``chrM`` are kept, as there."""
    chromosomes = _natural_sort([c for c in var["chromosome"].astype(str).unique() if c.startswith("chr") and c != "chrM"])
    if not chromosomes:
        raise ValueError("tl.infercnv: no gene is placed on a chromosome named 'chr*'")
    order, rows, cols, weights, chr_pos = [], [], [], [], {}
    n_genes = n_windows = 0
    for chromosome in chromosomes:
        genes = var.loc[var["chromosome"] == chromosome].sort_values("start").index.values
        order.append(var.index.get_indexer(genes))
        length = len(genes)
        if window_size < length:
            r = np.arange(1, window_size + 1)
            pyramid = np.minimum(r, r[::-1])
            members = np.arange(0, length - window_size + 1, step)[:, None] + np.arange(window_size)
            weight = np.broadcast_to(pyramid / pyramid.sum(), members.shape)
        else:
            members = np.arange(length)[None, :]
            weight = np.full(members.shape, 1.0 / length)
        rows.append(n_genes + members.ravel())
        cols.append(np.repeat(n_windows + np.arange(len(members)), members.shape[1]))
        weights.append(weight.ravel())
        chr_pos[chromosome] = n_windows
        n_genes += length
        n_windows += len(members)
    matrix = sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n_genes, n_windows),
    )
    return np.concatenate(order), matrix, chr_pos


def infercnv(
    data: str | Path | AnnData | ad.AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    reference: np.ndarray | None = None,
    lfc_clip: float = 3,
    window_size: int = 100,
    step: int = 10,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    layer: str | None = None,
    key_added: str = "cnv",
    chunk_size: int | None = None,
    memory_limit_gb: float | None = None,
    inplace: bool = True,
    verbose: int | bool = True,
) -> tuple[dict[str, int], sp.csr_matrix] | None:
    """Copy-number profiles of cells from their expression along the genome:
    infercnvpy's method (:func:`infercnvpy.tl.infercnv`), streamed.

    Each cell's log-normalised expression less the reference cells' mean
    (between the lowest and highest reference group's mean when there are
    several, else 0) is clipped at ``lfc_clip``, averaged over windows of
    ``window_size`` genes in genomic order (pyramid weights, every
    ``step``-th window, each chromosome on its own), centred on the cell's
    median window, and values under ``dynamic_threshold`` standard
    deviations are set to 0. Same arguments and defaults as infercnvpy.

    Memory holds a chunk of cells densely over the placed genes, never
    ``X``: the reference means take one pass over the matrix, the standard
    deviation a second, and the profiles a third, which writes them a chunk
    at a time. infercnvpy takes the standard deviation within each chunk of
    5,000 cells, so its threshold, and which values it zeroes, change with
    ``chunksize``; crispyx takes it over all cells, and equals infercnvpy
    when every cell is in one of its chunks. A CSC file is read whole for
    every chunk of cells: convert it first (:func:`crispyx.pp.convert_to_csr`).

    Parameters
    ----------
    data
        Path to an h5ad file of log-normalised expression whose ``var`` has
        ``chromosome``, ``start`` and ``end``
        (:func:`genomic_position_from_gtf`), or a crispyx AnnData /
        backed anndata.AnnData (``inplace=False`` only).
    reference_key, reference_cat
        ``obs`` column and its value(s) of the normal cells; every cell's
        mean when ``None``.
    reference
        Reference expression instead, ``(n_vars,)`` or ``(groups, n_vars)``.
    lfc_clip
        Clip the differences from the reference at this value.
    window_size
        Genes in a window.
    step
        Keep every ``step``-th window.
    dynamic_threshold
        Set values within this many standard deviations (of all profiles) of
        0 to 0; ``None`` keeps them.
    exclude_chromosomes
        Chromosomes left out (by default the sex chromosomes).
    layer
        Use ``layers[layer]`` instead of ``X``.
    key_added
        The profiles go to ``obsm["X_<key_added>"]`` (cells x windows, CSR)
        and each chromosome's first window to ``uns[key_added]["chr_pos"]``.
    chunk_size
        Cells per chunk; from ``memory_limit_gb`` when ``None``.
    memory_limit_gb
        Memory limit of the process in GB; ``None`` uses the memory detected
        as available.
    inplace
        Write the result to the file; ``False`` returns it.
    verbose
        Print progress.

    Returns
    -------
    (chr_pos, profiles) or None
        Each chromosome's first window and the CSR profiles, when
        ``inplace=False``, as infercnvpy returns them (without its per-gene
        values).
    """
    name = "tl.infercnv"
    path = resolve_path_to_update(data, "obsm") if inplace else resolve_data_path(data)
    _messages.print_reading(verbose, name, path)
    var = load_var(path)
    if var.index.has_duplicates:
        raise ValueError(f"{name}: var_names are not unique")
    missing = {"chromosome", "start", "end"} - set(var.columns)
    if missing:
        raise ValueError(f"{name}: var has no {sorted(missing)}: add the genes' positions "
                         "(pp.genomic_position_from_gtf)")
    unplaced = var["chromosome"].isna()
    if unplaced.any():
        _messages.vprint(verbose, name, f"{int(unplaced.sum()):,} genes without a position are left out")
    keep = ~unplaced
    if exclude_chromosomes is not None:
        keep &= ~var["chromosome"].isin(exclude_chromosomes)
    positioned = var.loc[keep, ["chromosome", "start", "end"]]
    order, smooth, chr_pos = _windows(positioned, window_size, step)
    columns = np.flatnonzero(keep.to_numpy())[order]  # var positions, in genomic order
    smooth_t = smooth.T.tocsr()

    scratch = path.with_name(f".{path.name}.X_{key_added}.{os.getpid()}.partial")
    parts = []
    try:
        backed = read_backed(path)
        try:
            matrix = backed.X if layer is None else backed.layers[layer]
            n_obs, n_vars = matrix.shape
            if chunk_size is None:
                # The chunk densely over the placed genes, about four times
                # (the values, their differences, clipped, and the indexing).
                budget = _resolve_memory_limit_bytes(memory_limit_gb)
                chunk_size = int(min(n_obs, 5000, max(1, 0.25 * budget // (32 * len(columns)))))

            def blocks(label):
                with _create_progress_context(-(-n_obs // chunk_size), label, verbose, unit="chunk") as bar:
                    for slc, block in iter_matrix_chunks(
                        backed, axis=0, matrix=matrix, chunk_size=chunk_size, convert_to_dense=False,
                    ):
                        bar.update(1)
                        block = block[:, columns]
                        yield slc, (block.toarray() if sp.issparse(block) else np.asarray(block)).astype(np.float64)

            if reference is None:
                if reference_key is None or reference_cat is None:
                    _messages.vprint(verbose, name, "reference: the mean of every cell (pass reference_key and "
                                                    "reference_cat to use normal cells)")
                    groups = np.zeros(n_obs, dtype=np.int64)
                    n_groups = 1
                else:
                    labels = load_obs(path)[reference_key]
                    categories = list(np.atleast_1d(np.asarray(reference_cat, dtype=object)))
                    absent = [c for c in categories if not (labels == c).any()]
                    if absent:
                        raise ValueError(f"{name}: obs[{reference_key!r}] has no {absent}")
                    groups = np.full(n_obs, -1, dtype=np.int64)
                    for k, category in enumerate(categories):
                        groups[(labels == category).to_numpy()] = k
                    n_groups = len(categories)
                sums = np.zeros((n_groups, len(columns)))
                for slc, x in blocks("reference"):
                    within = np.flatnonzero(groups[slc] >= 0)
                    member = sp.csr_matrix(
                        (np.ones(within.size), (groups[slc][within], within)), shape=(n_groups, x.shape[0]),
                    )
                    sums += member @ x
                ref = sums / np.bincount(groups[groups >= 0], minlength=n_groups)[:, None]
            else:
                ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
                if ref.shape[1] != n_vars:
                    raise ValueError(f"{name}: reference has {ref.shape[1]} genes, the file {n_vars}")
                ref = ref[:, columns]
            low, high = ref.min(axis=0), ref.max(axis=0)

            def profiles(x):
                if len(ref) == 1:
                    x = x - ref[0]
                else:  # between the reference groups' means counts as no change
                    x = np.where(x > high, x - high, np.where(x < low, x - low, 0.0))
                np.clip(x, -lfc_clip, lfc_clip, out=x)
                smoothed = (smooth_t @ x.T).T
                return smoothed - np.median(smoothed, axis=1, keepdims=True)

            threshold = None
            if dynamic_threshold is not None:
                # The standard deviation of every profile value: the chunks'
                # means and sums of squares combined (Chan et al.).
                count, mean, m2 = 0, 0.0, 0.0
                for _, x in blocks("spread"):
                    values = profiles(x)
                    n, m = values.size, values.mean()
                    delta = m - mean
                    mean += delta * n / (count + n)
                    m2 += ((values - m) ** 2).sum() + delta ** 2 * count * n / (count + n)
                    count += n
                threshold = dynamic_threshold * np.sqrt(m2 / count)

            # Profiles go to a scratch file beside the input, copied in once
            # complete: a run killed part way leaves the file as it was.
            with h5py.File(scratch, "w") if inplace else nullcontext() as out:
                for _, x in blocks("profiles"):
                    values = profiles(x)
                    if threshold is not None:
                        values[np.abs(values) < threshold] = 0
                    block = sp.csr_matrix(values.astype(np.float32))
                    if inplace:
                        _append_sparse_block(out, block, f"X_{key_added}")
                    else:
                        parts.append(block)
        finally:
            backed.file.close()
        chr_pos = {chromosome: int(start) for chromosome, start in chr_pos.items()}
        _messages.print_done(verbose, name, f"{n_obs:,} cells x {smooth.shape[1]:,} windows on {len(chr_pos)} chromosomes")
        if not inplace:
            return chr_pos, sp.vstack(parts, format="csr")
        _close_wrappers_on(path)
        with h5py.File(path, "r+") as dst, h5py.File(scratch, "r") as tmp:
            obsm = dst.require_group("obsm")
            if f"X_{key_added}" in obsm:
                del obsm[f"X_{key_added}"]
            obsm.copy(tmp[f"X_{key_added}"], f"X_{key_added}")
            write_uns_dict_to_h5ad(dst, key_added, {"chr_pos": chr_pos})
    finally:
        scratch.unlink(missing_ok=True)
    return None
