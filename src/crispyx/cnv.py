"""Copy-number profiles inferred from expression along the genome, streamed:
infercnvpy's method (:func:`infercnvpy.tl.infercnv`), the gene positions it
needs, read from a GTF file, and each group's mean profile per chromosome."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Literal, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem, sparse_dataset, write_elem

from . import _messages
from ._checkpoint import _create_progress_context
from ._disk import warn_if_disk_space_low
from ._memory import _EvictEvery, _resolve_memory_limit_bytes
from .data import (
    AnnData,
    _append_sparse_block,
    _check_results_writable,
    _close_wrappers_on,
    _results_open_elsewhere,
    _scratch_root,
    iter_matrix_chunks,
    load_obs,
    load_var,
    read_backed,
    resolve_data_path,
    resolve_path_to_update,
    write_var,
)

#: GTF lines parsed per block.
_GTF_BLOCK_LINES = 200_000

_PRIMARY = [str(i) for i in range(1, 23)] + ["X", "Y"]
#: chr1..chr22, chrX, chrY under UCSC (``chr1``), Ensembl (``1``) and human
#: RefSeq (``NC_000001``, its version dropped; 23 is X, 24 Y) names.
_CHROMOSOME_NAMES = {
    alias: f"chr{name}" for i, name in enumerate(_PRIMARY, 1) for alias in (name, f"chr{name}", f"NC_{i:06d}")
}


def _primary_chromosomes(names: pd.Series) -> pd.Series:
    """``chr1`` ... ``chr22``, ``chrX``, ``chrY`` for each name; NA for any
    other sequence (scaffolds, alternate loci, the mitochondrion)."""
    return names.str.replace(r"^(NC_\d+)\.\d+$", r"\1", regex=True).map(_CHROMOSOME_NAMES)


def _gtf_genes(gtf_file: str | Path) -> pd.DataFrame:
    """The ``gene`` records of a GTF file (plain or compressed, by its
    suffix) on chr1-22, X and Y, read a block of lines at a time:
    ``chromosome`` (``chr1`` ...), ``start``, ``end``, ``gene_id`` and
    ``gene_name`` (RefSeq's ``gene`` attribute when there is no
    ``gene_name``; NA when a record has neither)."""
    # No comment character: it would also cut a record at a '#' inside an
    # attribute ("protein #2"). Header lines have one field, so no feature,
    # and the feature == "gene" filter drops them.
    reader = pd.read_csv(
        gtf_file, sep="\t", header=None, names=range(9), usecols=[0, 2, 3, 4, 8], dtype=str,
        quoting=csv.QUOTE_NONE, chunksize=_GTF_BLOCK_LINES,
    )
    parts = []
    for block in reader:
        block = block.set_axis(["chromosome", "feature", "start", "end", "attributes"], axis=1)
        block = block[block["feature"] == "gene"]
        chromosome = _primary_chromosomes(block["chromosome"])
        block = block[chromosome.notna()]
        attributes = block["attributes"]
        symbol = attributes.str.extract(r'(?:^|;)\s*gene_name "([^"]*)"', expand=False)
        parts.append(pd.DataFrame({
            "chromosome": chromosome[chromosome.notna()],
            "start": block["start"].astype(np.int64),
            "end": block["end"].astype(np.int64),
            "gene_id": attributes.str.extract(r'(?:^|;)\s*gene_id "([^"]*)"', expand=False),
            "gene_name": symbol.fillna(attributes.str.extract(r'(?:^|;)\s*gene "([^"]*)"', expand=False)),
        }))
    genes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if genes.empty:
        raise ValueError(f"pp.genomic_position_from_gtf: {gtf_file} has no 'gene' records on chr1-22, X or Y")
    return genes


def _unversioned(ids: pd.Index | pd.Series):
    """Ensembl IDs without their version (``ENSG00000223972.5`` and
    GENCODE's ``ENSG00000182378.14_PAR_Y`` are ``ENSG00000223972``,
    ``ENSG00000182378``); other IDs as they are."""
    return ids.str.replace(r"^(ENS[A-Z]*\d{11})\.\d+(?:_PAR_Y)?$", r"\1", regex=True)


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
    GENCODE or RefSeq file need not fit in memory, and may be gzipped) and
    matched to ``var`` on ``gtf_gene_id``; only ``var`` is read from the
    h5ad file. Chromosomes are named ``chr1`` ... ``chr22``, ``chrX``,
    ``chrY`` whether the GTF names them so, ``1`` (Ensembl) or
    ``NC_000001.11`` (human RefSeq); genes on any other sequence (scaffolds,
    alternate loci, the mitochondrion) are left unplaced. A key on several
    records (a symbol on two loci, or both copies of a pseudoautosomal
    gene) takes the first, and how many there are is reported (infercnvpy
    leaves them all unplaced). Ensembl versions (``.12``) are ignored on
    both sides when matching IDs.

    Parameters
    ----------
    data
        Path to an h5ad file, or a crispyx AnnData / backed anndata.AnnData
        (``inplace=False`` only).
    gtf_file
        GTF annotation of the genome the counts were aligned to (Ensembl,
        GENCODE or NCBI RefSeq).
    gtf_gene_id
        GTF attribute matched to ``var``: ``"gene_name"`` (symbols; RefSeq's
        ``gene``) or ``"gene_id"`` (Ensembl IDs; RefSeq's are symbols).
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
    genes = _gtf_genes(gtf_file)
    if gtf_gene_id == "gene_id":
        keys = _unversioned(keys)
        genes["gene_id"] = _unversioned(genes["gene_id"])
    genes = genes[genes[gtf_gene_id].notna()].drop_duplicates()
    repeated = genes[gtf_gene_id].duplicated(keep=False)
    if repeated.any():
        _messages.warn(name, f"{genes.loc[repeated, gtf_gene_id].nunique():,} {gtf_gene_id}s are on more than one "
                             "GTF record; each takes its first")
    placed = genes.drop_duplicates(gtf_gene_id).set_index(gtf_gene_id).reindex(keys)
    positions = pd.DataFrame({
        "chromosome": placed["chromosome"].to_numpy(dtype=object),
        "start": pd.array(placed["start"].to_numpy(), dtype="Int64"),
        "end": pd.array(placed["end"].to_numpy(), dtype="Int64"),
    }, index=var.index)
    n_placed = int(positions["chromosome"].notna().sum())
    if n_placed < len(var):
        _messages.warn(name, f"{len(var) - n_placed:,} of {len(var):,} genes have no position on chr1-22, X or Y "
                             f"in {gtf_file}")
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


#: Cells in each partial sum of the reference means and of the spread. A
#: chunk is a whole number of them, so every sum adds the same numbers in the
#: same order whatever ``chunk_size``: the profiles do not change with it.
_SUM_ROWS = 256

#: Cells of the profiles read at a time by :func:`cnv_summary`.
_SUMMARY_ROWS = 2048


def _windows(
    var: pd.DataFrame, window_size: int, step: int,
) -> tuple[np.ndarray, sp.csr_matrix, dict[str, int], pd.DataFrame]:
    """infercnvpy's running means as one sparse product. Returns the rows of
    ``var`` chromosome by chromosome in genomic order; a genes (in that order)
    x windows matrix of pyramid weights, each window ``window_size`` genes and
    every ``step``-th of a chromosome's windows kept (one flat window over a
    chromosome with no more genes than that); each chromosome's first window;
    and each window's chromosome, first and last gene, and the genomic span
    of its genes. Only chromosomes named ``chr*`` other than ``chrM`` are
    kept, as there."""
    chromosomes = _natural_sort([c for c in var["chromosome"].astype(str).unique() if c.startswith("chr") and c != "chrM"])
    if not chromosomes:
        raise ValueError("tl.infercnv: no gene is placed on a chromosome named 'chr*'")
    names = var.index.to_numpy(dtype=object)
    starts, ends = var["start"].to_numpy(dtype=np.int64), var["end"].to_numpy(dtype=np.int64)
    order, rows, cols, weights, tables, chr_pos = [], [], [], [], [], {}
    n_genes = n_windows = 0
    for chromosome in chromosomes:
        genes = var.index.get_indexer(var.loc[var["chromosome"] == chromosome].sort_values("start").index)
        order.append(genes)
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
        first, last = genes[members[:, 0]], genes[members[:, -1]]
        tables.append(pd.DataFrame({
            "chromosome": chromosome, "first_gene": names[first], "last_gene": names[last],
            "start": starts[first], "end": ends[genes[members]].max(axis=1),
        }))
        chr_pos[chromosome] = n_windows
        n_genes += length
        n_windows += len(members)
    matrix = sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n_genes, n_windows),
    )
    return np.concatenate(order), matrix, chr_pos, pd.concat(tables, ignore_index=True)


def infercnv(
    data: str | Path | AnnData | ad.AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    reference: np.ndarray | None = None,
    batch_key: str | None = None,
    min_reference_cells: int = 100,
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
) -> tuple[dict, sp.csr_matrix] | None:
    """Copy-number profiles of cells from their expression along the genome:
    infercnvpy's method (:func:`infercnvpy.tl.infercnv`), streamed.

    Each cell's log-normalised expression less the reference cells' mean
    (between the lowest and highest reference group's mean when there are
    several, else 0) is clipped at ``lfc_clip``, averaged over windows of
    ``window_size`` genes in genomic order (pyramid weights, every
    ``step``-th window, each chromosome on its own), centred on the cell's
    median window, and values under ``dynamic_threshold`` standard
    deviations are set to 0. Same arguments and defaults as infercnvpy.
    With ``batch_key`` each batch (e.g. a sample) is compared with its own
    reference cells and has its own threshold, as if run on its own; a
    reference group a batch lacks is left out of that batch's range.

    Memory holds a chunk of cells densely over the placed genes, never
    ``X``, which is read twice: for the reference means, then for the
    profiles, written with ``dynamic_threshold`` to a scratch file (4 bytes
    per cell and window; in the temporary directory when it is on disk with
    room, else beside the input) while their standard deviation is taken,
    and thresholded as they are read back. infercnvpy takes the standard
    deviation within each chunk of 5,000 cells, so its threshold, and which
    values it zeroes, change with ``chunksize`` and the order of the cells;
    crispyx takes it over all cells of a batch, and equals infercnvpy when
    every cell is in one of its chunks (on float32 data infercnvpy computes
    in float32 and crispyx in float64: values agree to about 1e-6, and a
    value that close to the threshold can be zeroed by one and not the
    other). The sums are taken over the same blocks of 256 cells whatever
    the chunk size, so the profiles are the same for any ``chunk_size`` or
    ``memory_limit_gb``. A CSC file is read whole for every chunk of cells:
    convert it first (:func:`crispyx.pp.convert_to_csr`). With ``inplace``,
    a file that could not be written (open elsewhere, read-only) is an error
    before the passes.

    Parameters
    ----------
    data
        Path to an h5ad file of log-normalised expression whose ``var`` has
        ``chromosome``, ``start`` and ``end``
        (:func:`genomic_position_from_gtf`), or a crispyx AnnData /
        backed anndata.AnnData (``inplace=False`` only).
    reference_key, reference_cat
        ``obs`` column and its value(s) of the normal cells; every cell's
        mean (of its batch) when ``None``.
    reference
        Reference expression instead, ``(n_vars,)`` or ``(groups, n_vars)``;
        not with ``batch_key``.
    batch_key
        ``obs`` column of the batches (e.g. ``"sample_id"``), each compared
        with its own reference cells, with its own threshold.
    min_reference_cells
        A batch (all cells without ``batch_key``) with fewer reference cells
        is an error: the mean of a few cells puts their noise into every
        profile.
    lfc_clip
        Clip the differences from the reference at this value.
    window_size
        Genes in a window.
    step
        Keep every ``step``-th window.
    dynamic_threshold
        Set values within this many standard deviations (of all profiles of
        the batch) of 0 to 0; ``None`` keeps them.
    exclude_chromosomes
        Chromosome(s) left out (by default the sex chromosomes).
    layer
        Use ``layers[layer]`` instead of ``X``.
    key_added
        The profiles go to ``obsm["X_<key_added>"]`` (cells x windows, CSR);
        each chromosome's first window to ``uns[key_added]["chr_pos"]``, and
        each window's ``chromosome``, ``first_gene``, ``last_gene``,
        ``start`` and ``end`` (the span of its genes) to
        ``uns[key_added]["windows"]``.
    chunk_size
        Cells per chunk, rounded down to a multiple of 256 (at least 256);
        from ``memory_limit_gb`` when ``None``.
    memory_limit_gb
        Memory limit of the process in GB; ``None`` uses the memory detected
        as available.
    inplace
        Write the result to the file; ``False`` returns it.
    verbose
        Print progress.

    Returns
    -------
    (uns, profiles) or None
        ``{"chr_pos": ..., "windows": ...}`` and the CSR profiles, when
        ``inplace=False``.
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
    if reference is not None and (batch_key is not None or reference_key is not None):
        raise ValueError(f"{name}: pass reference, or reference_key / batch_key, not both")
    unplaced = var["chromosome"].isna()
    if unplaced.any():
        _messages.vprint(verbose, name, f"{int(unplaced.sum()):,} genes without a position are left out")
    keep = ~unplaced
    if exclude_chromosomes is not None:
        if isinstance(exclude_chromosomes, str):
            exclude_chromosomes = [exclude_chromosomes]
        keep &= ~var["chromosome"].isin(exclude_chromosomes)
    positioned = var.loc[keep, ["chromosome", "start", "end"]]
    if positioned[["start", "end"]].isna().any(axis=None):
        raise ValueError(f"{name}: genes with a chromosome but no start or end in var")
    order, smooth, chr_pos, windows = _windows(positioned, window_size, step)
    columns = np.flatnonzero(keep.to_numpy())[order]  # var positions, in genomic order
    smooth_t = smooth.T.tocsr()

    obs = load_obs(path)
    n_obs = len(obs)
    if batch_key is None:
        batch, batches = np.zeros(n_obs, dtype=np.int64), pd.Index(["all cells"])
    else:
        if batch_key not in obs.columns:
            raise KeyError(f"{name}: obs has no column {batch_key!r}")
        batch, batches = pd.factorize(obs[batch_key])
        if (batch < 0).any():
            raise ValueError(f"{name}: obs[{batch_key!r}] has missing values")
    n_batches = len(batches)

    scratch = path.with_name(f".{path.name}.X_{key_added}.{os.getpid()}.partial")
    alternative = "pass inplace=False to get them in memory"
    if inplace:  # before the passes, not after them
        _close_wrappers_on(path)
        _check_results_writable(path, scratch, name, alternative)
    unthresholded = None
    parts = []
    try:
        backed = read_backed(path)
        try:
            matrix = backed.X if layer is None else backed.layers[layer]
            n_vars = matrix.shape[1]
            if chunk_size is None:
                # The chunk densely over the placed genes, about four times
                # (the values, their differences, clipped, and the indexing).
                budget = _resolve_memory_limit_bytes(memory_limit_gb)
                chunk_size = int(min(n_obs, 5000, max(1, 0.25 * budget // (32 * len(columns)))))
            chunk_size = max(1, int(chunk_size) // _SUM_ROWS) * _SUM_ROWS

            def blocks(label):
                with _create_progress_context(-(-n_obs // chunk_size), label, verbose, unit="chunk") as bar:
                    for slc, block in iter_matrix_chunks(
                        backed, axis=0, matrix=matrix, chunk_size=chunk_size, convert_to_dense=False,
                    ):
                        bar.update(1)
                        block = block[:, columns]
                        yield slc, (block.toarray() if sp.issparse(block) else np.asarray(block)).astype(np.float64)

            def sum_blocks(slc):
                """The chunk's blocks of ``_SUM_ROWS`` cells, as slices of it."""
                length = slc.stop - slc.start
                return [slice(a, min(a + _SUM_ROWS, length)) for a in range(0, length, _SUM_ROWS)]

            if reference is None:
                if reference_key is None or reference_cat is None:
                    _messages.vprint(verbose, name, "reference: the mean of every cell (pass reference_key and "
                                                    "reference_cat to use normal cells)")
                    category, n_categories = np.zeros(n_obs, dtype=np.int64), 1
                else:
                    labels = obs[reference_key]
                    categories = list(np.atleast_1d(np.asarray(reference_cat, dtype=object)))
                    absent = [c for c in categories if not (labels == c).any()]
                    if absent:
                        raise ValueError(f"{name}: obs[{reference_key!r}] has no {absent}")
                    category = np.full(n_obs, -1, dtype=np.int64)
                    for k, value in enumerate(categories):
                        category[(labels == value).to_numpy()] = k
                    n_categories = len(categories)
                group = np.where(category >= 0, batch * n_categories + category, -1)
                counts = np.bincount(group[group >= 0], minlength=n_batches * n_categories).reshape(
                    n_batches, n_categories,
                )
                short = counts.sum(axis=1) < max(1, min_reference_cells)
                if short.any():
                    found = ", ".join(f"{b} ({n:,})" for b, n in zip(batches[short], counts.sum(axis=1)[short]))
                    raise ValueError(f"{name}: fewer than min_reference_cells={min_reference_cells} reference "
                                     f"cells in {found}")
                sums = np.zeros((n_batches * n_categories, len(columns)))
                for slc, x in blocks("reference"):
                    for part in sum_blocks(slc):
                        g = group[slc][part]
                        within = np.flatnonzero(g >= 0)
                        member = sp.csr_matrix(
                            (np.ones(within.size), (g[within], within)), shape=(len(sums), part.stop - part.start),
                        )
                        sums += member @ x[part]
                with np.errstate(invalid="ignore", divide="ignore"):
                    means = (sums / counts.reshape(-1, 1)).reshape(n_batches, n_categories, -1)
                means[counts == 0] = np.nan  # a group a batch lacks is not in its range
                low, high = np.nanmin(means, axis=1), np.nanmax(means, axis=1)
            else:
                ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))
                if ref.shape[1] != n_vars:
                    raise ValueError(f"{name}: reference has {ref.shape[1]} genes, the file {n_vars}")
                ref = ref[:, columns]
                low, high = ref.min(axis=0, keepdims=True), ref.max(axis=0, keepdims=True)

            def profiles(x, codes):
                # Between the lowest and highest reference mean counts as no
                # change; with one reference group that is x less its mean.
                present = np.unique(codes)
                for k in present:
                    rows = slice(None) if len(present) == 1 else codes == k
                    part = x[rows]
                    x[rows] = np.where(part > high[k], part - high[k], np.where(part < low[k], part - low[k], 0.0))
                np.clip(x, -lfc_clip, lfc_clip, out=x)
                smoothed = (smooth_t @ x.T).T
                return smoothed - np.median(smoothed, axis=1, keepdims=True)

            n_chunks = -(-n_obs // chunk_size)
            if dynamic_threshold is None:
                finished = (profiles(x, batch[slc]) for slc, x in blocks("profiles"))
            else:
                # The profiles go to a scratch file as they are computed, with
                # the standard deviation of every value of a batch (the
                # blocks' means and sums of squares combined, Chan et al.);
                # the threshold is then applied to them as read back, rather
                # than reading X and smoothing again.
                size = 4.0 * n_obs * len(windows)
                where = _scratch_root(size, path.parent)
                warn_if_disk_space_low(size, where, context=f"{name} profiles before the threshold")
                handle, unthresholded = tempfile.mkstemp(prefix=".cx_infercnv_", suffix=".h5", dir=where)
                os.close(handle)
                unthresholded = Path(unthresholded)
                evict = _EvictEvery(unthresholded)
                count, mean, m2 = np.zeros(n_batches), np.zeros(n_batches), np.zeros(n_batches)
                with h5py.File(unthresholded, "w") as store:
                    stored = store.create_dataset(
                        "profiles", shape=(n_obs, len(windows)), dtype=np.float32,
                        chunks=(min(chunk_size, n_obs), len(windows)),
                    )
                    for slc, x in blocks("profiles"):
                        codes = batch[slc]
                        values = profiles(x, codes)
                        stored[slc] = values
                        evict(values.size * 4)
                        for part in sum_blocks(slc):
                            for k in np.unique(codes[part]):
                                v = values[part][codes[part] == k]
                                n, m = v.size, v.mean()
                                delta = m - mean[k]
                                mean[k] += delta * n / (count[k] + n)
                                m2[k] += ((v - m) ** 2).sum() + delta ** 2 * count[k] * n / (count[k] + n)
                                count[k] += n
                threshold = dynamic_threshold * np.sqrt(m2 / count)

                def read_back():
                    with h5py.File(unthresholded, "r") as store, \
                            _create_progress_context(n_chunks, "threshold", verbose, unit="chunk") as bar:
                        for start in range(0, n_obs, chunk_size):
                            slc = slice(start, min(start + chunk_size, n_obs))
                            values = store["profiles"][slc]
                            evict(values.nbytes)
                            bar.update(1)
                            values[np.abs(values) < threshold[batch[slc]][:, None]] = 0
                            yield values

                finished = read_back()

            # Profiles go to a scratch file beside the input, copied in once
            # complete: a run killed part way leaves the file as it was.
            with h5py.File(scratch, "w") if inplace else nullcontext() as out:
                for values in finished:
                    block = sp.csr_matrix(values.astype(np.float32, copy=False))
                    if inplace:
                        _append_sparse_block(out, block, f"X_{key_added}")
                    else:
                        parts.append(block)
        finally:
            backed.file.close()
        uns = {"chr_pos": {chromosome: int(start) for chromosome, start in chr_pos.items()}, "windows": windows}
        _messages.print_done(verbose, name, f"{n_obs:,} cells x {len(windows):,} windows on {len(chr_pos)} chromosomes")
        if not inplace:
            return uns, sp.vstack(parts, format="csr")
        _close_wrappers_on(path)
        try:
            dst = h5py.File(path, "r+")
        except OSError as err:
            raise _results_open_elsewhere(path, err, name, alternative) from err
        with dst, h5py.File(scratch, "r") as tmp:
            obsm = dst.require_group("obsm")
            if f"X_{key_added}" in obsm:
                del obsm[f"X_{key_added}"]
            obsm.copy(tmp[f"X_{key_added}"], f"X_{key_added}")
            group = dst.require_group("uns")
            if key_added in group:
                del group[key_added]
            write_elem(group, key_added, uns)
    finally:
        scratch.unlink(missing_ok=True)
        if unthresholded is not None:
            unthresholded.unlink(missing_ok=True)
    return None


@contextmanager
def _stored_profiles(data, use_rep: str) -> Iterator[tuple[pd.DataFrame, dict, object]]:
    """``obs``, ``uns[use_rep]`` and ``obsm["X_<use_rep>"]`` (left on disk,
    sliced by rows, when ``data`` is a file)."""
    key = f"X_{use_rep}"
    if isinstance(data, ad.AnnData) and not data.isbacked:
        if key not in data.obsm or use_rep not in data.uns:
            raise KeyError(f"no obsm[{key!r}] / uns[{use_rep!r}]: run tl.infercnv first")
        yield data.obs, data.uns[use_rep], data.obsm[key]
        return
    path = resolve_data_path(data)
    obs = load_obs(path)
    with h5py.File(path, "r") as h5:
        if f"obsm/{key}" not in h5 or f"uns/{use_rep}" not in h5:
            raise KeyError(f"{path} has no obsm[{key!r}] / uns[{use_rep!r}]: run tl.infercnv first")
        elem = h5["obsm"][key]
        yield obs, read_elem(h5["uns"][use_rep]), sparse_dataset(elem) if isinstance(elem, h5py.Group) else elem


def cnv_summary(
    data: str | Path | AnnData | ad.AnnData,
    groupby: str,
    *,
    batch_key: str | None = None,
    use_rep: str = "cnv",
    min_cells: int = 1,
) -> pd.DataFrame:
    """Each group's mean copy-number profile per chromosome and its CNV
    score (infercnvpy's :func:`infercnvpy.tl.cnv_score`), as a table.

    The profiles (:func:`infercnv`) are read a block of cells at a time;
    nothing else is read from the file but ``obs`` and ``uns[use_rep]``.

    Parameters
    ----------
    data
        Path to an h5ad file, a crispyx AnnData or an anndata.AnnData with
        ``obsm["X_<use_rep>"]`` and ``uns[use_rep]["chr_pos"]``.
    groupby
        ``obs`` column of the groups (e.g. ``"leiden"``).
    batch_key
        ``obs`` column (e.g. ``"sample_id"``): a row per batch and group.
    use_rep
        The profiles' key (``key_added`` of :func:`infercnv`).
    min_cells
        Leave out groups with fewer cells.

    Returns
    -------
    pd.DataFrame
        A row per group (per batch and group): ``batch_key`` and ``groupby``;
        ``cells``; ``score``, the mean absolute value over the group's cells
        and all windows; and per chromosome, the mean value over the group's
        cells and that chromosome's windows.
    """
    keys = [batch_key, groupby] if batch_key is not None else [groupby]
    with _stored_profiles(data, use_rep) as (obs, uns, matrix):
        absent = [key for key in keys if key not in obs.columns]
        if absent:
            raise KeyError(f"tl.cnv_summary: obs has no column {', '.join(map(repr, absent))}")
        grouped = obs[keys].groupby(keys, observed=True, sort=True)
        codes, sizes = grouped.ngroup().fillna(-1).to_numpy(dtype=np.int64), grouped.size()  # -1: no label
        n_obs, n_windows = matrix.shape
        sums, magnitude = np.zeros((len(sizes), n_windows)), np.zeros(len(sizes))
        for start in range(0, n_obs, _SUMMARY_ROWS):
            block = matrix[start:start + _SUMMARY_ROWS]
            block = block.tocsr() if sp.issparse(block) else sp.csr_matrix(np.asarray(block))
            group = codes[start:start + block.shape[0]]
            within = np.flatnonzero(group >= 0)
            member = sp.csr_matrix((np.ones(within.size), (group[within], within)), shape=(len(sizes), block.shape[0]))
            sums += (member @ block).toarray()
            magnitude += member @ np.asarray(abs(block).sum(axis=1)).ravel()
    starts = sorted(((chromosome, int(first)) for chromosome, first in uns["chr_pos"].items()), key=lambda t: t[1])
    bounds = [first for _, first in starts] + [n_windows]
    cells = sizes.to_numpy()
    table = sizes.rename("cells").reset_index()
    table["score"] = magnitude / (cells * n_windows)
    for (chromosome, a), b in zip(starts, bounds[1:]):
        table[chromosome] = sums[:, a:b].sum(axis=1) / (cells * (b - a))
    return table[table["cells"] >= min_cells].reset_index(drop=True)
