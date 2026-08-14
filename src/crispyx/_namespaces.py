"""Scanpy-style namespace classes (pp, pb, tl, pl) for crispyx."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import anndata as ad
import numpy as np

from ._preflight import estimate_disk_usage
from .batch import BatchReducer, batch_process
from .data import (
    AnnData,
    compute_overlap,
    convert_to_csc,
    convert_to_csr,
    downsample_counts,
    ensure_gene_symbol_column,
    normalize_total_log1p,
    read_backed,
    resolve_control_label,
    resolve_data_path,
    resolve_output_path,
)
from .de import (
    RankGenesGroupsResult,
    _adjust_pvalue_matrix,
    nb_glm_test,
    shrink_lfc,
    t_test,
    wilcoxon_test,
)
from .plotting import (
    materialize_rank_genes_groups,
    plot_ma,
    plot_overlap_heatmap,
    plot_pca,
    plot_pca_loadings,
    plot_pca_variance_ratio,
    plot_qc_perturbation_counts,
    plot_qc_summary,
    plot_top_genes_bar,
    plot_umap,
    plot_volcano,
    rank_genes_groups as plot_rank_genes_groups,
    rank_genes_groups_df,
)
from .pseudobulk import (
    NormalizedEffectMethod,
    PseudobulkMethod,
    aggregate_pseudobulk,
    compute_normalized_effects,
    compute_pseudobulk_effects,
)
from .qc import (
    filter_cells_by_gene_count,
    filter_genes_by_cell_count,
    filter_perturbations_by_cell_count,
    quality_control_summary,
)
from .sample import subsample


# ---------------------------------------------------------------------------
# Helpers used only by _ToolsNamespace
# ---------------------------------------------------------------------------

def _infer_control_label(
    path: Path,
    perturbation_column: str,
    control_label: str | None,
) -> str:
    if control_label is not None:
        return str(control_label)
    backed = read_backed(path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                "Perturbation column '%s' was not found in adata.obs. Available columns: %s"
                % (perturbation_column, list(backed.obs.columns))
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
    finally:
        backed.file.close()
    return resolve_control_label(labels, None)


def _t_test_results_to_rank_genes(
    path: Path,
    results,
    *,
    gene_name_column: str | None,
    perturbation_column: str,
    control_label: str,
    corr_method: str,
    output_dir: str | Path | None,
    data_name: str | None,
) -> RankGenesGroupsResult:
    groups = list(results.keys())
    if groups:
        first = results[groups[0]]
        genes = first.genes
        effect_matrix = np.vstack([results[group].effect_size for group in groups])
        statistic_matrix = np.vstack([results[group].statistic for group in groups])
        pvalue_matrix = np.vstack([results[group].pvalue for group in groups])
        result_view = first.result
    else:
        backed = read_backed(path)
        try:
            if gene_name_column is None:
                genes = backed.var_names.astype(str)
            else:
                genes = ensure_gene_symbol_column(backed, gene_name_column)
        finally:
            backed.file.close()
        effect_matrix = np.zeros((0, genes.size), dtype=float)
        statistic_matrix = np.zeros_like(effect_matrix)
        pvalue_matrix = np.ones_like(effect_matrix)
        result_path = resolve_output_path(
            path,
            suffix="t_test_de",
            output_dir=output_dir,
            data_name=data_name,
        )
        result_view = AnnData(result_path)

    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        raise ValueError(
            "corr_method must be 'benjamini-hochberg' or 'bonferroni' for t-tests"
        )

    pvalue_adj = (
        _adjust_pvalue_matrix(pvalue_matrix, corr_method)
        if pvalue_matrix.size
        else np.zeros_like(pvalue_matrix)
    )
    order = (
        np.argsort(-np.abs(statistic_matrix), axis=1, kind="mergesort")
        if statistic_matrix.size
        else np.zeros(statistic_matrix.shape, dtype=int)
    )
    zeros = np.zeros_like(statistic_matrix)

    result = RankGenesGroupsResult(
        genes=genes,
        groups=groups,
        statistics=statistic_matrix,
        pvalues=pvalue_matrix,
        pvalues_adj=pvalue_adj,
        logfoldchanges=effect_matrix,
        effect_size=effect_matrix,
        u_statistics=zeros,
        pts=zeros,
        pts_rest=zeros,
        order=order,
        groupby=perturbation_column,
        method="t_test",
        control_label=control_label,
        tie_correct=False,
        pvalue_correction=corr_method,
        result=result_view,
    )
    if result.result is not None:
        memory = result.result.to_memory()
        memory.uns["rank_genes_groups"] = result.to_rank_genes_groups_dict()
        memory.uns["genes"] = genes.to_numpy()
        memory.uns["method"] = "t_test"
        memory.uns["control_label"] = control_label
        memory.uns["tie_correct"] = False
        memory.uns["pvalue_correction"] = corr_method
        memory.write(result.result.path)
        result.result.close()
        result.result = AnnData(result.result.path)
    return result


# ---------------------------------------------------------------------------
# Namespace classes
# ---------------------------------------------------------------------------

class _PreprocessingNamespace:
    """Scanpy-style preprocessing entry points (``cx.pp``)."""

    def filter_cells(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_genes: int = 100,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
        verbose: int | bool = True,
    ):
        path = resolve_data_path(data)
        return filter_cells_by_gene_count(
            path,
            min_genes=min_genes,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            verbose=verbose,
        )

    def filter_genes(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_cells: int = 100,
        cell_mask: np.ndarray | None = None,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
        verbose: int | bool = True,
    ):
        path = resolve_data_path(data)
        return filter_genes_by_cell_count(
            path,
            min_cells=min_cells,
            cell_mask=cell_mask,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            verbose=verbose,
        )

    def filter_perturbations(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str,
        control_label: str | None = None,
        min_cells: int = 50,
        base_mask: np.ndarray | None = None,
        verbose: int | bool = True,
    ):
        path = resolve_data_path(data)
        return filter_perturbations_by_cell_count(
            path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            min_cells=min_cells,
            base_mask=base_mask,
            verbose=verbose,
        )

    def subsample(
        self,
        data: str | Path | ad.AnnData,
        *,
        n: int | None = None,
        frac: float | None = None,
        groupby: str | Sequence[str] | None = None,
        unit: str = "cell",
        drop_insufficient: bool = True,
        random_state: int = 0,
        chunk_size: int = 4096,
        output_path: str | Path | None = None,
        data_name: str | None = None,
        verbose: int | bool = True,
    ):
        """Stream a stratified or cluster-sampled subset to disk.

        See :func:`crispyx.subsample` for the full parameter documentation.
        """
        path = resolve_data_path(data)
        return subsample(
            path,
            n=n,
            frac=frac,
            groupby=groupby,
            unit=unit,
            drop_insufficient=drop_insufficient,
            random_state=random_state,
            chunk_size=chunk_size,
            output_path=output_path,
            data_name=data_name,
            verbose=verbose,
        )

    def qc_summary(
        self,
        data: str | Path | ad.AnnData,
        *,
        min_genes: int = 100,
        min_cells_per_perturbation: int = 50,
        min_cells_per_gene: int = 100,
        perturbation_column: str,
        control_label: str | None = None,
        gene_name_column: str | None = None,
        chunk_size: int = 2048,
        data_name: str | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,  # deprecated; use output_path; will be removed in next major version
        cache_mode: Literal['memory', 'memmap', 'none'] = 'memmap',
        verbose: int | bool = True,
    ):
        path = resolve_data_path(data)
        result = quality_control_summary(
            path,
            min_genes=min_genes,
            min_cells_per_perturbation=min_cells_per_perturbation,
            min_cells_per_gene=min_cells_per_gene,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
            output_path=output_path,
            cache_mode=cache_mode,
            verbose=verbose,
        )
        return result.filtered

    def convert_to_csc(
        self,
        data: str | Path | ad.AnnData,
        *,
        output_path: str | Path | None = None,
        chunk_size: int = 4096,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        verbose: int | bool = True,
    ) -> AnnData:
        """Convert a backed h5ad file's matrix to CSC format.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData.
        output_path
            Explicit output path.  If None, derived from output_dir/data_name.
        chunk_size
            Rows per streaming chunk.  Default 4096.
        output_dir
            Output directory.  Defaults to input file's directory.
        data_name
            Custom name suffix.
        verbose
            Print progress.

        Returns
        -------
        AnnData
            Backed AnnData pointing to the CSC output file.
        """
        return convert_to_csc(
            data,
            output_path=output_path,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
            verbose=verbose,
        )

    def convert_to_csr(
        self,
        data: str | Path | ad.AnnData,
        *,
        output_path: str | Path | None = None,
        chunk_size: int | None = None,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        verbose: int | bool = True,
    ) -> AnnData:
        """Convert a backed h5ad file's matrix to CSR format.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData.
        output_path
            Explicit output path.  If None, derived from output_dir/data_name.
        chunk_size
            Rows (or columns for CSC source) per streaming chunk.  Default auto.
        output_dir
            Output directory.  Defaults to input file's directory.
        data_name
            Custom name suffix.
        verbose
            Print progress.

        Returns
        -------
        AnnData
            Backed AnnData pointing to the CSR output file.
        """
        return convert_to_csr(
            data,
            output_path=output_path,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
            verbose=verbose,
        )

    def normalize_total_log1p(
        self,
        data: str | Path | ad.AnnData,
        output_path: str | Path | None = None,
        *,
        normalize: bool = True,
        log1p: bool = True,
        target_sum: float = 1e4,
        chunk_size: int = 4096,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        format_mismatch_policy: str = "warn",
        verbose: int | bool = True,
    ) -> AnnData:
        """Stream normalize and/or log-transform an h5ad file.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData.
        output_path
            Path for output. If None, uses output_dir/data_name pattern.
        normalize
            Apply total-count normalization. Default True.
        log1p
            Apply log1p transformation. Default True.
        target_sum
            Target counts per cell. Default 1e4.
        chunk_size
            Cells per chunk. Default 4096.
        output_dir
            Output directory. Defaults to input file's directory.
        data_name
            Custom output name suffix.
        format_mismatch_policy
            How to handle a CSC source (slow for cell-streaming): 'warn'
            (default), 'convert' (transparently stream via a temporary CSR
            copy), or 'off'.
        verbose
            Print progress.

        Returns
        -------
        AnnData
            Read-only AnnData wrapper pointing to output file.
        """
        return normalize_total_log1p(
            data,
            output_path,
            normalize=normalize,
            log1p=log1p,
            target_sum=target_sum,
            chunk_size=chunk_size,
            output_dir=output_dir,
            data_name=data_name,
            format_mismatch_policy=format_mismatch_policy,
            verbose=verbose,
        )

    def downsample_counts(
        self,
        data: str | Path | ad.AnnData,
        output_path: str | Path | None = None,
        *,
        counts_per_cell: int,
        chunk_size: int = 4096,
        data_name: str | None = None,
        random_state: int = 0,
        verbose: int | bool = True,
    ) -> AnnData:
        """Stream-thin every cell's counts down to at most ``counts_per_cell``.

        See :func:`crispyx.data.downsample_counts` for the full parameter
        documentation. Dependency-free streaming equivalent of
        ``scanpy.pp.downsample_counts(..., replace=False)``.
        """
        return downsample_counts(
            data,
            output_path,
            counts_per_cell=counts_per_cell,
            chunk_size=chunk_size,
            data_name=data_name,
            random_state=random_state,
            verbose=verbose,
        )

    def pca(
        self,
        data: str | Path | ad.AnnData,
        n_comps: int = 50,
        method: str = "auto",
        use_highly_variable: bool = True,
        chunk_size: int | None = None,
        random_state: int = 0,
        copy: bool = False,
        show_progress: bool = True,
    ) -> ad.AnnData | None:
        """Compute streaming PCA on backed AnnData.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData.
        n_comps
            Number of principal components. Default 50.
        method
            'auto', 'sparse_cov', or 'incremental'. Default 'auto'.
        use_highly_variable
            Use only HVGs if available. Default True.
        chunk_size
            Cells per chunk. Auto-calculated if None.
        random_state
            Random seed.
        copy
            If True, return copy with results instead of in-place.
        show_progress
            Show progress bars.

        Returns
        -------
        AnnData or None
            Modified AnnData if copy=True, else None.
        """
        from .dimred import pca as _pca

        if isinstance(data, (str, Path)):
            adata = ad.read_h5ad(data, backed='r')
        else:
            adata = data

        return _pca(
            adata,
            n_comps=n_comps,
            method=method,
            use_highly_variable=use_highly_variable,
            chunk_size=chunk_size,
            random_state=random_state,
            copy=copy,
            show_progress=show_progress,
        )

    def neighbors(
        self,
        data: str | Path | ad.AnnData,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        use_rep: str = "X_pca",
        metric: str = "euclidean",
        method: str = "umap",
        random_state: int = 0,
        copy: bool = False,
        show_progress: bool = True,
    ) -> ad.AnnData | None:
        """Compute k-nearest neighbors graph from embeddings.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData with PCA results.
        n_neighbors
            Number of neighbors. Default 15.
        n_pcs
            Number of PCs to use. Default None uses all.
        use_rep
            Key in .obsm for embeddings. Default 'X_pca'.
        metric
            Distance metric. Default 'euclidean'.
        method
            'umap' (fast, pynndescent) or 'sklearn' (exact).
        random_state
            Random seed.
        copy
            If True, return copy with results.
        show_progress
            Show progress.

        Returns
        -------
        AnnData or None
            Modified AnnData if copy=True, else None.
        """
        from .dimred import neighbors as _neighbors

        if isinstance(data, (str, Path)):
            adata = ad.read_h5ad(data, backed='r')
        else:
            adata = data

        return _neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
            metric=metric,
            method=method,
            random_state=random_state,
            copy=copy,
            show_progress=show_progress,
        )


class _PseudobulkNamespace:
    """Pseudo-bulk estimators (``cx.pb``)."""

    def aggregate(
        self,
        data: str | Path | ad.AnnData,
        *,
        groupby: str | Sequence[str],
        method: PseudobulkMethod = "mean_log1p",
        layer: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        min_cells: int = 5,
        bootstrap_size: int | None = None,
        random_state: int = 0,
        chunk_size: int | None = None,
        memory_limit_gb: float | None = None,
        data_name: str | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        verbose: int | bool = True,
        force: bool = False,
    ) -> AnnData:
        """Aggregate absolute profiles for observed combinations in ``groupby``.

        See :func:`crispyx.pseudobulk.aggregate_pseudobulk` for the input
        contracts, bootstrap behavior, and output schema.
        """
        return aggregate_pseudobulk(
            data,
            groupby=groupby,
            method=method,
            layer=layer,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            min_cells=min_cells,
            bootstrap_size=bootstrap_size,
            random_state=random_state,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            data_name=data_name,
            output_path=output_path,
            output_dir=output_dir,
            verbose=verbose,
            force=force,
        )

    def effects(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str | None = None,
        groupby: str | None = None,
        batch_column: str | None = None,
        control_label: str | None = None,
        reference: str | None = None,
        aggregate_batches: bool = False,
        method: PseudobulkMethod = "mean_log1p",
        layer: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        min_cells: int = 5,
        bootstrap_size: int | None = None,
        random_state: int = 0,
        chunk_size: int | None = None,
        memory_limit_gb: float | None = None,
        bulk_output_path: str | Path | None = None,
        data_name: str | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        verbose: int | bool = True,
        force: bool = False,
    ) -> AnnData:
        """Compute within-batch target-minus-reference pseudobulk effects."""
        return compute_pseudobulk_effects(
            data,
            perturbation_column=perturbation_column,
            groupby=groupby,
            batch_column=batch_column,
            control_label=control_label,
            reference=reference,
            aggregate_batches=aggregate_batches,
            method=method,
            layer=layer,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            min_cells=min_cells,
            bootstrap_size=bootstrap_size,
            random_state=random_state,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            bulk_output_path=bulk_output_path,
            data_name=data_name,
            output_path=output_path,
            output_dir=output_dir,
            verbose=verbose,
            force=force,
        )

    def normalized_effects(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str | None = None,
        groupby: str | None = None,
        control_label: str | None = None,
        reference: str | None = None,
        method: NormalizedEffectMethod = "mean_log1p",
        baseline_count: float = 1.0,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        batch_column: str | None = None,
        chunk_size: int | None = None,
        memory_limit_gb: float | None = None,
        data_name: str | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        verbose: int | bool = True,
    ):
        """Library-size-normalised target-minus-reference effects, in one pass.

        See :func:`crispyx.compute_normalized_effects`.
        """
        return compute_normalized_effects(
            resolve_data_path(data),
            perturbation_column=perturbation_column,
            groupby=groupby,
            control_label=control_label,
            reference=reference,
            method=method,
            baseline_count=baseline_count,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            batch_column=batch_column,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            data_name=data_name,
            output_path=output_path,
            output_dir=output_dir,
            verbose=verbose,
        )


class _ToolsNamespace:
    """Differential expression and analysis tools (``cx.tl``)."""

    def umap(
        self,
        data: str | Path | ad.AnnData,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        neighbors_key: str = "neighbors",
        random_state: int = 0,
        copy: bool = False,
    ) -> ad.AnnData | None:
        """Compute UMAP embedding from pre-computed neighbor graph.

        Parameters
        ----------
        data
            Path to h5ad file or backed AnnData with neighbors computed.
        min_dist
            Minimum distance between embedded points. Default 0.5.
        spread
            Effective scale of embedded points. Default 1.0.
        n_components
            Number of UMAP dimensions. Default 2.
        neighbors_key
            Key in .uns for neighbor graph. Default 'neighbors'.
        random_state
            Random seed.
        copy
            Return copy with results instead of in-place.

        Returns
        -------
        AnnData or None
            Modified AnnData if copy=True, else None.
        """
        from .dimred import umap as _umap

        if isinstance(data, (str, Path)):
            adata = ad.read_h5ad(data, backed='r')
        else:
            adata = data

        return _umap(
            adata,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            neighbors_key=neighbors_key,
            random_state=random_state,
            copy=copy,
        )

    def batch_process(
        self,
        data: str | Path | ad.AnnData,
        reducer: BatchReducer,
        *,
        perturbation_column: str | None = None,
        groupby: str | None = None,
        control_label: str | None = None,
        reference: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        batch_column: str,
        mode: Literal["group", "comparison"] = "group",
        statistic_name: str,
        chunk_size: int | None = None,
        cell_chunk_size: int | None = None,
        data_name: str | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        verbose: int | bool = True,
        memory_limit_gb: float | None = None,
        force: bool = False,
    ) -> AnnData:
        """Compute a streaming gene-wise statistic within biological batches.

        ``data`` must be an h5ad path or backed AnnData. ``reducer`` must be a
        :class:`crispyx.BatchReducer` that consumes dense cell chunks and
        finalizes to one value per gene.

        The grouping arguments match differential expression:
        ``groupby`` aliases ``perturbation_column`` and ``reference`` aliases
        ``control_label``. See :func:`crispyx.batch.batch_process` for complete
        input requirements and reducer examples.
        """
        return batch_process(
            data,
            reducer,
            perturbation_column=perturbation_column,
            groupby=groupby,
            control_label=control_label,
            reference=reference,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            batch_column=batch_column,
            mode=mode,
            statistic_name=statistic_name,
            chunk_size=chunk_size,
            cell_chunk_size=cell_chunk_size,
            data_name=data_name,
            output_path=output_path,
            output_dir=output_dir,
            verbose=verbose,
            memory_limit_gb=memory_limit_gb,
            force=force,
        )

    def rank_genes_groups(
        self,
        data: str | Path | ad.AnnData,
        *,
        perturbation_column: str | None = None,
        groupby: str | None = None,
        method: str = "wilcoxon",
        control_label: str | None = None,
        reference: str | None = None,
        gene_name_column: str | None = None,
        perturbations: Iterable[str] | None = None,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        corr_method: str = "benjamini-hochberg",
        verbose: int | bool = True,
        resume: bool = False,
        memory_limit_gb: float | None = None,
        force: bool = False,
        **kwargs,
    ) -> RankGenesGroupsResult:
        # Resolve groupby / reference aliases
        if groupby is not None and perturbation_column is not None:
            raise TypeError(
                "rank_genes_groups() received both 'perturbation_column' and 'groupby'; "
                "they are aliases for the same parameter — pass only one."
            )
        if groupby is not None:
            perturbation_column = groupby
        if perturbation_column is None:
            raise TypeError(
                "rank_genes_groups() requires either 'perturbation_column' or its alias 'groupby'."
            )
        if reference is not None and control_label is not None:
            raise TypeError(
                "rank_genes_groups() received both 'control_label' and 'reference'; "
                "they are aliases for the same parameter — pass only one."
            )
        if reference is not None:
            control_label = reference

        path = resolve_data_path(data)
        method_key = method.lower().replace("_", "-")
        method_map = {
            "wilcoxon": "wilcoxon",
            "wilcox": "wilcoxon",
            "wilcoxon-test": "wilcoxon",
            "wilcox-test": "wilcoxon",
            "t-test": "t_test",
            "ttest": "t_test",
            "nb-glm": "nb_glm",
            "nb-glm-test": "nb_glm",
        }
        normalised = method_map.get(method_key, method_key)
        control = _infer_control_label(path, perturbation_column, control_label)

        base_kwargs = dict(
            perturbation_column=perturbation_column,
            control_label=control,
            gene_name_column=gene_name_column,
            perturbations=perturbations,
            output_dir=output_dir,
            data_name=data_name,
            verbose=verbose,
            resume=resume,
            memory_limit_gb=memory_limit_gb,
            force=force,
        )

        if normalised == "wilcoxon":
            allowed = {
                "min_cells_expressed", "min_pct_ctrl", "min_pct_pert", "min_pct_both",
                "min_mean_ctrl", "min_mean_pert", "chunk_size", "tie_correct",
                "n_jobs",
                "checkpoint_interval",
                "batch_column",
            }
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    "Unexpected keyword arguments for wilcoxon method: %s"
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            result = wilcoxon_test(
                path,
                corr_method=corr_method,
                **base_kwargs,
                **method_kwargs,
            )
            if result.result is None:
                raise RuntimeError("Wilcoxon test did not produce an AnnData result.")
            return result.result

        if normalised == "nb_glm":
            allowed = {
                "covariates",
                "dispersion",
                "fit_method",
                "share_dispersion",
                "max_iter",
                "tol",
                "poisson_init_iter",
                "min_cells_expressed",
                "min_pct_ctrl", "min_pct_pert", "min_pct_both",
                "min_mean_ctrl",
                "min_mean_pert",
                "min_total_count",
                "chunk_size",
                "n_jobs",
                "checkpoint_interval",
            }
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    "Unexpected keyword arguments for nb_glm method: %s"
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            result = nb_glm_test(
                path,
                corr_method=corr_method,
                **base_kwargs,
                **method_kwargs,
            )
            if result.result is None:
                raise RuntimeError("NB-GLM test did not produce an AnnData result.")
            return result.result

        if normalised == "t_test":
            allowed = {
                "min_cells_expressed", "min_pct_ctrl", "min_pct_pert", "min_pct_both",
                "min_mean_ctrl", "min_mean_pert", "cell_chunk_size",
                "n_jobs",
                "checkpoint_interval",
            }
            unexpected = set(kwargs) - allowed
            if unexpected:
                raise TypeError(
                    (
                        "Unexpected keyword arguments for t_test method: %s. "
                        "Supported options include cell_chunk_size (cells per chunk), "
                        "min_cells_expressed, and n_jobs; perturbation_chunk_size is not yet supported."
                    )
                    % ", ".join(sorted(unexpected))
                )
            method_kwargs = {key: kwargs[key] for key in allowed if key in kwargs}
            results = t_test(
                path,
                **base_kwargs,
                **method_kwargs,
            )
            mapped = _t_test_results_to_rank_genes(
                path,
                results,
                gene_name_column=gene_name_column,
                perturbation_column=perturbation_column,
                control_label=control,
                corr_method=corr_method,
                output_dir=output_dir,
                data_name=data_name,
            )
            if mapped.result is None:
                raise RuntimeError("t-test did not produce an AnnData result.")
            return mapped.result

        raise ValueError(
            f"Unsupported differential expression method: {method}. "
            "Choose from 'wilcoxon', 't-test', or 'nb_glm'."
        )

    def shrink_lfc(
        self,
        data: str | Path | ad.AnnData,
        *,
        output_dir: str | Path | None = None,
        data_name: str | None = None,
        method: str = "stats",
        prior_scale_mode: str = "global",
        min_mu: float = 0.0,
        n_jobs: int = -1,
        batch_size: int = 128,
        profiling: bool = False,
        memory_limit_gb: float | None = None,
        verbose: int | bool = True,
    ):
        """Apply apeGLM LFC shrinkage to NB-GLM results.

        Parameters
        ----------
        data
            Path to h5ad file from ``nb_glm_test()`` or a backed AnnData object.
        output_dir
            Directory for output. Defaults to input file's directory.
        data_name
            Custom name for output file.
        method
            Shrinkage computation method: "stats" (faster) or "full".
        prior_scale_mode
            Prior estimation scope: "global" or "per_comparison".
        min_mu
            Minimum mean for numerical stability.
        n_jobs
            Number of parallel workers. -1 uses all cores.
        batch_size
            Number of genes per batch.
        profiling
            Enable timing/memory profiling.
        memory_limit_gb
            Optional memory budget in GB. None auto-detects via psutil.

        Returns
        -------
        RankGenesGroupsResult
            Shrunk differential expression results.
        """
        path = resolve_data_path(data)
        return shrink_lfc(
            path,
            output_dir=output_dir,
            data_name=data_name,
            method=method,
            prior_scale_mode=prior_scale_mode,
            min_mu=min_mu,
            n_jobs=n_jobs,
            batch_size=batch_size,
            profiling=profiling,
            memory_limit_gb=memory_limit_gb,
            verbose=verbose,
        )

    def compute_overlap(self, sets_dict, *, metric="both"):
        """Compute pairwise overlap statistics. See :func:`crispyx.compute_overlap`."""
        return compute_overlap(sets_dict, metric=metric)

    def estimate_disk_usage(self, func, data, **kwargs):
        """Estimate the disk space a crispyx function will need. See :func:`crispyx.estimate_disk_usage`."""
        return estimate_disk_usage(func, data, **kwargs)


class _PlottingNamespace:
    """Scanpy-style plotting entry points (``cx.pl``)."""

    def rank_genes_groups(self, data, **kwargs):
        return plot_rank_genes_groups(data, **kwargs)

    def rank_genes_groups_df(self, data, group, **kwargs):
        return rank_genes_groups_df(data, group, **kwargs)

    def volcano(self, **kwargs):
        return plot_volcano(**kwargs)

    def ma(self, **kwargs):
        return plot_ma(**kwargs)

    def top_genes_bar(self, **kwargs):
        return plot_top_genes_bar(**kwargs)

    def qc_perturbation_counts(self, **kwargs):
        return plot_qc_perturbation_counts(**kwargs)

    def qc_summary(self, qc_result, **kwargs):
        return plot_qc_summary(qc_result, **kwargs)

    def materialize_rank_genes_groups(self, data, **kwargs):
        return materialize_rank_genes_groups(data, **kwargs)

    def pca(self, data, **kwargs):
        """Plot PCA scatter."""
        return plot_pca(data, **kwargs)

    def pca_variance_ratio(self, data, **kwargs):
        """Plot PCA variance ratio."""
        return plot_pca_variance_ratio(data, **kwargs)

    def pca_loadings(self, data, **kwargs):
        """Plot PCA loadings."""
        return plot_pca_loadings(data, **kwargs)

    def umap(self, data, **kwargs):
        """Plot UMAP embedding."""
        return plot_umap(data, **kwargs)

    def overlap_heatmap(self, result, **kwargs):
        """Plot pairwise overlap heatmap. See :func:`crispyx.plot_overlap_heatmap`."""
        return plot_overlap_heatmap(result, **kwargs)
