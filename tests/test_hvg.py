"""Tests for highly variable gene (HVG) selection (crispyx.hvg)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from crispyx.data import get_matrix_storage_format, load_var
from crispyx.hvg import _stream_gene_moments, highly_variable_genes


def _make_hvg_dataset(
    dir_path,
    fmt: str,
    *,
    seed: int = 0,
    n: int = 300,
    g: int = 50,
    perturbation_effect_gene: int | None = None,
    raw_counts: bool = False,
    name: str = "hvg",
) -> Path:
    """Write a small synthetic h5ad for HVG tests.

    Per-gene Poisson means vary so dispersion differs across genes. Stores
    log1p-normalized-looking data by default (for the ``mean_dispersion``
    flavor); pass ``raw_counts=True`` for ``seurat_v3``, which expects raw
    counts. When ``perturbation_effect_gene`` is given, that gene gets a
    modest additional signal in non-control cells only -- simulating an
    on-target perturbation effect that should dominate an all-cells HVG pass
    but not a control-cells-only pass.
    """
    rng = np.random.default_rng(seed)
    base_lambda = rng.uniform(0.5, 5.0, size=g)
    raw = rng.poisson(base_lambda, size=(n, g)).astype(np.float64)

    is_control = np.zeros(n, dtype=bool)
    is_control[: n // 2] = True
    rng.shuffle(is_control)
    labels = np.where(is_control, "NTC", "PERT")

    if perturbation_effect_gene is not None:
        # A modest additive burst confined to non-control cells: enough to
        # dominate normalized dispersion among genes of similar mean
        # expression, but not so large that the gene's mean gets shifted
        # into an isolated bin (where the single-gene-per-bin rule pins
        # normalized dispersion to exactly 1, masking the effect).
        raw[~is_control, perturbation_effect_gene] += rng.poisson(
            4.0, size=int((~is_control).sum())
        )

    X = raw if raw_counts else np.log1p(raw)
    Xf = sp.csr_matrix(X) if fmt == "csr" else sp.csc_matrix(X)
    obs = pd.DataFrame({"perturbation": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    path = Path(dir_path) / f"{name}_{fmt}.h5ad"
    ad.AnnData(X=Xf.astype(np.float32), obs=obs, var=var).write_h5ad(path)
    return path


def _scanpy_seurat_hvg(path, *, n_top_genes=None, min_mean=0.0125, max_mean=3.0, min_disp=0.5, n_bins=20):
    adata = ad.read_h5ad(path)
    return sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_top_genes,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        n_bins=n_bins,
        inplace=False,
    )


def _scanpy_seurat_v3_hvg(path, *, n_top_genes, span=0.3):
    adata = ad.read_h5ad(path)
    return sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top_genes, span=span, inplace=False,
    )


class TestScanpyParityMeanDispersion:
    def test_matches_scanpy_top_n(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=1)
        result = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_hvg(path, n_top_genes=10)

        # crispyx's "means" column is the raw (pre-log1p) mean, kept
        # consistent with the seurat_v3 schema (Item 4); scanpy's "seurat"
        # flavor instead overwrites its own "means" column with
        # log1p(mean) internally, so only the normalized statistic and the
        # resulting selection are directly comparable across the two.
        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )
        assert np.allclose(
            result["variances_norm"].to_numpy(),
            expected["dispersions_norm"].to_numpy(),
            atol=1e-5,
        )

    def test_matches_scanpy_thresholds(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=2)
        result = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=None, n_top_genes=None,
            inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_hvg(path, n_top_genes=None)

        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )

    def test_matches_scanpy_csc_source(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csc", seed=3)
        result = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_hvg(path, n_top_genes=10)

        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )


class TestScanpyParitySeuratV3:
    def test_matches_scanpy_top_n(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=21, raw_counts=True)
        result = highly_variable_genes(
            path, flavor="seurat_v3", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_v3_hvg(path, n_top_genes=10)

        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )
        assert np.allclose(result["means"].to_numpy(), expected["means"].to_numpy(), atol=1e-4)
        assert np.allclose(
            result["variances"].to_numpy(), expected["variances"].to_numpy(), atol=1e-3,
        )
        assert np.allclose(
            result["variances_norm"].to_numpy(), expected["variances_norm"].to_numpy(), rtol=1e-3, atol=1e-3,
        )

    def test_matches_scanpy_csc_source(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csc", seed=22, raw_counts=True)
        result = highly_variable_genes(
            path, flavor="seurat_v3", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_v3_hvg(path, n_top_genes=10)

        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )

    def test_default_flavor_is_seurat_v3(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=23, raw_counts=True)
        result_default = highly_variable_genes(
            path, cell_mask=None, n_top_genes=10, inplace=False, verbose=False,
        )
        result_explicit = highly_variable_genes(
            path, flavor="seurat_v3", cell_mask=None, n_top_genes=10, inplace=False, verbose=False,
        )
        assert np.array_equal(
            result_default["highly_variable"].to_numpy(), result_explicit["highly_variable"].to_numpy()
        )

    def test_n_top_genes_required(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=24, raw_counts=True)
        with pytest.raises(ValueError, match="n_top_genes is required"):
            highly_variable_genes(
                path, flavor="seurat_v3", cell_mask=None, n_top_genes=None, verbose=False,
            )

    def test_exact_ties_at_cutoff_select_exactly_n_top_genes(self, tmp_path):
        """Regression test: several genes tied exactly at the n_top_genes
        cutoff must not all be let through.

        scanpy's ``seurat_v3`` selects by an exact rank (``argsort(argsort
        (-x))``), which always returns precisely ``n_top_genes`` -- ties are
        broken by array position, never all included. This is a real
        scenario on production data (many low-count genes can share an
        identical normalized variance), and differs from the ``>=`` cutoff
        the dispersion-based ``mean_dispersion`` flavor legitimately uses.
        """
        rng = np.random.default_rng(42)
        n, g = 200, 60
        base_lambda = rng.uniform(0.5, 20, size=g)
        raw = rng.poisson(base_lambda, size=(n, g)).astype(np.float64)

        # Duplicate the gene that would land exactly at the n_top_genes
        # boundary several times -- genuine exact ties, without perturbing
        # the LOESS fit's numerical stability (unlike near-all-zero genes).
        n_top = 20
        boundary_gene = np.argsort(-raw.var(axis=0, ddof=1))[n_top - 1]
        raw = np.concatenate([raw, np.tile(raw[:, [boundary_gene]], 5)], axis=1)

        obs = pd.DataFrame({"perturbation": pd.Categorical(["NTC"] * n)})
        var = pd.DataFrame(index=[f"g{i}" for i in range(raw.shape[1])])
        path = Path(tmp_path) / "tied_boundary.h5ad"
        ad.AnnData(X=sp.csr_matrix(raw).astype(np.float32), obs=obs, var=var).write_h5ad(path)

        result = highly_variable_genes(
            path, flavor="seurat_v3", cell_mask=None, n_top_genes=n_top, inplace=False, verbose=False,
        )
        expected = _scanpy_seurat_v3_hvg(path, n_top_genes=n_top)

        assert result["highly_variable"].to_numpy().sum() == n_top
        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )


class TestCSCCSREquivalence:
    def test_identical_across_storage_formats_mean_dispersion(self, tmp_path):
        csr_path = _make_hvg_dataset(tmp_path, "csr", seed=4)
        csc_path = _make_hvg_dataset(tmp_path, "csc", seed=4)
        assert get_matrix_storage_format(csr_path) == "csr"
        assert get_matrix_storage_format(csc_path) == "csc"

        r_csr = highly_variable_genes(
            csr_path, flavor="mean_dispersion", cell_mask=None, inplace=False, verbose=False,
        )
        r_csc = highly_variable_genes(
            csc_path, flavor="mean_dispersion", cell_mask=None, inplace=False, verbose=False,
        )

        # float32-on-disk storage plus a different chunking/summation order
        # between the CSC (column-chunked) and CSR (row-accumulated) paths
        # leaves only float32-level precision agreement, not bit-identity.
        assert np.array_equal(r_csr["highly_variable"].to_numpy(), r_csc["highly_variable"].to_numpy())
        assert np.allclose(r_csr["means"].to_numpy(), r_csc["means"].to_numpy(), atol=1e-4)
        assert np.allclose(r_csr["variances"].to_numpy(), r_csc["variances"].to_numpy(), atol=1e-4)
        assert np.allclose(
            r_csr["variances_norm"].to_numpy(), r_csc["variances_norm"].to_numpy(),
            atol=1e-4, equal_nan=True,
        )

    def test_identical_across_storage_formats_seurat_v3(self, tmp_path):
        csr_path = _make_hvg_dataset(tmp_path, "csr", seed=25, raw_counts=True)
        csc_path = _make_hvg_dataset(tmp_path, "csc", seed=25, raw_counts=True)

        r_csr = highly_variable_genes(
            csr_path, flavor="seurat_v3", cell_mask=None, n_top_genes=10, inplace=False, verbose=False,
        )
        r_csc = highly_variable_genes(
            csc_path, flavor="seurat_v3", cell_mask=None, n_top_genes=10, inplace=False, verbose=False,
        )

        assert np.array_equal(r_csr["highly_variable"].to_numpy(), r_csc["highly_variable"].to_numpy())
        assert np.allclose(r_csr["means"].to_numpy(), r_csc["means"].to_numpy(), atol=1e-3)
        assert np.allclose(r_csr["variances"].to_numpy(), r_csc["variances"].to_numpy(), atol=1e-2)
        assert np.allclose(
            r_csr["variances_norm"].to_numpy(), r_csc["variances_norm"].to_numpy(), rtol=1e-2, atol=1e-2,
        )


class TestCellMaskControlDefault:
    def test_default_differs_from_all_cells_and_matches_subset(self, tmp_path):
        path = _make_hvg_dataset(
            tmp_path, "csr", seed=1, g=200, perturbation_effect_gene=0,
        )

        result_control = highly_variable_genes(
            path,
            flavor="mean_dispersion",
            perturbation_column="perturbation",
            control_label="NTC",
            n_top_genes=5,
            inplace=False,
            verbose=False,
        )
        result_all = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=None, n_top_genes=5,
            inplace=False, verbose=False,
        )

        # The perturbation-driven gene dominates the all-cells pass but not
        # the control-only default.
        assert result_all["highly_variable"].to_numpy()[0]
        assert not result_control["highly_variable"].to_numpy()[0]

        # The control-only default must equal a plain scanpy call on cells
        # pre-subset to controls only.
        adata = ad.read_h5ad(path)
        control_adata = adata[adata.obs["perturbation"] == "NTC"].copy()
        expected = sc.pp.highly_variable_genes(
            control_adata, flavor="seurat", n_top_genes=5, inplace=False,
        )
        assert np.array_equal(
            result_control["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )

    def test_explicit_cell_mask_array_matches_manual_subset(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=6)
        adata = ad.read_h5ad(path)
        mask = np.zeros(adata.n_obs, dtype=bool)
        mask[::2] = True

        result = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=mask, n_top_genes=8,
            inplace=False, verbose=False,
        )
        expected = sc.pp.highly_variable_genes(
            adata[mask].copy(), flavor="seurat", n_top_genes=8, inplace=False,
        )
        assert np.array_equal(
            result["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy()
        )


class TestGuardErrors:
    def test_control_default_without_perturbation_column_raises(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=7, raw_counts=True)
        with pytest.raises(ValueError, match="requires perturbation_column"):
            highly_variable_genes(path, n_top_genes=10, verbose=False)

    def test_zero_control_cells_raises(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=8, raw_counts=True)
        with pytest.raises(ValueError, match="No cells found with label"):
            highly_variable_genes(
                path,
                n_top_genes=10,
                perturbation_column="perturbation",
                control_label="NOT_A_REAL_LABEL",
                verbose=False,
            )

    def test_all_false_explicit_cell_mask_raises(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=9, raw_counts=True)
        adata = ad.read_h5ad(path)
        mask = np.zeros(adata.n_obs, dtype=bool)
        with pytest.raises(ValueError, match="cell_mask selects zero cells"):
            highly_variable_genes(path, n_top_genes=10, cell_mask=mask, verbose=False)

    def test_unknown_cell_mask_string_raises(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=10, raw_counts=True)
        with pytest.raises(ValueError, match="cell_mask must be"):
            highly_variable_genes(path, n_top_genes=10, cell_mask="all", verbose=False)

    def test_unknown_flavor_raises(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=26, raw_counts=True)
        with pytest.raises(ValueError, match="Unknown flavor"):
            highly_variable_genes(path, flavor="cell_ranger", cell_mask=None, verbose=False)


class TestEdgeCases:
    def test_constant_gene_is_not_selected_and_does_not_crash(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=11)
        adata = ad.read_h5ad(path)
        adata.X = adata.X.toarray()
        adata.X[:, 0] = 1.0  # constant gene (var == 0 after expm1 too)
        adata.X = sp.csr_matrix(adata.X)
        path2 = Path(tmp_path) / "constant_gene.h5ad"
        adata.write_h5ad(path2)

        result = highly_variable_genes(
            path2, flavor="mean_dispersion", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        assert not result["highly_variable"].to_numpy()[0]
        assert np.isnan(result["variances_norm"].to_numpy()[0])

    def test_constant_gene_seurat_v3_does_not_crash(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=27, raw_counts=True)
        adata = ad.read_h5ad(path)
        adata.X = adata.X.toarray()
        adata.X[:, 0] = 3.0  # constant gene (var == 0)
        adata.X = sp.csr_matrix(adata.X)
        path2 = Path(tmp_path) / "constant_gene_v3.h5ad"
        adata.write_h5ad(path2)

        result = highly_variable_genes(
            path2, flavor="seurat_v3", cell_mask=None, n_top_genes=10,
            inplace=False, verbose=False,
        )
        assert not result["highly_variable"].to_numpy()[0]

    def test_n_top_genes_gte_n_vars_selects_all_without_crash(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=12, g=20)
        result = highly_variable_genes(
            path, flavor="mean_dispersion", cell_mask=None, n_top_genes=1000,
            inplace=False, verbose=False,
        )
        assert result["highly_variable"].to_numpy().sum() <= 20

    def test_chunk_size_invariance_mean_dispersion(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=13)
        results = [
            highly_variable_genes(
                path, flavor="mean_dispersion", cell_mask=None, n_top_genes=10,
                chunk_size=cs, inplace=False, verbose=False,
            )
            for cs in (16, 64, 4096)
        ]
        base = results[0]
        for other in results[1:]:
            assert np.array_equal(base["highly_variable"].to_numpy(), other["highly_variable"].to_numpy())
            assert np.allclose(base["means"].to_numpy(), other["means"].to_numpy(), atol=1e-4)
            assert np.allclose(base["variances"].to_numpy(), other["variances"].to_numpy(), atol=1e-4)

    def test_chunk_size_invariance_seurat_v3(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=28, raw_counts=True)
        results = [
            highly_variable_genes(
                path, flavor="seurat_v3", cell_mask=None, n_top_genes=10,
                chunk_size=cs, inplace=False, verbose=False,
            )
            for cs in (16, 64, 4096)
        ]
        base = results[0]
        for other in results[1:]:
            assert np.array_equal(base["highly_variable"].to_numpy(), other["highly_variable"].to_numpy())
            assert np.allclose(base["means"].to_numpy(), other["means"].to_numpy(), atol=1e-4)
            assert np.allclose(base["variances"].to_numpy(), other["variances"].to_numpy(), atol=1e-4)


class TestInplaceWrite:
    def test_inplace_writes_var_columns(self, tmp_path):
        path = _make_hvg_dataset(tmp_path, "csr", seed=14, raw_counts=True)
        out = highly_variable_genes(path, cell_mask=None, n_top_genes=10, verbose=False)
        assert out is None

        var_df = load_var(path)
        for col in ("highly_variable", "means", "variances", "variances_norm"):
            assert col in var_df.columns
        # Ties at the cutoff can pass more than n_top_genes through the
        # ">= cutoff" selection -- same tie-breaking scanpy itself uses.
        assert var_df["highly_variable"].sum() >= 10

    def test_verbose_prefix(self, tmp_path, capsys):
        path = _make_hvg_dataset(tmp_path, "csr", seed=15, raw_counts=True)
        highly_variable_genes(path, cell_mask=None, n_top_genes=10, verbose=1)
        out = capsys.readouterr().out
        assert "[cx] pp.highly_variable_genes:" in out


class TestDimredIntegration:
    """highly_variable_genes() -> pca(use_highly_variable=True) end to end."""

    def test_pca_restricts_to_exactly_the_selected_genes(self, tmp_path):
        from crispyx.dimred import pca as _pca

        path = _make_hvg_dataset(tmp_path, "csr", seed=30, n=200, g=60, raw_counts=True)
        highly_variable_genes(
            path, flavor="seurat_v3", cell_mask=None, n_top_genes=15, inplace=True, verbose=False,
        )
        mask = load_var(path)["highly_variable"].to_numpy()
        assert mask.sum() == 15

        adata_backed = ad.read_h5ad(path, backed="r")
        try:
            _pca(adata_backed, n_comps=5, use_highly_variable=True, show_progress=False)
            assert adata_backed.uns["pca"]["use_highly_variable"] is True
            pcs = adata_backed.varm["PCs"]
            assert pcs.shape == (60, 5)

            nonzero_rows = np.any(pcs != 0, axis=1)
            assert np.array_equal(nonzero_rows, mask)
        finally:
            adata_backed.file.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_hvg_writes_to_a_file_its_wrapper_holds_open(tmp_path):
    """A crispyx wrapper whose handle is open (any .shape or .obs access opens
    it) must not block the in-place var write, and sees the new column."""
    from crispyx.data import AnnData as CrispyxAnnData

    import crispyx as cx

    for call in (highly_variable_genes, cx.pp.highly_variable_genes):
        wrapper = CrispyxAnnData(_make_hvg_dataset(tmp_path, "csr", name=call.__qualname__))
        assert wrapper.shape == (300, 50)  # opens the read handle
        call(wrapper, flavor="mean_dispersion", n_top_genes=10, cell_mask=None)
        assert int(wrapper.var["highly_variable"].sum()) >= 10  # ties at the cutoff are kept, as in Scanpy


def test_hvg_in_place_on_a_backed_anndata_object_says_to_pass_the_path(tmp_path):
    """Closing its handle would leave its layers dead and its var stale, so it
    is refused up front; inplace=False still works on it."""
    a = ad.read_h5ad(_make_hvg_dataset(tmp_path, "csr"), backed="r")
    with pytest.raises(ValueError, match="pass the path"):
        highly_variable_genes(a, flavor="mean_dispersion", n_top_genes=10, cell_mask=None)
    result = highly_variable_genes(a, flavor="mean_dispersion", n_top_genes=10, cell_mask=None, inplace=False)
    assert result["highly_variable"].sum() >= 10
    assert a.X[0:2].shape == (2, 50)  # the object still works


def test_moments_of_integer_counts_do_not_overflow(tmp_path):
    """Squares of int32 counts above 46340 overflow int32; the moments are taken in float64."""
    X = np.random.default_rng(0).poisson(5, (40, 20)).astype(np.int32)
    X[0, 0] = 60_000
    path = tmp_path / "int.h5ad"
    ad.AnnData(sp.csr_matrix(X)).write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    sums, sqsums, n = _stream_gene_moments(
        backed, storage_format="csr", layer=None, codes=np.zeros(40, dtype=np.int64), n_batches=1,
        transform="identity", chunk_size=16, show_progress=False,
    )
    np.testing.assert_array_equal(sqsums[0], np.square(X.astype(np.float64)).sum(axis=0))


def _batched_counts(tmp_path, fmt: str, *, g: int = 60) -> tuple[Path, ad.AnnData]:
    """Three batches at different depths; one gene expressed in one batch only.

    Counts are high enough that no two genes of a batch share a mean (checked
    in the seurat_v3 test): LOESS on tied means turns last-bit differences in
    the moments (Scanpy 1.12 computes them another way) into percent-sized
    differences in the fit."""
    rng = np.random.default_rng(1)
    n = 240
    batch = np.repeat(["b0", "b1", "b2"], n // 3)
    depth = np.repeat([1.0, 3.0, 0.5], n // 3)[:, None]
    means = rng.gamma(0.6, 2.0, g)[None, :] * depth * 20
    X = rng.negative_binomial(2, 2 / (2 + means)).astype(np.float32)
    X[batch != "b1", 0] = 0
    adata = ad.AnnData(X, obs=pd.DataFrame({"sample": pd.Categorical(batch)}, index=[f"c{i}" for i in range(n)]),
                       var=pd.DataFrame(index=[f"G{j}" for j in range(g)]))
    stored = adata.copy()
    stored.X = sp.csr_matrix(X) if fmt == "csr" else sp.csc_matrix(X)
    path = tmp_path / f"batched_{fmt}.h5ad"
    stored.write_h5ad(path)
    return path, adata


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("flavor", ["seurat_v3", "seurat_v3_paper"])
def test_batch_key_matches_scanpy_seurat_v3(tmp_path, fmt, flavor):
    path, adata = _batched_counts(tmp_path, fmt)
    for batch in ("b0", "b1", "b2"):
        sums = np.asarray(adata.X[(adata.obs["sample"] == batch).to_numpy()].sum(axis=0)).ravel()
        assert len(np.unique(sums[sums > 0])) == (sums > 0).sum(), "tied means make the LOESS fit unstable"
    expected = sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=15, batch_key="sample",
                                           inplace=False)
    got = highly_variable_genes(path, flavor=flavor, n_top_genes=15, batch_key="sample", cell_mask=None,
                                inplace=False, verbose=False)
    np.testing.assert_array_equal(got["highly_variable"], expected["highly_variable"])
    np.testing.assert_array_equal(got["highly_variable_nbatches"], expected["highly_variable_nbatches"])
    np.testing.assert_allclose(got["highly_variable_rank"], expected["highly_variable_rank"])
    np.testing.assert_allclose(got["variances_norm"], expected["variances_norm"], rtol=1e-6)
    np.testing.assert_allclose(got["means"], expected["means"], rtol=1e-6)
    np.testing.assert_allclose(got["variances"], expected["variances"], rtol=1e-5)


@pytest.mark.parametrize("n_top_genes", [15, None])
def test_batch_key_matches_scanpy_mean_dispersion(tmp_path, n_top_genes):
    # Enough genes that no bin holds two (whose normalised dispersions tie
    # at 1/sqrt(2) up to rounding, which then decides the cut).
    path, adata = _batched_counts(tmp_path, "csr", g=400)
    logged = adata.copy()
    sc.pp.normalize_total(logged)
    sc.pp.log1p(logged)
    logged.write_h5ad(path)
    expected = sc.pp.highly_variable_genes(logged, flavor="seurat", n_top_genes=n_top_genes, batch_key="sample",
                                           inplace=False)
    expected = expected.loc[logged.var_names]  # Scanpy returns the threshold case sorted by name
    got = highly_variable_genes(path, flavor="mean_dispersion", n_top_genes=n_top_genes, batch_key="sample",
                                cell_mask=None, inplace=False, verbose=False)
    assert got["highly_variable"].sum() > 0
    np.testing.assert_array_equal(got["highly_variable"], expected["highly_variable"])
    np.testing.assert_array_equal(got["highly_variable_nbatches"], expected["highly_variable_nbatches"])
    np.testing.assert_array_equal(got["highly_variable_intersection"], expected["highly_variable_intersection"])
    np.testing.assert_allclose(got["variances_norm"], expected["dispersions_norm"], rtol=1e-6)


def test_mean_dispersion_batch_of_empty_cells_selects_nothing(tmp_path):
    path, adata = _batched_counts(tmp_path, "csr", g=400)
    logged = adata.copy()
    logged.X[logged.obs["sample"].to_numpy() == "b2"] = 0
    sc.pp.normalize_total(logged)
    sc.pp.log1p(logged)
    logged.write_h5ad(path)
    got = highly_variable_genes(path, flavor="mean_dispersion", n_top_genes=15, batch_key="sample",
                                cell_mask=None, inplace=False, verbose=False)
    assert got["highly_variable"].sum() == 15 and got["highly_variable_nbatches"].max() == 2


def test_batch_key_with_cell_mask_matches_the_subset(tmp_path):
    path, adata = _batched_counts(tmp_path, "csr")
    mask = np.random.default_rng(1).random(adata.n_obs) < 0.7
    expected = sc.pp.highly_variable_genes(adata[mask].copy(), flavor="seurat_v3", n_top_genes=15,
                                           batch_key="sample", inplace=False)
    got = highly_variable_genes(path, n_top_genes=15, batch_key="sample", cell_mask=mask,
                                inplace=False, verbose=False)
    np.testing.assert_array_equal(got["highly_variable"], expected["highly_variable"])
    np.testing.assert_allclose(got["variances_norm"], expected["variances_norm"], rtol=1e-6)


def test_batch_key_writes_scanpy_columns_and_checks_batches(tmp_path):
    path, adata = _batched_counts(tmp_path, "csr")
    highly_variable_genes(path, n_top_genes=15, batch_key="sample", cell_mask=None, verbose=False)
    var = load_var(path)
    assert {"highly_variable_nbatches", "highly_variable_rank"} <= set(var.columns)
    highly_variable_genes(path, n_top_genes=15, cell_mask=None, verbose=False)
    assert "highly_variable_nbatches" not in load_var(path).columns  # the batched run's columns are gone
    single = highly_variable_genes(path, flavor="seurat_v3_paper", n_top_genes=15, cell_mask=None,
                                   inplace=False, verbose=False)
    plain = highly_variable_genes(path, n_top_genes=15, cell_mask=None, inplace=False, verbose=False)
    pd.testing.assert_frame_equal(single, plain)
    assert list(plain.columns) == ["means", "variances", "variances_norm", "highly_variable"]
    mask = np.ones(adata.n_obs, dtype=bool)
    mask[adata.obs["sample"].to_numpy() == "b2"] = False
    mask[-1] = True
    with pytest.raises(ValueError, match="'b2' has fewer than 2 cells"):
        highly_variable_genes(path, n_top_genes=15, batch_key="sample", cell_mask=mask, verbose=False)
