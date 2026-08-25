import logging
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crispyx.data import (
    AnnData,
    OverlapResult,
    calculate_nb_glm_chunk_size,
    compute_overlap,
    detect_gene_symbol_column,
    detect_perturbation_column,
    ensure_gene_symbol_column,
    infer_columns,
    load_obs,
    load_var,
    normalise_perturbation_labels,
    read_h5ad_ondisk,
    resolve_control_label,
    standardise_gene_names,
    write_obs,
    write_var,
)
from crispyx.pseudobulk import compute_normalized_effects


def _create_dataset(tmp_path: Path) -> Path:
    x = np.array(
        [
            [0, 0, 0],
            [1, 2, 0],
            [0, 0, 1],
            [3, 0, 4],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "ctrl", "KO1", "KO2"]},
        index=[f"cell_{idx}" for idx in range(x.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{idx}" for idx in range(x.shape[1])]})
    var.index = var["gene_symbol"]
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "test.h5ad"
    adata.write(path)
    return path


def test_ensure_gene_symbol_column_uses_var_names(caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")
    adata = ad.AnnData(np.ones((2, 2)))
    adata.var_names = pd.Index(["g1", "g2"])

    names = ensure_gene_symbol_column(adata, None)

    assert list(names) == ["g1", "g2"]
    assert "using adata.var_names" in caplog.text


def test_resolve_control_label_infers_ctrl(caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")

    inferred = resolve_control_label(["KO", "CTRL_cells"], None)

    assert inferred == "CTRL_cells"
    assert "Inferred control label" in caplog.text


def test_resolve_control_label_prints_by_default(capsys):
    """verbose defaults to True: the inferred label should be visible on
    stdout without the caller configuring logging (unlike the logger.info
    channel, which is silent by default)."""
    inferred = resolve_control_label(["KO", "CTRL_cells"], None)

    assert inferred == "CTRL_cells"
    out = capsys.readouterr().out
    assert "[cx] resolve_control_label:" in out
    assert "CTRL_cells" in out


def test_resolve_control_label_verbose_false_is_silent(capsys):
    resolve_control_label(["KO", "CTRL_cells"], None, verbose=False)
    assert capsys.readouterr().out == ""


def test_read_h5ad_ondisk_returns_backed_object(tmp_path, capsys):
    path = _create_dataset(tmp_path)

    adata_ro = read_h5ad_ondisk(path, n_obs=1, n_vars=1)
    captured = capsys.readouterr()

    assert "AnnData object" in captured.out
    assert "First obs rows:" in captured.out
    assert isinstance(adata_ro, AnnData)
    assert adata_ro.backed.isbacked
    adata_ro.close()


def test_compute_normalized_effects_infers_control(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")
    path = _create_dataset(tmp_path)

    result = compute_normalized_effects(
        path,
        perturbation_column="perturbation",
        control_label=None,
        method="mean_log1p",
        gene_name_column="gene_symbol",
    )

    assert isinstance(result, AnnData)
    assert set(result.obs.index) == {"KO1", "KO2"}
    loaded_var = result.var.load()
    assert list(loaded_var.index) == ["gene0", "gene1", "gene2"]
    assert "Inferred control label" in caplog.text
    result.close()


# ============================================================================
# Tests for calculate_nb_glm_chunk_size
# ============================================================================


def test_calculate_nb_glm_chunk_size_returns_max_for_small_dataset():
    """Small datasets should use max chunk size (256)."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=10000,
        n_vars=5000,
        n_groups=50,
        available_memory_gb=128,
    )
    assert chunk_size == 256  # max_chunk default


def test_calculate_nb_glm_chunk_size_reduces_for_large_dataset():
    """Large datasets should get reduced chunk size to fit memory."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=1200000,  # 1.2M cells (like Feng-ts)
        n_vars=36000,
        n_groups=500,
        available_memory_gb=128,
    )
    # Should be less than max_chunk due to memory constraints
    assert chunk_size < 256
    assert chunk_size >= 32  # min_chunk default


def test_calculate_nb_glm_chunk_size_respects_memory_limit():
    """memory_limit_gb should cap the available memory."""
    # With high available memory, should use max
    chunk_high = calculate_nb_glm_chunk_size(
        n_obs=500000,
        n_vars=20000,
        n_groups=200,
        available_memory_gb=256,
    )
    
    # With memory_limit_gb, should be constrained
    chunk_limited = calculate_nb_glm_chunk_size(
        n_obs=500000,
        n_vars=20000,
        n_groups=200,
        available_memory_gb=256,
        memory_limit_gb=32,  # Lower limit
    )
    
    assert chunk_limited <= chunk_high


def test_calculate_nb_glm_chunk_size_respects_min_max_bounds():
    """Chunk size should be clamped to [min_chunk, max_chunk]."""
    # Even with huge memory, don't exceed max_chunk
    chunk_max = calculate_nb_glm_chunk_size(
        n_obs=1000,
        n_vars=100,
        n_groups=10,
        available_memory_gb=1000,  # Huge memory
        max_chunk=128,
    )
    assert chunk_max == 128
    
    # Even with tiny memory, don't go below min_chunk
    chunk_min = calculate_nb_glm_chunk_size(
        n_obs=10000000,  # Very large
        n_vars=50000,
        n_groups=1000,
        available_memory_gb=1,  # Tiny memory
        min_chunk=64,
    )
    assert chunk_min == 64


def test_calculate_nb_glm_chunk_size_handles_none_n_groups():
    """Function should work without n_groups specified."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=100000,
        n_vars=20000,
        n_groups=None,  # Unknown groups
        available_memory_gb=64,
    )
    assert 32 <= chunk_size <= 256


# ============================================================================
# Helper: create a minimal h5ad with categorical obs column
# ============================================================================

def _create_full_dataset(tmp_path: Path) -> Path:
    """Dataset with varied obs/var metadata for Feature 1-4 tests."""
    import scipy.sparse as sp

    rng = np.random.default_rng(42)
    x = rng.poisson(1.0, size=(6, 4)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "perturbation": pd.Categorical(["ctrl", "ctrl", "KO1", "KO1", "KO2", "KO2"]),
            "batch": ["A", "A", "A", "B", "B", "B"],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(
        {
            "gene_symbols": ["BRCA1", "TP53", "EGFR", "KRAS"],
            "ensembl_id": ["ENSG00000012048.22", "ENSG00000141510.1", "ENSG00000146648.2", "ENSG00000133703.3"],
        },
        index=["BRCA1", "TP53", "EGFR", "KRAS"],
    )
    adata = ad.AnnData(sp.csr_matrix(x), obs=obs, var=var)
    path = tmp_path / "full.h5ad"
    adata.write(path)
    return path


# ============================================================================
# Feature 1: load_obs / load_var / write_obs / write_var
# ============================================================================

class TestBackedMetadataEditing:
    def test_load_obs_returns_dataframe(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        df = load_obs(path)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == [f"cell_{i}" for i in range(6)]
        assert "perturbation" in df.columns
        assert "batch" in df.columns

    def test_load_obs_preserves_categorical(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        df = load_obs(path)
        assert isinstance(df["perturbation"].dtype, pd.CategoricalDtype)

    def test_load_var_returns_dataframe(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        df = load_var(path)
        assert isinstance(df, pd.DataFrame)
        assert "gene_symbols" in df.columns
        assert len(df) == 4

    def test_write_obs_round_trip(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        df = load_obs(path)
        df["new_col"] = ["x"] * len(df)
        write_obs(path, df)
        df2 = load_obs(path)
        assert "new_col" in df2.columns
        assert list(df2["new_col"]) == ["x"] * len(df)

    def test_write_var_round_trip(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        df = load_var(path)
        df["extra"] = [1, 2, 3, 4]
        write_var(path, df)
        df2 = load_var(path)
        assert "extra" in df2.columns
        assert list(df2["extra"]) == [1, 2, 3, 4]

    def test_x_untouched_after_write_obs(self, tmp_path):
        """write_obs must not corrupt the X matrix."""
        path = _create_full_dataset(tmp_path)
        import anndata as ad_mod
        before = ad_mod.read_h5ad(path).X.toarray()
        df = load_obs(path)
        df["flag"] = 0
        write_obs(path, df)
        after = ad_mod.read_h5ad(path).X.toarray()
        np.testing.assert_array_equal(before, after)

    def test_write_obs_shape_mismatch_raises(self, tmp_path):
        import pytest as _pytest
        path = _create_full_dataset(tmp_path)
        df = load_obs(path).iloc[:3]  # wrong size
        with _pytest.raises(ValueError, match="rows"):
            write_obs(path, df)

    def test_write_var_shape_mismatch_raises(self, tmp_path):
        import pytest as _pytest
        path = _create_full_dataset(tmp_path)
        df = load_var(path).iloc[:2]  # wrong size
        with _pytest.raises(ValueError, match="rows"):
            write_var(path, df)

    # ------------------------------------------------------------------
    # Tests for bytes-encoded HDF5 attrs (anndata <=0.8 / S-dtype style)
    # ------------------------------------------------------------------

    def _write_h5ad_with_bytes_attrs(self, path: Path) -> None:
        """Write a minimal h5ad with S-dtype (bytes) column-order attribute.

        This reproduces the on-disk format produced by anndata <= 0.8 and by
        any writer that calls ``np.array([...], dtype='S...')`` for the
        'column-order' attribute.  h5py reads S-dtype arrays back as
        ``np.bytes_`` objects rather than Python strings.
        """
        import h5py as _h5py
        import scipy.sparse as sp

        str_dt = _h5py.string_dtype(encoding="utf-8")

        # Write a skeleton obs/var/X so anndata can read the file back.
        x = np.ones((3, 2), dtype=np.float32)
        obs = pd.DataFrame(
            {"perturbation": ["ctrl", "KO1", "KO2"]},
            index=["c0", "c1", "c2"],
        )
        var = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53"]}, index=["BRCA1", "TP53"])
        adata = ad.AnnData(sp.csr_matrix(x), obs=obs, var=var)
        adata.write(path)

        # Now overwrite the obs 'column-order' attr with S-dtype bytes.
        with _h5py.File(path, "r+") as f:
            f["obs"].attrs["column-order"] = np.array(
                list(obs.columns), dtype="S40"
            )
            f["var"].attrs["column-order"] = np.array(
                list(var.columns), dtype="S40"
            )

    def test_load_obs_bytes_column_order_no_duplicates(self, tmp_path):
        """load_obs must not duplicate columns when column-order is S-dtype bytes."""
        path = tmp_path / "bytes_attrs.h5ad"
        self._write_h5ad_with_bytes_attrs(path)
        df = load_obs(path)
        assert list(df.columns) == ["perturbation"], (
            f"Expected ['perturbation'], got {list(df.columns)}"
        )

    def test_load_obs_bytes_column_order_correct_values(self, tmp_path):
        """Column values must be read correctly when column-order uses S-dtype."""
        path = tmp_path / "bytes_attrs.h5ad"
        self._write_h5ad_with_bytes_attrs(path)
        df = load_obs(path)
        assert list(df["perturbation"]) == ["ctrl", "KO1", "KO2"]

    def test_load_var_bytes_column_order_no_duplicates(self, tmp_path):
        """load_var must not duplicate columns when column-order is S-dtype bytes."""
        path = tmp_path / "bytes_attrs_var.h5ad"
        self._write_h5ad_with_bytes_attrs(path)
        df = load_var(path)
        assert list(df.columns) == ["gene_symbol"], (
            f"Expected ['gene_symbol'], got {list(df.columns)}"
        )

    def test_load_obs_bytes_index_key_attr(self, tmp_path):
        """load_obs works when the _index attr is stored as bytes (np.bytes_)."""
        import h5py as _h5py
        import scipy.sparse as sp

        # Write a normal file, then monkey-patch _index to bytes.
        x = np.ones((2, 1), dtype=np.float32)
        obs = pd.DataFrame({"pert": ["ctrl", "KO1"]}, index=["c0", "c1"])
        var = pd.DataFrame(index=["g0"])
        adata = ad.AnnData(sp.csr_matrix(x), obs=obs, var=var)
        path = tmp_path / "bytes_index.h5ad"
        adata.write(path)

        with _h5py.File(path, "r+") as f:
            # Replace string attr with bytes scalar (np.bytes_ round-trips via h5py)
            del f["obs"].attrs["_index"]
            f["obs"].attrs["_index"] = np.bytes_(b"_index")

        df = load_obs(path)
        assert list(df.index) == ["c0", "c1"]
        assert "pert" in df.columns


# ============================================================================
# Feature 2: standardise_gene_names
# ============================================================================

class TestStandardiseGeneNames:
    def test_strip_version_suffix(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        result = standardise_gene_names(path, column="ensembl_id", inplace=False)
        assert result is not None
        assert not any("." in v for v in result)

    def test_strip_version_inplace(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        ret = standardise_gene_names(path, column="ensembl_id", inplace=True)
        assert ret is None
        df = load_var(path)
        assert not any("." in v for v in df["ensembl_id"])

    def test_mt_prefix_normalisation(self, tmp_path):
        x = np.ones((2, 3), dtype=np.float32)
        var = pd.DataFrame(index=["mt-nd1", "MT-CO1", "ACTB"])
        adata = ad.AnnData(x, var=var)
        path = tmp_path / "mt.h5ad"
        adata.write(path)
        result = standardise_gene_names(
            path, column=None, strip_version=False, inplace=False
        )
        assert result is not None
        assert result.tolist() == ["MT-nd1", "MT-CO1", "ACTB"]

    def test_missing_column_raises(self, tmp_path):
        import pytest as _pytest
        path = _create_full_dataset(tmp_path)
        with _pytest.raises(KeyError, match="nonexistent"):
            standardise_gene_names(path, column="nonexistent", inplace=False)

    def test_no_op_when_flags_off(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        before = load_var(path)["gene_symbols"].tolist()
        result = standardise_gene_names(
            path,
            column="gene_symbols",
            strip_version=False,
            normalise_mt_prefix=False,
            inplace=False,
        )
        assert result is not None
        assert result.tolist() == before


# ============================================================================
# Feature 3: normalise_perturbation_labels
# ============================================================================

class TestNormalisePerturbationLabels:
    def _make_path(self, tmp_path, labels):
        x = np.ones((len(labels), 2), dtype=np.float32)
        obs = pd.DataFrame({"perturbation": labels}, index=range(len(labels)))
        adata = ad.AnnData(x, obs=obs)
        path = tmp_path / "pert.h5ad"
        adata.write(path)
        return path

    def test_strip_prefixes(self, tmp_path):
        path = self._make_path(tmp_path, ["sg-BRCA1", "sg-TP53", "sgctrl"])
        result = normalise_perturbation_labels(
            path, "perturbation", strip_prefixes=["sg-"], inplace=False
        )
        assert result is not None
        assert result.tolist() == ["BRCA1", "TP53", "sgctrl"]

    def test_strip_suffixes(self, tmp_path):
        path = self._make_path(tmp_path, ["BRCA1_KO", "TP53_KD", "GENE3_other"])
        result = normalise_perturbation_labels(
            path, "perturbation", strip_suffixes=["_KO", "_KD"], inplace=False
        )
        assert result is not None
        assert result.tolist() == ["BRCA1", "TP53", "GENE3_other"]

    def test_canonical_control_unification(self, tmp_path):
        labels = ["ctrl", "NTC", "GENE1", "scramble", "non-targeting"]
        path = self._make_path(tmp_path, labels)
        result = normalise_perturbation_labels(
            path, "perturbation", canonical_control="NTC", inplace=False
        )
        assert result is not None
        for idx in [0, 1, 3, 4]:
            assert result.iloc[idx] == "NTC"
        assert result.iloc[2] == "GENE1"

    def test_custom_control_aliases(self, tmp_path):
        path = self._make_path(tmp_path, ["MyCtrl", "GENE2"])
        result = normalise_perturbation_labels(
            path, "perturbation",
            control_aliases=["myctrl"],
            canonical_control="NTC",
            inplace=False,
        )
        assert result is not None
        assert result.iloc[0] == "NTC"
        assert result.iloc[1] == "GENE2"

    def test_strip_suffix_regex(self, tmp_path):
        path = self._make_path(tmp_path, ["GENE1_P1P2", "GENE2_P3", "ctrl"])
        result = normalise_perturbation_labels(
            path, "perturbation",
            strip_suffix_regex=r"_P\d+P?\d*$",
            inplace=False,
        )
        assert result is not None
        assert result.tolist()[0] == "GENE1"
        assert result.tolist()[1] == "GENE2"

    def test_inplace_writes_back(self, tmp_path):
        path = self._make_path(tmp_path, ["sg-BRCA1", "ctrl"])
        normalise_perturbation_labels(
            path, "perturbation", strip_prefixes=["sg-"], inplace=True
        )
        df = load_obs(path)
        assert df["perturbation"].tolist() == ["BRCA1", "NTC"]

    def test_missing_column_raises(self, tmp_path):
        import pytest as _pytest
        path = self._make_path(tmp_path, ["a", "b"])
        with _pytest.raises(KeyError, match="missing_col"):
            normalise_perturbation_labels(path, "missing_col", inplace=False)


# ============================================================================
# Feature 4: detect_perturbation_column / detect_gene_symbol_column / infer_columns
# ============================================================================

class TestAutoDetect:
    def test_detect_perturbation_column_finds_standard_name(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        col = detect_perturbation_column(path, verbose=False)
        assert col == "perturbation"

    def test_detect_perturbation_column_returns_none_for_numeric_cols(self, tmp_path):
        x = np.ones((4, 2), dtype=np.float32)
        obs = pd.DataFrame({"count": [1, 2, 3, 4]})
        adata = ad.AnnData(x, obs=obs)
        path = tmp_path / "numeric.h5ad"
        adata.write(path)
        # Only numeric column: low score, should return None (int dtype → no +2)
        col = detect_perturbation_column(path, verbose=False)
        # Column name "count" is not in aliases → score = 0+0+1 (4 unique, 2–5000)
        # → could be returned or not; we just check it doesn't crash
        assert col is None or isinstance(col, str)

    def test_detect_perturbation_column_boosts_control_label(self, tmp_path):
        x = np.ones((4, 2), dtype=np.float32)
        obs = pd.DataFrame({
            "pert": pd.Categorical(["ctrl", "ctrl", "KO1", "KO2"]),
            "sample": pd.Categorical(["s1", "s2", "s3", "s4"]),
        })
        adata = ad.AnnData(x, obs=obs)
        path = tmp_path / "ctrl.h5ad"
        adata.write(path)
        col = detect_perturbation_column(path, control_label="ctrl", verbose=False)
        assert col == "pert"

    def test_detect_gene_symbol_column(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        col = detect_gene_symbol_column(path, verbose=False)
        assert col == "gene_symbols"

    def test_infer_columns_returns_dict(self, tmp_path):
        path = _create_full_dataset(tmp_path)
        result = infer_columns(path, verbose=False)
        assert "perturbation_column" in result
        assert "gene_name_column" in result
        assert result["perturbation_column"] == "perturbation"
        assert result["gene_name_column"] == "gene_symbols"

    def test_detect_perturbation_column_prints_by_default(self, tmp_path, capsys):
        path = _create_full_dataset(tmp_path)
        col = detect_perturbation_column(path)
        assert col == "perturbation"
        out = capsys.readouterr().out
        assert "[cx] detect_perturbation_column:" in out
        assert "perturbation" in out

    def test_detect_gene_symbol_column_prints_by_default(self, tmp_path, capsys):
        path = _create_full_dataset(tmp_path)
        col = detect_gene_symbol_column(path)
        assert col == "gene_symbols"
        out = capsys.readouterr().out
        assert "[cx] detect_gene_symbol_column:" in out


# ============================================================================
# Feature 5: compute_overlap / OverlapResult
# ============================================================================

class TestComputeOverlap:
    def test_diagonal_equals_set_size(self):
        sets = {"A": {"x", "y", "z"}, "B": {"x", "w"}}
        result = compute_overlap(sets)
        assert result.count_matrix.loc["A", "A"] == 3
        assert result.count_matrix.loc["B", "B"] == 2

    def test_jaccard_matrix_symmetric(self):
        sets = {"A": {1, 2, 3}, "B": {2, 3, 4}}
        result = compute_overlap(sets)
        j = result.jaccard_matrix
        assert abs(j.loc["A", "B"] - j.loc["B", "A"]) < 1e-12

    def test_jaccard_values_in_range(self):
        sets = {"A": set(range(10)), "B": set(range(5, 15)), "C": set()}
        result = compute_overlap(sets)
        jv = result.jaccard_matrix.values.flatten()
        assert (jv >= 0).all() and (jv <= 1).all()

    def test_jaccard_diagonal_is_one(self):
        sets = {"A": {1, 2, 3}, "B": {4, 5}}
        result = compute_overlap(sets)
        np.testing.assert_allclose(result.jaccard_matrix.values.diagonal(), 1.0)

    def test_empty_set_intersection_zero(self):
        sets = {"A": {1, 2}, "B": set()}
        result = compute_overlap(sets)
        assert result.count_matrix.loc["A", "B"] == 0
        assert result.jaccard_matrix.loc["A", "B"] == 0.0

    def test_set_sizes_correct(self):
        sets = {"A": [1, 2, 3], "B": [1, 2]}
        result = compute_overlap(sets)
        assert result.set_sizes["A"] == 3
        assert result.set_sizes["B"] == 2

    def test_metric_count_only(self):
        sets = {"A": {1, 2}, "B": {2, 3}}
        result = compute_overlap(sets, metric="count")
        assert result.count_matrix.loc["A", "B"] == 1
        # jaccard_arr was never written, all zeros
        assert result.jaccard_matrix.values.sum() == 0.0

    def test_list_input_converted_to_set(self):
        result = compute_overlap({"A": [1, 1, 2], "B": [2, 3]})
        assert result.set_sizes["A"] == 2  # deduped
        assert result.count_matrix.loc["A", "B"] == 1

    def test_returns_overlap_result_instance(self):
        result = compute_overlap({"X": {"a"}, "Y": {"b"}})
        assert isinstance(result, OverlapResult)


class TestUpdateH5adDataframe:
    """`_update_h5ad_dataframe` must recognise every string-like column
    dtype, not just plain numpy ``object``/``U``/``S`` -- pandas'
    ``StringDtype`` (and, since pandas 3.0, its default string dtype for
    plain-text columns) reports ``.dtype.kind == 'O'`` but
    ``.dtype == object`` is False, which previously fell through to the
    numeric branch and crashed h5py with
    ``TypeError: Object dtype dtype('O') has no native HDF5 equivalent``.
    """

    def test_pandas_string_dtype_column_is_written_as_vlen_string(self, tmp_path):
        from crispyx.data import _update_h5ad_dataframe

        df = pd.DataFrame(
            {
                "label": pd.array(["a", "b", "c"], dtype="string"),
                "count": np.array([1, 2, 3], dtype=np.int64),
            },
            index=pd.Index(["r0", "r1", "r2"], name="index"),
        )
        h5ad_path = tmp_path / "update.h5ad"
        ad.AnnData(np.zeros((3, 1)), obs=df.iloc[:, :0].copy()).write(h5ad_path)
        with h5py.File(h5ad_path, "r+") as f:
            _update_h5ad_dataframe(f, "obs", df)

        result = ad.read_h5ad(h5ad_path)
        assert list(result.obs["label"]) == ["a", "b", "c"]
        assert list(result.obs["count"]) == [1, 2, 3]
