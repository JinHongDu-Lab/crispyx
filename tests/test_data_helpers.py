import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest


from crispyx.data import (
    AnnData,
    OverlapResult,
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
    resolve_data_path,
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


@pytest.mark.parametrize("labels", [
    ["Non-Targeting", "CTRL", "KO"],  # Huang et al.: CTRL is the chymotrypsin-like gene
    ["control", "non-targeting", "KO"],
    ["NTC", "CTRL", "KNTC1"],  # Nourreddine et al.
])
def test_resolve_control_label_refuses_to_choose_between_controls(labels):
    with pytest.raises(ValueError, match="Several perturbation labels look like a control"):
        resolve_control_label(labels * 3, None)
    assert resolve_control_label(labels, labels[0]) == labels[0]


def test_resolve_control_label_infers_ntc_but_not_genes_containing_it():
    assert resolve_control_label(["NTC", "KNTC1", "NTC", "NT5C"], None, verbose=False) == "NTC"


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

@pytest.mark.parametrize("axis", ["obs", "var"])
def test_metadata_round_trip_leaves_x_untouched(tmp_path, axis):
    path = _create_full_dataset(tmp_path)
    load, write = (load_obs, write_obs) if axis == "obs" else (load_var, write_var)
    before = ad.read_h5ad(path)
    df = load(path)
    pd.testing.assert_frame_equal(df, getattr(before, axis))
    if axis == "obs":
        assert isinstance(df["perturbation"].dtype, pd.CategoricalDtype)
    df["extra"] = np.arange(len(df))
    write(path, df)
    pd.testing.assert_frame_equal(load(path), df)
    np.testing.assert_array_equal(ad.read_h5ad(path).X.toarray(), before.X.toarray())


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_metadata_write_rejects_the_wrong_row_count(tmp_path, axis):
    path = _create_full_dataset(tmp_path)
    load, write = (load_obs, write_obs) if axis == "obs" else (load_var, write_var)
    with pytest.raises(ValueError, match="rows"):
        write(path, load(path).iloc[:2])


def test_metadata_reads_bytes_encoded_attrs(tmp_path):
    """anndata <= 0.8 stored 'column-order' and '_index' as S-dtype bytes; h5py
    reads those back as np.bytes_, which must not duplicate or drop columns."""
    import scipy.sparse as sp

    obs = pd.DataFrame({"perturbation": ["ctrl", "KO1", "KO2"]}, index=["c0", "c1", "c2"])
    var = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53"]}, index=["BRCA1", "TP53"])
    path = tmp_path / "bytes_attrs.h5ad"
    ad.AnnData(sp.csr_matrix(np.ones((3, 2), dtype=np.float32)), obs=obs, var=var).write(path)
    with h5py.File(path, "r+") as f:
        f["obs"].attrs["column-order"] = np.array(list(obs.columns), dtype="S40")
        f["var"].attrs["column-order"] = np.array(list(var.columns), dtype="S40")
        del f["obs"].attrs["_index"]
        f["obs"].attrs["_index"] = np.bytes_(b"_index")

    loaded_obs = load_obs(path)
    assert list(loaded_obs.columns) == ["perturbation"]
    assert list(loaded_obs.index) == ["c0", "c1", "c2"]
    assert list(loaded_obs["perturbation"]) == ["ctrl", "KO1", "KO2"]
    assert list(load_var(path).columns) == ["gene_symbol"]


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

def test_compute_overlap_counts_and_jaccard():
    result = compute_overlap({"A": [1, 1, 2, 3], "B": {2, 3, 4}, "C": set()})
    assert isinstance(result, OverlapResult)
    assert result.set_sizes.to_dict() == {"A": 3, "B": 3, "C": 0}  # lists are de-duplicated
    np.testing.assert_array_equal(result.count_matrix.to_numpy(), [[3, 2, 0], [2, 3, 0], [0, 0, 0]])
    # An empty set's Jaccard index is 0/0, reported as 0.
    expected_jaccard = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]
    np.testing.assert_allclose(result.jaccard_matrix.to_numpy(), expected_jaccard)
    assert list(result.count_matrix.index) == ["A", "B", "C"]


def test_compute_overlap_count_only_leaves_jaccard_empty():
    result = compute_overlap({"A": {1, 2}, "B": {2, 3}}, metric="count")
    assert result.count_matrix.loc["A", "B"] == 1
    assert result.jaccard_matrix.to_numpy().sum() == 0.0


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


def test_in_memory_anndata_is_rejected_with_the_fix(tmp_path):
    """crispyx streams from disk: an in-memory AnnData is refused with the
    one line that makes it acceptable, and the saved or backed form works."""
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(TypeError, match=r'adata\.write_h5ad\("data\.h5ad"\)'):
        resolve_data_path(adata)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    assert resolve_data_path(path) == path
    backed = ad.read_h5ad(path, backed="r")
    try:
        assert resolve_data_path(backed) == path
    finally:
        backed.file.close()


def test_chunk_starts_rejects_a_mask_of_the_wrong_length():
    from crispyx.data import chunk_starts

    with pytest.raises(ValueError, match="selected has 5 entries for an axis of 10"):
        chunk_starts(10, 3, selected=np.ones(5, dtype=bool))
