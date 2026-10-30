"""pp.check_data: the mistakes that mislead an analysis without failing it."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx.data import normalise_perturbation_labels


def _write(path: Path, labels, *, genes=None, n_counts=True, x=None) -> Path:
    n = len(labels)
    genes = genes if genes is not None else ["NKX2-5", "GATA4", "TBX5", "HLA-B", "CTRL", "MYH6"]
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, (n, len(genes))).astype(np.float32) if x is None else x
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(n)])
    if n_counts:
        obs["n_counts"] = counts.sum(axis=1)
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes)).write(path)
    return path


def _messages(issues, level):
    return [i.message for i in issues if i.level == level]


def test_clean_file_has_no_issues(tmp_path):
    path = _write(tmp_path / "ok.h5ad", ["control", "NKX2-5", "GATA4"] * 4)
    assert cx.pp.check_data(path, perturbation_column="perturbation", verbose=False) == []


def test_label_problems_are_reported(tmp_path):
    labels = ["control", "NTC", "Gata4", "GATA4", "TBX5_GTGGGAGCCTGGGGGTGAGG", "sg HLA-B", "MYH6_2", "CTRL", "ZZZ9"] * 2
    path = _write(tmp_path / "labels.h5ad", labels)
    issues = cx.pp.check_data(path, perturbation_column="perturbation", control_label="control", verbose=False)
    errors, warnings = _messages(issues, "error"), _messages(issues, "warning")
    assert any("control synonyms" in m and "'NTC'" in m for m in errors)
    assert any("differing only by case" in m for m in errors)
    assert any("spacer" in m for m in errors)
    assert any("whitespace or a separator" in m for m in errors)
    assert any("read like a control but are also measured genes" in m and "CTRL" in m for m in warnings)
    assert any("underscore" in m and "MYH6_2" in m for m in warnings)
    assert any("name no gene in var" in m and "ZZZ9" in m for m in warnings)
    with pytest.raises(ValueError, match="error\\(s\\)"):
        cx.raise_on_errors(issues)


def test_missing_column_control_and_labels(tmp_path):
    path = _write(tmp_path / "x.h5ad", ["control", "GATA4", None, "nan"] * 3)
    issues = cx.pp.check_data(path, perturbation_column="nope", verbose=False)
    assert any("no column 'nope'" in m for m in _messages(issues, "error"))
    issues = cx.pp.check_data(path, perturbation_column="perturbation", control_label="NTC", verbose=False)
    assert any("no cell carries the control label 'NTC'" in m for m in _messages(issues, "error"))
    assert any("6 cells have no perturbation label" in m for m in _messages(issues, "warning"))


def test_ambiguous_control_inference_is_an_error(tmp_path):
    path = _write(tmp_path / "x.h5ad", ["control", "NTC", "GATA4"] * 3)
    issues = cx.pp.check_data(path, perturbation_column="perturbation", verbose=False)
    assert any("look like a control" in m for m in _messages(issues, "error"))


def test_gene_and_count_problems(tmp_path):
    genes = ["ENSG00000183072", "ENSG00000136574", "ENSG00000089225", "ENSG00000234745", "ENSG00000141510", "ENSG00000197616"]
    path = _write(tmp_path / "x.h5ad", ["control", "GATA4"] * 6, genes=genes)
    var = cx.load_var(path)
    var["symbol"] = ["A", "B", "C", "D", "E", "F"]
    cx.write_var(path, var)
    issues = cx.pp.check_data(path, perturbation_column="perturbation", verbose=False)
    assert any("Ensembl IDs" in m and "gene_name_column='symbol'" in m for m in _messages(issues, "warning"))
    # duplicate var_names
    var.index = ["A", "A", "C", "D", "E", "F"]
    cx.write_var(path, var)
    issues = cx.pp.check_data(path, perturbation_column="perturbation", verbose=False)
    assert any("duplicate var_names" in m for m in _messages(issues, "error"))
    # non-integer X and a row sum above n_counts
    with h5py.File(path, "r+") as f:
        f["X/data"][0] = 0.5
    obs = cx.load_obs(path)
    obs["n_counts"] = obs["n_counts"].astype(float) - 3
    cx.write_obs(path, obs)
    issues = cx.pp.check_data(path, perturbation_column="perturbation", verbose=False)
    assert any("does not hold integer counts" in m for m in _messages(issues, "warning"))
    assert any("row sum above their 'n_counts'" in m for m in _messages(issues, "error"))


def test_inconsistent_pointers_are_an_error(tmp_path):
    path = _write(tmp_path / "x.h5ad", ["control", "GATA4"] * 6)
    with h5py.File(path, "r+") as f:
        indptr = f["X/indptr"][...]
        del f["X/indptr"]
        f["X"].create_dataset("indptr", data=indptr[:-2])
    issues = cx.pp.check_data(path, perturbation_column="perturbation", verbose=False)
    assert any("not written completely" in m for m in _messages(issues, "error"))


def test_verbose_prints_each_issue(tmp_path, capsys):
    path = _write(tmp_path / "x.h5ad", ["control", "NTC", "GATA4"] * 3)
    cx.pp.check_data(path, perturbation_column="perturbation", control_label="control")
    out = capsys.readouterr().out
    assert "error: control synonyms" in out and "errors, " in out


# --- normalise_perturbation_labels(target_pattern=...) -------------------------


def test_target_pattern_keeps_hyphenated_symbols_whole(tmp_path):
    labels = ["NKX2-5_-_172659581.23-P1P2", "HLA-B_+_31355001.23-P1", "non-targeting_00006", "GATA4_-_1.23-P1"]
    path = _write(tmp_path / "x.h5ad", labels)
    got = normalise_perturbation_labels(
        path, "perturbation", strip_suffix_regex=r"_\d+$", target_pattern=r"^(.+?)_[+-]_\d+",
        canonical_control="control", inplace=False,
    )
    assert got.tolist() == ["NKX2-5", "HLA-B", "control", "GATA4"]
    with pytest.raises(ValueError, match="exactly one capture group"):
        normalise_perturbation_labels(path, "perturbation", target_pattern=r"^.+?_[+-]_\d+", inplace=False)
