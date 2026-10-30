"""pp.assign_guides: the dominant guide per cell, or NA."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx

GUIDES = ["sgA", "sgB", "sgC"]
#            A   B   C
UMIS = np.array([
    [10, 0, 0],   # clear
    [10, 5, 0],   # exactly 2x: assigned
    [10, 6, 0],   # under 2x: ambiguous
    [2, 0, 0],    # below min_umi
    [0, 0, 0],    # no guide
    [4, 4, 0],    # tie
    [0, 0, 3],    # exactly min_umi
], dtype=np.float32)
EXPECTED = ["sgA", "sgA", pd.NA, pd.NA, pd.NA, pd.NA, "sgC"]


def _check(series):
    assert list(series.index) == list(range(len(UMIS))) or len(series) == len(UMIS)
    got = [v if v is pd.NA else str(v) for v in series.to_numpy()]
    assert got == EXPECTED


def test_dense_and_sparse_arrays_agree():
    _check(cx.pp.assign_guides(UMIS, GUIDES, verbose=False))
    _check(cx.pp.assign_guides(sp.csr_matrix(UMIS), GUIDES, verbose=False))


def test_anndata_in_memory_and_on_disk(tmp_path):
    a = ad.AnnData(sp.csr_matrix(UMIS), obs=pd.DataFrame(index=[f"c{i}" for i in range(len(UMIS))]),
                   var=pd.DataFrame(index=GUIDES))
    got = cx.pp.assign_guides(a, verbose=False)
    assert list(got.index) == list(a.obs_names)
    _check(got)
    a.write(tmp_path / "guides.h5ad")
    streamed = cx.pp.assign_guides(tmp_path / "guides.h5ad", chunk_size=2, verbose=False)
    assert list(streamed.index) == list(a.obs_names)
    _check(streamed)


def test_thresholds_and_single_guide_library():
    one = cx.pp.assign_guides(UMIS[:, :1], ["sgA"], min_umi=3, verbose=False)
    assert one.tolist()[:4] == ["sgA", "sgA", "sgA", pd.NA]
    relaxed = cx.pp.assign_guides(UMIS, GUIDES, min_umi=1, min_ratio=1.0, verbose=False)
    assert relaxed.iloc[2] == "sgA" and relaxed.iloc[3] == "sgA"
    assert relaxed.iloc[5] is pd.NA  # a tie never dominates


def test_arguments_are_validated():
    with pytest.raises(ValueError, match="guide_names is required"):
        cx.pp.assign_guides(UMIS, verbose=False)
    with pytest.raises(ValueError, match="guide names for"):
        cx.pp.assign_guides(UMIS, ["a", "b"], verbose=False)
    with pytest.raises(ValueError, match="min_umi"):
        cx.pp.assign_guides(UMIS, GUIDES, min_umi=0, verbose=False)
    with pytest.raises(ValueError, match="min_ratio"):
        cx.pp.assign_guides(UMIS, GUIDES, min_ratio=0.5, verbose=False)


def test_summary_is_printed(capsys):
    cx.pp.assign_guides(UMIS, GUIDES)
    out = capsys.readouterr().out
    assert "3/7 cells assigned" in out and "4 unassigned" in out
