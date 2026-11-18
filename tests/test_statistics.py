from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from crispyx._statistics import _add_group_moments


def _reference(block: sp.csr_matrix, codes: np.ndarray, n_groups: int):
    """One mask per group, as t_test summed before the helper existed."""
    n_genes = block.shape[1]
    out = (np.zeros((n_groups, n_genes)), np.zeros((n_groups, n_genes)),
           np.zeros((n_groups, n_genes), dtype=np.int32), np.zeros(n_groups, dtype=np.int64))
    for code in np.unique(codes[codes >= 0]):
        rows = block[codes == code]
        out[0][code] += rows.sum(axis=0).A1
        out[1][code] += rows.power(2).sum(axis=0).A1
        out[2][code] += rows.getnnz(axis=0)
        out[3][code] += rows.shape[0]
    return out


@pytest.mark.parametrize("layout", ["runs", "scattered", "runs-with-untested", "scattered-with-untested", "untested"])
def test_group_moments_match_one_mask_per_group(layout):
    rng = np.random.default_rng(0)
    n_rows, n_genes, n_groups = 300, 25, 12
    block = sp.random(n_rows, n_genes, density=0.3, format="csr", random_state=1, dtype=np.float32)
    block.data = np.round(block.data * 5, 2)
    codes = rng.integers(0, n_groups, n_rows)
    if layout.startswith("runs"):
        codes = np.sort(codes)
    if layout.endswith("untested"):
        codes[rng.random(n_rows) < 0.3] = -1  # cells of groups this run does not test
    if layout == "untested":
        codes[:] = -1
    got = (np.zeros((n_groups, n_genes)), np.zeros((n_groups, n_genes)),
           np.zeros((n_groups, n_genes), dtype=np.int32), np.zeros(n_groups, dtype=np.int64))
    for start in range(0, n_rows, 70):  # accumulates over chunks
        _add_group_moments(block[start:start + 70], codes[start:start + 70], *got)
    for actual, expected in zip(got, _reference(block, codes, n_groups)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(got[2], _reference(block, codes, n_groups)[2])
    np.testing.assert_array_equal(got[3], _reference(block, codes, n_groups)[3])
