"""Memory-driven sizing: chunk widths and ``memory_limit_gb``.

The heuristics are checked against their documented contract plus one table
of the benchmark datasets they were tuned on; the end-to-end test checks that
whatever chunking ``memory_limit_gb`` selects, the DE result is the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import crispyx as cx
from crispyx._memory import _resolve_memory_limit_bytes
from crispyx.data import calculate_optimal_gene_chunk_size, calculate_wilcoxon_chunk_size

# Benchmark dataset shapes: (name, n_groups, n_genes, n_cells)
DATASETS = [
    ("Adamson_subset",         2,  11_630,     1_716),
    ("Adamson",               91,  32_738,    65_337),
    ("Frangieh",             248,  23_712,   218_331),
    ("Tian-crispra",         100,  33_538,    21_193),
    ("Tian-crispri",         184,  33_538,    32_300),
    ("Feng-gwsf",          2_254,  36_518,   322_746),
    ("Feng-gwsnf",         4_955,  36_518,   396_458),
    ("Feng-ts",              444,  36_518, 1_161_864),
    ("Huang-HCT116",      18_293,  36_601, 3_409_169),
    ("Huang-HEK293T",     18_311,  38_606, 4_534_299),
]
_IDS = [d[0] for d in DATASETS]


@pytest.mark.parametrize("limit_gb", [0.5, 128])
def test_explicit_memory_limit_is_used_as_given(limit_gb):
    assert _resolve_memory_limit_bytes(limit_gb) == limit_gb * 1e9


def test_detected_memory_limit_is_positive():
    assert _resolve_memory_limit_bytes(None) > 0


@pytest.mark.parametrize("name, n_groups, n_genes, n_cells", DATASETS, ids=_IDS)
def test_gene_chunk_size_is_bounded_and_capped_by_cell_count(name, n_groups, n_genes, n_cells):
    """Hard caps by cell count apply below 32 GB; at 128 GB the memory formula
    decides, and never gives a smaller chunk than the capped one."""
    low = calculate_optimal_gene_chunk_size(n_cells, n_genes, n_groups=n_groups, available_memory_gb=16)
    high = calculate_optimal_gene_chunk_size(n_cells, n_genes, n_groups=n_groups, available_memory_gb=128)
    cap = 32 if n_cells > 1_000_000 else 64 if n_cells > 500_000 else 128 if n_cells > 300_000 else 512
    assert 32 <= low <= cap
    assert 32 <= high <= 512
    if n_groups > 2000:
        assert high <= 384
    if n_cells > 300_000:
        assert high >= low


@pytest.mark.parametrize(
    "n_obs, n_vars, n_groups, expected",
    [(10_000, 5_000, 50, 512), (1_161_864, 33_165, 444, range(32, 512))],
    ids=["small-hits-max", "Feng-ts-budget-capped"],
)
def test_gene_chunk_size_per_chunk_budget(n_obs, n_vars, n_groups, expected):
    chunk = calculate_optimal_gene_chunk_size(n_obs, n_vars, n_groups=n_groups, available_memory_gb=128.0)
    assert chunk == expected if isinstance(expected, int) else chunk in expected


@pytest.mark.parametrize(
    "n_obs, n_vars, memory_gb, low, high",
    [
        (21_071, 22_040, 128, 4096, 4096),       # small: hits max_chunk
        (393_465, 32_373, 128, 3001, 4096),      # Feng-gwsnf
        (1_161_864, 33_165, 128, 1001, 4096),    # Feng-ts
        (1_970_000, 8_248, 128, 701, 4096),      # Replogle-GW
        (393_465, 32_373, 16, 32, 999),          # Feng-gwsnf on a small node
    ],
    ids=["small", "Feng-gwsnf", "Feng-ts", "Replogle-GW", "Feng-gwsnf-16GB"],
)
def test_wilcoxon_chunk_size(n_obs, n_vars, memory_gb, low, high):
    assert low <= calculate_wilcoxon_chunk_size(n_obs, n_vars, available_memory_gb=memory_gb) <= high


@pytest.mark.parametrize("name, n_groups, n_genes, n_cells", DATASETS, ids=_IDS)
def test_wilcoxon_chunk_fits_its_budget_with_every_group(name, n_groups, n_genes, n_cells):
    """One chunk -- its cells and every group's results for its genes -- fits
    15% of the budget, however many groups there are."""
    chunk = calculate_wilcoxon_chunk_size(n_cells, n_genes, n_groups=n_groups, available_memory_gb=128)
    assert chunk == 32 or chunk * (n_cells * 12 + n_groups * 128) <= 0.15 * 128e9


@pytest.fixture(scope="module")
def log_screen(tmp_path_factory):
    rng = np.random.default_rng(42)
    labels = ["control"] * 30 + [f"pert_{i}" for i in range(3) for _ in range(10)]
    counts = rng.poisson(2, size=(len(labels), 20)).astype(np.float32)
    X = np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e4)
    obs = pd.DataFrame({"perturbation": labels}, index=[f"c{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(20)])
    path = tmp_path_factory.mktemp("memory") / "norm.h5ad"
    ad.AnnData(sp.csr_matrix(X), obs=obs, var=var).write(path)
    return path


@pytest.mark.parametrize("fn", [cx.wilcoxon_test, cx.t_test], ids=["wilcoxon", "t_test"])
def test_memory_limit_does_not_change_the_result(log_screen, tmp_path, fn):
    """A tiny limit forces the smallest chunks; the numbers must not move."""
    results = [
        fn(log_screen, perturbation_column="perturbation", control_label="control",
           output_path=tmp_path / f"{i}.h5ad", memory_limit_gb=limit, verbose=False)
        for i, limit in enumerate([128, None, 1e-7])
    ]
    for other in results[1:]:
        assert other.groups == results[0].groups
        for field in ("statistics", "pvalues", "pvalues_adj", "logfoldchanges", "auc", "mean_difference"):
            if getattr(results[0], field) is None:
                assert getattr(other, field) is None
                continue
            np.testing.assert_allclose(getattr(other, field), getattr(results[0], field), atol=1e-10, err_msg=field)
