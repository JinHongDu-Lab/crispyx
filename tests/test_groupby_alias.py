"""Scanpy-style ``groupby`` / ``reference`` aliases on the DE entry points.

The aliases are resolved by ``_grouping.resolve_group_reference_aliases``
(``cx.tl.rank_genes_groups`` repeats the same checks inline); the tests here
check the resolver directly and that every entry point is wired to it.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import crispyx as cx
from crispyx._grouping import resolve_group_reference_aliases

_COLUMN_CONFLICT = "perturbation_column.*groupby|groupby.*perturbation_column"
_LABEL_CONFLICT = "control_label.*reference|reference.*control_label"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(perturbation_column="p", control_label="c"), ("p", "c")),
        (dict(groupby="p", control_label="c"), ("p", "c")),
        (dict(perturbation_column="p", reference="c"), ("p", "c")),
        (dict(groupby="p", reference="c"), ("p", "c")),
        (dict(groupby="p"), ("p", None)),
    ],
    ids=["canonical", "groupby", "reference", "both-aliases", "no-control"],
)
def test_resolver_maps_aliases(kwargs, expected):
    args = dict(perturbation_column=None, groupby=None, control_label=None, reference=None)
    assert resolve_group_reference_aliases(**{**args, **kwargs}, fn_name="f") == expected


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(perturbation_column="p", groupby="p"), _COLUMN_CONFLICT),
        (dict(perturbation_column="p", control_label="c", reference="c"), _LABEL_CONFLICT),
        (dict(control_label="c"), _COLUMN_CONFLICT),
    ],
    ids=["column-conflict", "label-conflict", "missing-column"],
)
def test_resolver_rejects_conflicts(kwargs, match):
    args = dict(perturbation_column=None, groupby=None, control_label=None, reference=None)
    with pytest.raises(TypeError, match=match):
        resolve_group_reference_aliases(**{**args, **kwargs}, fn_name="f")


@pytest.fixture(scope="module")
def screen(tmp_path_factory) -> tuple[Path, Path]:
    """(log-normalised, raw-count) copies of one small screen."""
    root = tmp_path_factory.mktemp("alias")
    rng = np.random.default_rng(0)
    n_ctrl, n_pert, n_genes = 40, 30, 20
    counts = rng.poisson(5, (n_ctrl + n_pert, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"c{i}" for i in range(n_ctrl + n_pert)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    log = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    ad.AnnData(sp.csr_matrix(log), obs=obs, var=var).write(root / "log.h5ad")
    ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var).write(root / "counts.h5ad")
    return root / "log.h5ad", root / "counts.h5ad"


def _rank_genes_groups(path, out, **kw):
    result = cx.tl.rank_genes_groups(
        path, method="wilcoxon", output_dir=out.parent, data_name=out.stem, verbose=False, **kw
    )
    effect = np.asarray(result.to_memory().X)
    result.close()
    return effect


_ENTRY_POINTS = {
    "t_test": (lambda p, out, **kw: cx.t_test(p, output_path=out, verbose=False, **kw).pvalues, 0),
    "wilcoxon_test": (lambda p, out, **kw: cx.wilcoxon_test(p, output_path=out, verbose=False, **kw).pvalues, 0),
    "nb_glm_test": (lambda p, out, **kw: cx.nb_glm_test(p, output_path=out, verbose=False, n_jobs=1, **kw).pvalues, 1),
    "rank_genes_groups": (_rank_genes_groups, 0),
}


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
def test_aliases_give_the_canonical_result(screen, tmp_path, entry):
    run, source = _ENTRY_POINTS[entry]
    path = screen[source]
    canonical = run(path, tmp_path / "canonical.h5ad", perturbation_column="perturbation", control_label="ctrl")
    aliased = run(path, tmp_path / "aliased.h5ad", groupby="perturbation", reference="ctrl")
    np.testing.assert_array_equal(aliased, canonical)


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(perturbation_column="perturbation", groupby="perturbation"), _COLUMN_CONFLICT),
        (dict(perturbation_column="perturbation", control_label="ctrl", reference="ctrl"), _LABEL_CONFLICT),
        (dict(control_label="ctrl"), _COLUMN_CONFLICT),
    ],
    ids=["column-conflict", "label-conflict", "missing-column"],
)
def test_entry_points_reject_conflicting_aliases(screen, tmp_path, entry, kwargs, match):
    run, source = _ENTRY_POINTS[entry]
    with pytest.raises(TypeError, match=match):
        run(screen[source], tmp_path / "never.h5ad", **kwargs)
