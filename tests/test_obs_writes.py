"""write_obs / write_var: anndata's encoding, swapped in so that a kill at
any point leaves a readable table that the next call finishes."""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from anndata.io import write_elem

from crispyx.data import load_obs, write_obs


def _obs_file(tmp_path, n=40):
    obs = pd.DataFrame({"g": [f"t{i % 4}" for i in range(n)]}, index=[f"c{i}" for i in range(n)])
    path = tmp_path / "obs.h5ad"
    ad.AnnData(np.ones((n, 2), dtype=np.float32), obs=obs).write(path)
    return path


def test_write_obs_round_trips_what_anndata_writes(tmp_path):
    """Nullable columns, missing text and > 32,767 categories (int16 codes
    overflowed in the hand-written encoder) come back as written."""
    n = 40_000
    path = _obs_file(tmp_path, n)
    obs = pd.DataFrame(
        {
            "guide": pd.Categorical([f"g{i}" for i in range(n)]),
            "age": pd.array([1, None] * (n // 2), dtype="Int64"),
            "flag": pd.array([True, None] * (n // 2), dtype="boolean"),
            "note": ["x", None] * (n // 2),
        },
        index=[f"c{i}" for i in range(n)],
    )
    write_obs(path, obs)
    back = load_obs(path)
    assert back["guide"].tolist() == obs["guide"].tolist()
    assert back["age"].dtype == "Int64" and back["age"].isna().sum() == n // 2
    assert back["flag"].dtype == "boolean"
    assert back["note"].isna().sum() == n // 2
    assert ad.read_h5ad(path).obs["guide"].tolist()[-1] == f"g{n - 1}"


def test_write_obs_checks_the_row_count(tmp_path):
    path = _obs_file(tmp_path)
    with pytest.raises(ValueError, match="has 40 cells"):
        write_obs(path, pd.DataFrame(index=["a"]))


@pytest.mark.parametrize("killed_after", ["write", "first_move", "second_move"])
def test_an_interrupted_obs_swap_is_finished_by_the_next_call(tmp_path, killed_after):
    """Whichever step of the swap a kill interrupts, the file still has a
    readable table and the next write completes cleanly."""
    path = _obs_file(tmp_path)
    new = load_obs(path).assign(h=1)
    with h5py.File(path, "r+") as f:
        write_elem(f, "obs.__new__", new)
        if killed_after != "write":
            f.move("obs", "obs.__old__")
        if killed_after == "second_move":
            f.move("obs.__new__", "obs")
    assert ("h" in load_obs(path)) == (killed_after != "write")
    write_obs(path, new.assign(h=2))
    with h5py.File(path, "r") as f:
        assert not [k for k in f if "__" in k]
    assert load_obs(path)["h"].eq(2).all()
