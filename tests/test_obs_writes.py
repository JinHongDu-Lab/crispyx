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


def test_a_swap_killed_between_its_moves_is_finished_by_any_crispyx_read(tmp_path):
    """read_backed (behind cx.AnnData) finishes the swap, after which anndata
    reads the file too; a read-only file is read without being repaired."""
    import os
    import stat

    from crispyx.data import read_backed

    def interrupted():
        path = _obs_file(tmp_path)
        with h5py.File(path, "r+") as f:
            write_elem(f, "obs.__new__", load_obs(path).assign(h=1))
            f.move("obs", "obs.__old__")
        return path

    path = interrupted()
    backed = read_backed(path)
    assert "h" in backed.obs
    backed.file.close()
    with h5py.File(path, "r") as f:
        assert "obs" in f and not [k for k in f if "__" in k]
    assert "h" in ad.read_h5ad(path).obs

    path = interrupted()
    os.chmod(path, stat.S_IRUSR)
    try:
        backed = read_backed(path)
        assert "h" in backed.obs
        backed.file.close()
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_a_crispyx_handle_open_on_the_file_is_closed_and_reopens(tmp_path, monkeypatch):
    """HDF5 refuses to open for writing a file this process reads: crispyx's
    own wrappers are closed first, whatever spelling of the path they hold."""
    import crispyx as cx

    path = _obs_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    by_absolute_path, by_relative_path = cx.AnnData(path), cx.AnnData("obs.h5ad")
    assert by_absolute_path.shape == by_relative_path.shape == (40, 2)  # both hold the file open

    write_obs(path, load_obs(path).assign(h=1))
    cx.write_var("obs.h5ad", cx.load_var(path).assign(v=2))
    assert by_absolute_path.obs.load()["h"].eq(1).all()  # reopened, and sees the new table
    assert by_relative_path.var.load()["v"].eq(2).all()
    by_absolute_path.close()
    by_relative_path.close()

    # a wrapper opened by a relative path is still found after the working directory changes
    held = cx.AnnData("obs.h5ad")
    assert held.shape == (40, 2)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    write_obs(path, load_obs(path).assign(h=3))
    assert load_obs(path)["h"].eq(3).all()


def test_a_wrapper_that_showed_a_table_reads_it_again_after_a_rewrite(tmp_path):
    """obs and var are cached without opening the file, so a wrapper with no
    open handle is told of a rewrite as well: through write_obs / write_var,
    and through the functions that update a file in place."""
    import crispyx as cx

    path = _obs_file(tmp_path)
    shown, passed_in = cx.AnnData(path), cx.AnnData(path)
    assert list(shown.obs.load().columns) == list(passed_in.obs.head().columns) == ["g"]
    assert shown.var.load().shape[1] == 0
    assert shown._backed is None and passed_in._backed is None  # neither holds the file open

    write_obs(path, load_obs(path).assign(h=1))
    assert shown.obs.load()["h"].eq(1).all()
    cx.normalise_perturbation_labels(passed_in, "g", control_aliases=["t0"], canonical_control="control")
    assert set(passed_in.obs.load()["g"]) == set(shown.obs.load()["g"]) == {"control", "t1", "t2", "t3"}
    cx.write_var(path, cx.load_var(path).assign(v=2))
    assert shown.var.load()["v"].eq(2).all()


def test_a_handle_crispyx_does_not_own_gives_a_clear_error(tmp_path):
    path = _obs_file(tmp_path)
    with h5py.File(path, "r"):
        with pytest.raises(OSError, match="open elsewhere in this process.*close that handle"):
            write_obs(path, load_obs(path).assign(h=1))
    assert "h" not in load_obs(path)
