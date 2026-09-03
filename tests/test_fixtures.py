"""Fixture sanity. Catches a stale or corrupt .npz before it produces confusing
failures in every other file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.needs_fixtures


def test_index_present_and_consistent():
    d = pd.read_csv(FIXTURES / "index.csv")
    assert len(d) > 0
    for col in ("run_key", "is_image", "n", "n_ok", "n_bad",
                "has_healthy", "has_failing"):
        assert col in d.columns, col
    assert d.run_key.is_unique, "duplicate run_key -- collision in the index"
    assert (d.n_ok + d.n_bad <= d.n).all()


@pytest.mark.parametrize("name", ["anchors", "healthy", "failing"])
def test_profiles_are_sane(name):
    """No minimum-size assumption: profiles must simply be non-empty 1-D finite
    arrays with some contrast. Sizes range from 1024 to 2592 across the real
    hardware and nothing here should care."""
    path = FIXTURES / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{name}.npz not generated")
    with np.load(path) as z:
        assert len(z.files) > 0, f"{name}.npz is empty"
        for key in z.files:
            prof = z[key]
            assert prof.ndim == 1, f"{key}: ndim={prof.ndim}"
            assert prof.size > 0, key
            assert np.all(np.isfinite(prof)), f"{key}: non-finite samples"
            assert prof.max() > prof.min(), f"{key} is flat"


def test_every_indexed_run_has_its_healthy_profiles():
    d = pd.read_csv(FIXTURES / "index.csv")
    with np.load(FIXTURES / "healthy.npz") as z:
        keys = set(z.files)
    missing = [r.run_key for r in d[d.has_healthy].itertuples()
               if f"{r.run_key}_px" not in keys or f"{r.run_key}_py" not in keys]
    assert not missing, f"index claims healthy frames absent from npz: {missing}"


def test_both_axes_present_for_every_frame():
    for name in ("anchors", "healthy", "failing"):
        path = FIXTURES / f"{name}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            stems = {k.rsplit("_", 1)[0] for k in z.files}
            for stem in stems:
                assert f"{stem}_px" in z.files, f"{name}: {stem} missing px"
                assert f"{stem}_py" in z.files, f"{name}: {stem} missing py"
