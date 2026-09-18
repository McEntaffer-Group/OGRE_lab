"""Shared fixtures and markers.

The E: data drive is treated as STRICTLY READ-ONLY throughout this suite.
Nothing here -- or in make_fixtures.py -- opens a file on E: for writing,
creates a directory there, or deletes anything. Fixtures are read from E: and
written only into tests/fixtures/ inside the repo.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

E_DRIVE = Path("E:/Reverse Telescope Test Data")
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: requires the E: data drive")
    config.addinivalue_line("markers", "needs_fixtures: requires generated fixture npz files")


def pytest_collection_modifyitems(config, items):
    no_drive = not E_DRIVE.exists()
    no_fixtures = not (FIXTURES / "index.csv").exists()
    for item in items:
        if no_drive and "needs_data" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="E: data drive not mounted"))
        if no_fixtures and "needs_fixtures" in item.keywords:
            item.add_marker(pytest.mark.skip(
                reason="fixtures not generated; run: python tests/make_fixtures.py"))


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def anchors():
    """Frames with independently known expected values."""
    return _load("anchors.npz")


@pytest.fixture(scope="session")
def healthy():
    """One fit_ok frame from every run."""
    return _load("healthy.npz")


@pytest.fixture(scope="session")
def failing():
    """One fit_ok=False frame from every run that has one."""
    return _load("failing.npz")


@pytest.fixture(scope="session")
def all_fixture_profiles():
    """Every committed 1D profile, for tests that assert on the population
    rather than on a named frame."""
    out = []
    for name in ("anchors.npz", "healthy.npz", "failing.npz"):
        for prof in _load(name).values():
            prof = np.asarray(prof, dtype=np.float64)
            if prof.ndim == 1 and prof.size >= 3 and np.all(np.isfinite(prof)):
                out.append(prof)
    return out


@pytest.fixture(scope="session")
def fit_cache():
    """Memoized _fit_one_profile keyed by fixture name.

    The survey runs several assertions over the same ~190 profiles, and a fit on
    a 2560px profile is not cheap -- especially before the fix, where the first
    ladder rung burns its full maxfev budget before raising. Caching turns four
    passes over every run into one.

    Tests that are specifically about repeated calls (determinism) must NOT use
    this -- they need two genuine invocations.
    """
    import fits_reprocess as fr

    store = {}

    def get(key, profile):
        if key not in store:
            store[key] = fr._fit_one_profile(profile)
        return store[key]

    return get
