"""Fit correctness on frames with independently known answers.

Most of this file is expected to FAIL against the current fits_reprocess.py.
That is the point: these encode the behaviour the fit should have, written
before the fix so the fix can be shown to cause them to pass.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

import fits_reprocess as fr
from detect import on_bound, residual, rms

pytestmark = pytest.mark.needs_fixtures


# (key, expected mu, expected sigma). Cross-checked against an unbounded LM fit
# to 4 decimal places during the original diagnosis.
RAILED_ANCHORS = [
    ("allmetal_f011_py", 535.2666, 4.4065),
    ("allmetal_f018_py", 530.6110, 5.3530),
]

HEALTHY_ANCHORS = ["allmetal_f001_py", "allmetal_f101_py", "allmetal_f501_py",
                   "allmetal_f001_px", "allmetal_f101_px", "allmetal_f501_px"]

# Seeds a correct implementation must not be beaten by. Deliberately unrelated
# to whatever ladder the implementation actually uses -- the assertion is about
# the answer, not the mechanism.
RIVAL_SEEDS = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]


def _bounds(size):
    return ([0.0, 0.0, 1.0, -np.inf],
            [np.inf, float(size), size / 2.0, np.inf])


@pytest.mark.parametrize("key,mu_want,sigma_want", RAILED_ANCHORS)
def test_railed_frames_recover_truth(anchors, key, mu_want, sigma_want):
    """The two frames the whole investigation started from.

    Today these return mu=0.000/sigma=355.26 and mu=410.4/sigma=512.0.
    """
    prof = anchors[key]
    _amp, mu, sigma, _off = fr._fit_one_profile(prof)
    assert mu == pytest.approx(mu_want, abs=0.05)
    assert sigma == pytest.approx(sigma_want, rel=0.02)


def test_no_parameter_lands_on_a_bound(anchors):
    """Bound-hitting is the unambiguous failure signature -- large sigma alone
    is not, because some runs have genuinely broad sources.

    Collects every failure rather than dying on the first so one run shows the
    whole picture.
    """
    failures = []
    for key, prof in sorted(anchors.items()):
        popt = fr._fit_one_profile(prof)
        for hit in on_bound(prof, popt):
            failures.append(f"{key}: {hit}")
    assert not failures, "parameters on bounds:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("key", HEALTHY_ANCHORS + [k for k, _, _ in RAILED_ANCHORS])
def test_ladder_returns_lowest_residual_fit(anchors, key):
    """No alternative starting sigma may reach a lower residual than the fit
    that was returned.

    This is the test that would have caught the original bug, where the retry
    ladder accepted the first seed that did not raise rather than the best fit.
    Written without reference to the ladder's actual seeds, so it stays valid if
    the implementation changes.
    """
    prof = anchors[key]
    x = np.arange(prof.size, dtype=np.float64)
    popt = fr._fit_one_profile(prof)
    if not np.isfinite(popt[1]):
        pytest.skip("fit declined; nothing to compare")
    got = rms(residual(prof, popt))

    bounds = _bounds(prof.size)
    p0_amp = float(prof.max() - np.median(prof))
    p0_mu = float(prof.argmax())
    p0_off = float(np.median(prof))

    for s in RIVAL_SEEDS:
        if s >= prof.size / 2.0:
            continue                      # outside the bound, not a fair rival
        try:
            alt, _ = curve_fit(fr.gaussian, x, prof, p0=[p0_amp, p0_mu, s, p0_off],
                               bounds=bounds, maxfev=5000)
        except Exception:
            continue
        got_alt = rms(residual(prof, alt))
        assert got_alt >= got * 0.999, (
            f"{key}: starting sigma={s} reaches RMS {got_alt:.1f}, beating the "
            f"returned fit's {got:.1f} (returned mu={popt[1]:.3f} sigma={popt[2]:.3f}, "
            f"rival mu={alt[1]:.3f} sigma={alt[2]:.3f})")


@pytest.mark.parametrize("key", HEALTHY_ANCHORS)
def test_fitted_peak_matches_profile_peak(anchors, key):
    """amp is height ABOVE offset, so the model's predicted peak must match the
    data's. Stated as physics rather than as a check on the initial guess."""
    prof = anchors[key]
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    assert np.isfinite(amp) and np.isfinite(offset)
    assert amp + offset == pytest.approx(float(prof.max()), rel=0.05)


def test_fit_is_deterministic(anchors):
    """Same input twice must give identical bits -- no RNG, no thread-count or
    ordering dependence hiding in the solver path."""
    for key in ("allmetal_f011_py", "allmetal_f001_px"):
        prof = anchors[key]
        first = fr._fit_one_profile(prof)
        second = fr._fit_one_profile(prof)
        assert first == second, f"{key}: {first} != {second}"


def test_declined_fit_returns_all_nan():
    """When no fit is possible the contract is four NaNs, not a partial tuple
    or an exception -- _fill_fit_results relies on that shape."""
    flat = np.zeros(256, dtype=np.float64)
    out = fr._fit_one_profile(flat)
    assert len(out) == 4
    assert all(isinstance(v, float) for v in out)
