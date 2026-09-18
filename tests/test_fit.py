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


# ---------------------------------------------------------------------------
# Evidence columns: nx/ny, resid, on_bound
# ---------------------------------------------------------------------------

def test_wrapper_agrees_with_full_fit(anchors):
    """_fit_one_profile must stay a pure view of _fit_profile. If they drift,
    the CSV and the tests stop describing the same fit."""
    for key in ("allmetal_f011_py", "allmetal_f001_px"):
        prof = anchors[key]
        full = fr._fit_profile(prof)
        assert fr._fit_one_profile(prof) == (
            full["amp"], full["mu"], full["sigma"], full["offset"])


def test_on_bound_ignores_broad_but_real_spots():
    """The discrimination the whole detector rests on: a genuinely wide spot is
    not a failure. Several runs fit sigma ~160 with the centroid nowhere near
    an edge, and flagging those would be a false positive."""
    x = np.arange(1024.0)
    broad = fr.gaussian(x, 900.0, 512.0, 160.0, 21000.0)
    r = fr._fit_profile(broad)
    assert r["sigma"] == pytest.approx(160.0, rel=1e-3)
    assert r["on_bound"] is False


def test_on_bound_catches_each_bound():
    """All four ways a fit can come to rest on its box."""
    size = 512
    assert fr.on_bound(size, mu=0.0, sigma=5.0) is True             # mu low
    assert fr.on_bound(size, mu=float(size), sigma=5.0) is True     # mu high
    assert fr.on_bound(size, mu=250.0, sigma=1.0) is True           # sigma low
    assert fr.on_bound(size, mu=250.0, sigma=size / 2.0) is True    # sigma high
    assert fr.on_bound(size, mu=250.0, sigma=5.0) is False          # interior


def test_on_bound_is_false_for_a_declined_fit():
    """A fit that never converged is not a railed fit; conflating them would
    put NaN rows in the railing census."""
    assert fr.on_bound(512, mu=np.nan, sigma=np.nan) is False


def test_residual_is_recorded_and_matches_a_recomputation(anchors):
    """resid must be the RMS residual of the fit actually returned -- that is
    the number that picks between seeds, and it was previously discarded."""
    for key in ("allmetal_f011_py", "allmetal_f001_px"):
        prof = anchors[key]
        r = fr._fit_profile(prof)
        expect = rms(residual(prof, (r["amp"], r["mu"], r["sigma"], r["offset"])))
        assert r["resid"] == pytest.approx(expect, rel=1e-9)


def test_declined_fit_reports_nan_residual_not_zero():
    """A fit that never ran reports NaN, not 0.0 -- zero would read as a
    perfect fit in any downstream sort by residual."""
    for prof in (np.full(256, np.nan), np.zeros(2, dtype=np.float64)):
        r = fr._fit_profile(prof)
        assert np.isnan(r["resid"])
        assert r["on_bound"] is False
        assert r["n_seeds"] == 0


@pytest.mark.parametrize("prof", [
    pytest.param(np.zeros(256), id="all_zeros"),
    pytest.param(np.full(256, 21000.0), id="flat_pedestal"),
])
def test_blank_frame_is_fit_ok_but_flagged_on_bound(prof):
    """A featureless frame does NOT decline. curve_fit converges on the
    degenerate amp=0 solution, which leaves mu on its lower bound and sigma on
    its lower bound of 1 -- and FWHM 2.355 sits inside the fit_ok gate, so
    fit_ok comes out True on a frame with no dot in it at all.

    This is the clearest case for recording on_bound: fit_ok cannot see it,
    and mu=1e-10 would otherwise enter the position series as a real centroid
    at pixel 0. Suspected to be the signature behind the runs that rail on the
    LOWER sigma bound (minutelyovernight, longweekend).
    """
    out = fr._fill_fit_results(
        fr._empty_row("blank.fits", 1, "2026-01-01 00:00:00"),
        np.asarray(prof, dtype=np.float64), np.asarray(prof, dtype=np.float64))
    assert out["fit_ok"] is True, "gate no longer lets a blank frame through"
    assert out["on_bound_x"] is True and out["on_bound_y"] is True
    assert out["sigma_x"] == pytest.approx(1.0)


def test_nx_ny_record_the_profile_lengths():
    """Every bound derives from these, so a railed fit is only detectable from
    the CSV if the frame size is in the CSV."""
    x = np.arange(1024.0)
    px = fr.gaussian(x, 900.0, 512.0, 5.0, 21000.0)
    py = fr.gaussian(np.arange(600.0), 900.0, 300.0, 5.0, 21000.0)
    out = fr._fill_fit_results(
        fr._empty_row("f.fits", 1, "2026-01-01 00:00:00"), px, py)
    assert out["nx"] == 1024
    assert out["ny"] == 600
    assert out["on_bound_x"] is False and out["on_bound_y"] is False
    assert np.isfinite(out["resid_x"]) and np.isfinite(out["resid_y"])


def test_fast_path_is_taken_on_almost_every_real_profile(all_fixture_profiles):
    """The ladder is a fallback now, not the normal path. If a change makes the
    alternates fire routinely, the estimator regressed -- that is the signal
    this test exists to catch, not the exact count."""
    counts = [fr._fit_profile(p)["n_seeds"] for p in all_fixture_profiles]
    fast = sum(1 for c in counts if c == 1)
    assert fast / len(counts) > 0.90, (
        f"only {fast}/{len(counts)} profiles took the single-fit fast path")


def test_fallback_fires_exactly_when_the_fast_path_rails(all_fixture_profiles):
    """The fallback's trigger condition must be the recorded one, so that
    on_bound in the CSV explains the cost after a reprocess."""
    for p in all_fixture_profiles:
        r = fr._fit_profile(p)
        if r["n_seeds"] == 1:
            assert r["on_bound"] is False


# ---------------------------------------------------------------------------
# noise / two_component
# ---------------------------------------------------------------------------

def test_resid_over_noise_is_about_one_for_a_clean_single_dot():
    """The calibration the whole column rests on: when the model is right the
    residual IS the noise, so the ratio sits near 1 regardless of brightness."""
    x = np.arange(1024.0)
    rng = np.random.default_rng(0)
    for amp, off in ((926.0, 21000.0), (150000.0, 26000.0)):   # 160x brightness
        prof = fr.gaussian(x, amp, 512.0, 4.3, off) + rng.normal(0, 60.0, 1024)
        r = fr._fit_profile(prof)
        noise = fr.wing_noise(prof, r["amp"], r["mu"], r["sigma"], r["offset"])
        assert 0.5 < r["resid"] / noise < 2.0, (
            f"amp={amp}: ratio {r['resid'] / noise:.2f} outside the noise-limited band")


def test_resid_over_noise_is_scale_free_where_resid_alone_is_not():
    """resid alone cannot be compared between runs -- a 160x brighter source
    has a far larger residual while fitting just as well. That is exactly the
    trap springgenie fell into, so assert the normalisation actually removes it."""
    x = np.arange(1024.0)
    rng = np.random.default_rng(1)
    faint = fr.gaussian(x, 926.0, 512.0, 4.3, 21000.0) + rng.normal(0, 60.0, 1024)
    bright = fr.gaussian(x, 148160.0, 512.0, 4.3, 21000.0) + rng.normal(0, 9600.0, 1024)
    rf, rb = fr._fit_profile(faint), fr._fit_profile(bright)
    nf = fr.wing_noise(faint, rf["amp"], rf["mu"], rf["sigma"], rf["offset"])
    nb = fr.wing_noise(bright, rb["amp"], rb["mu"], rb["sigma"], rb["offset"])
    assert rb["resid"] > 20 * rf["resid"], "test setup: brightness gap too small"
    assert abs(rb["resid"] / nb - rf["resid"] / nf) < 1.0, (
        "normalised ratios should agree even though raw residuals differ 20x")


def test_wing_noise_is_nan_when_the_source_fills_the_frame():
    """No background left to measure. NaN is the diagnosis, not a gap -- it is
    how 'the source is larger than the detector' appears in the CSV."""
    x = np.arange(256.0)
    prof = fr.gaussian(x, 900.0, 128.0, 400.0, 21000.0)
    r = fr._fit_profile(prof)
    assert np.isnan(fr.wing_noise(prof, r["amp"], r["mu"], r["sigma"], r["offset"]))


def test_broad_but_real_spot_is_not_flagged_two_component():
    """A wide single source must not read as two. This is the false positive
    that would put the 8 genie runs in the two-dot census."""
    x = np.arange(1024.0)
    rng = np.random.default_rng(2)
    prof = fr.gaussian(x, 926.0, 512.0, 160.0, 21000.0) + rng.normal(0, 60.0, 1024)
    r = fr._fit_profile(prof)
    popt = (r["amp"], r["mu"], r["sigma"], r["offset"])
    assert fr.second_component(prof, prof, popt, popt) is None


def test_two_well_separated_dots_are_flagged_with_their_separation():
    x = np.arange(1024.0)
    rng = np.random.default_rng(3)
    px = (fr.gaussian(x, 926.0, 312.0, 4.3, 21000.0)
          + fr.gaussian(x, 700.0, 712.0, 4.3, 0.0) + rng.normal(0, 60.0, 1024))
    out = fr._fill_fit_results(fr._empty_row("f.fits", 1, "t"), px, px)
    assert out["two_component"] is True
    assert np.isfinite(out["two_comp_sep"]) and out["two_comp_sep"] > 100


def test_two_component_is_blind_to_overlapping_pairs_but_resid_is_not():
    """Documents the known blind spot rather than pretending it is absent.

    second_component needs separation above a floor, so a small dot crossing a
    large one is invisible to it -- the case that also defeats n_peaks. The
    resid/noise ratio does see it, which is why both columns exist.
    """
    x = np.arange(1024.0)
    rng = np.random.default_rng(4)
    clean = fr.gaussian(x, 4000.0, 512.0, 40.0, 21000.0) + rng.normal(0, 60.0, 1024)
    overlap = (fr.gaussian(x, 4000.0, 512.0, 40.0, 21000.0)
               + fr.gaussian(x, 2000.0, 524.0, 4.0, 0.0) + rng.normal(0, 60.0, 1024))
    rc, ro = fr._fit_profile(clean), fr._fit_profile(overlap)
    poptc = (rc["amp"], rc["mu"], rc["sigma"], rc["offset"])
    popto = (ro["amp"], ro["mu"], ro["sigma"], ro["offset"])
    assert fr.second_component(overlap, overlap, popto, popto) is None, \
        "separation floor should make this invisible -- if not, retune the test"
    nc = fr.wing_noise(clean, *poptc)
    no = fr.wing_noise(overlap, *popto)
    assert ro["resid"] / no > 2.0 * (rc["resid"] / nc), \
        "resid/noise must catch what the separation test cannot"
