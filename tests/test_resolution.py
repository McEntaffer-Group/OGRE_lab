"""Resolution independence.

Nothing in the fit may assume a particular image size. The bug being fixed is
itself size-scaled -- _estimate_sigma clips at profile.size/4 and the sigma
bound is profile.size/2 -- so a 64px frame and a 2592px frame fail differently
and both need covering.

These use synthetic profiles only: no fixtures, no E:, always runnable.
"""

import numpy as np
import pytest

import fits_reprocess as fr
from detect import on_bound
from synth import one_hot, synth, two_peaks

# Spans the real hardware (allmetal 1280x1024, genie 1936x1216,
# postwinterbreak 2560x1920) plus small sizes no current run uses.
SIZES = [64, 128, 256, 512, 1024, 2592]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sigma", [2.0, 5.0, 20.0])
def test_estimator_does_not_saturate_at_any_resolution(size, sigma):
    """The estimator must measure the spot, not return its own clip ceiling.

    The band is deliberately loose (a factor of 3). The estimator's job is to
    seed the fit, not to be the answer -- a documented ~2x overshoot is fine
    because the residual-scored ladder brackets it. Asserting tightly here would
    push toward the over-engineered estimator that was rejected.
    """
    if sigma * 8 > size:
        pytest.skip(f"sigma={sigma} spot does not fit in a {size}px frame")
    prof, _mu = synth(size, sigma=sigma)
    est = fr._estimate_sigma(prof, np.arange(size, dtype=np.float64),
                             float(prof.argmax()))
    ceiling = size / 4.0
    assert not np.isclose(est, ceiling), (
        f"estimator pinned at its ceiling {ceiling:.1f} "
        f"(size={size}, true sigma={sigma})")
    assert sigma / 3.0 < est < sigma * 3.0, (
        f"est={est:.2f} not within 3x of true sigma={sigma} (size={size})")


@pytest.mark.parametrize("size", SIZES)
def test_fit_recovers_known_spot_at_any_resolution(size):
    true_sigma = max(2.0, size / 200.0)
    prof, true_mu = synth(size, sigma=true_sigma)
    _amp, mu, sigma, _off = fr._fit_one_profile(prof)
    assert mu == pytest.approx(true_mu, abs=0.5), f"size={size}"
    assert sigma == pytest.approx(true_sigma, rel=0.15), f"size={size}"


@pytest.mark.parametrize("size", [512, 1024, 2592])
def test_very_broad_spot_is_fit_not_railed(size):
    """A dot may legitimately be much wider than FWHM_MAX_PX.

    At size=2592 this spot has FWHM ~1221 px, past the 1000 px gate. The
    assertion is about the FIT, deliberately saying nothing about fit_ok: the
    gate is a downstream policy question, and a correct fit must not depend on
    it.
    """
    true_sigma = size / 5.0
    prof, true_mu = synth(size, sigma=true_sigma, amp=20000.0, offset=34000.0)
    popt = fr._fit_one_profile(prof)
    _amp, mu, sigma, _off = popt
    assert mu == pytest.approx(true_mu, rel=0.02), f"size={size}"
    assert sigma == pytest.approx(true_sigma, rel=0.15), f"size={size}"
    assert not on_bound(prof, popt), f"size={size}: {on_bound(prof, popt)}"


@pytest.mark.parametrize("size", [64, 512, 2592])
@pytest.mark.parametrize("name", ["flat", "zeros", "hot_pixel", "edge_peak", "two_peaks"])
def test_estimator_survives_degenerate_input(size, name):
    """Inputs with no right answer must not raise or return a value that would
    poison the fit. Assertions are deliberately weak."""
    cases = {
        "flat": np.full(size, 21000.0),
        "zeros": np.zeros(size),
        "hot_pixel": one_hot(size),
        "edge_peak": synth(size, mu_frac=0.002, sigma=4.0, noise=0.0)[0],
        "two_peaks": two_peaks(size),
    }
    prof = cases[name]
    est = fr._estimate_sigma(prof, np.arange(size, dtype=np.float64),
                             float(prof.argmax()))
    assert np.isfinite(est), f"{name}@{size}: est={est}"
    assert 0 < est <= size, f"{name}@{size}: est={est}"


@pytest.mark.parametrize("size", [64, 512, 2592])
@pytest.mark.parametrize("name", ["flat", "zeros", "hot_pixel", "edge_peak"])
def test_fit_survives_degenerate_input(size, name):
    """Same inputs through the whole fit. Either a finite answer or four NaNs;
    never an exception, never a partial result."""
    cases = {
        "flat": np.full(size, 21000.0),
        "zeros": np.zeros(size),
        "hot_pixel": one_hot(size),
        "edge_peak": synth(size, mu_frac=0.002, sigma=4.0, noise=0.0)[0],
    }
    out = fr._fit_one_profile(cases[name])
    assert len(out) == 4
    finite = [np.isfinite(v) for v in out]
    assert all(finite) or not any(finite), f"{name}@{size}: partial result {out}"
