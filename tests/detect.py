"""Detectors used by the test suite.

These are *test-side* helpers, not pipeline code. They exist so a test can ask
"is the fit sitting on the wrong thing?" without the pipeline having to record
it. If any of them earns its way into production it should move into
fits_reprocess.py and be recorded in {run}_frames.csv -- see TEST_PLAN.md.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

import fits_reprocess as fr

# Wide compared with a ~5px dot, narrow compared with a ~390px laser blob.
# Calibrated on 20260105/postwinterbreak.
BOXCAR = fr.BOXCAR_PX


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def mad(values):
    """Median absolute deviation, scaled to be comparable to a standard
    deviation. Used instead of RMS so a strong outlier -- which is exactly what
    we are looking for -- cannot inflate the noise floor it is measured
    against."""
    v = np.asarray(values, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def residual(profile, popt):
    x = np.arange(profile.size, dtype=np.float64)
    return profile - fr.gaussian(x, *popt)


def on_bound(profile, popt, tol_mu=1e-3, tol_sigma=1e-6):
    """Which fitted parameters are sitting on the bounds _fit_one_profile uses.

    Bounds are ([0, 0, 1, -inf], [inf, size, size/2, inf]), so everything here
    derives from profile.size and nothing assumes a resolution. Returns a list
    of human-readable strings; empty means the fit is interior.
    """
    _amp, mu, sigma, _offset = popt
    hits = []
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return hits                      # a declined fit is not a railed fit
    if mu < tol_mu:
        hits.append(f"mu={mu:.6f} on lower bound 0")
    if mu > profile.size - tol_mu:
        hits.append(f"mu={mu:.6f} on upper bound {profile.size}")
    if sigma > profile.size / 2.0 - tol_sigma:
        hits.append(f"sigma={sigma:.6f} on upper bound {profile.size / 2.0:.1f}")
    if sigma < 1.0 + tol_sigma:
        hits.append(f"sigma={sigma:.6f} on lower bound 1")
    return hits


def compact_peak(profile, popt, boxcar=BOXCAR):
    """Delegates to the promoted implementation in fits_reprocess.

    This used to be a separate copy. Now that the pipeline records
    two_component, a second copy here could drift from the code that actually
    produced the CSV, and the tests would stop describing production.
    """
    return fr._compact_peak(profile, popt, boxcar=boxcar)


def second_component(px, py, popt_x, popt_y, min_sigma=5.0,
                     min_sep_sigmas=2.0, min_sep_frac=0.05):
    """Delegates to the promoted implementation in fits_reprocess."""
    return fr.second_component(px, py, popt_x, popt_y, min_sigma=min_sigma,
                               min_sep_sigmas=min_sep_sigmas,
                               min_sep_frac=min_sep_frac)


def classify(profile, popt):
    """Label why a fit is unsatisfactory.

    Deliberately conservative: anything not matching a known signature comes
    back 'unclassified' so it fails a test loudly rather than being absorbed
    into a category it does not belong to.

    'now_fits' is the interesting one. The fixture's failing frames were chosen
    by the fit_ok flag recorded in each run's CSV, which was written by the old
    code. A frame that now converges to an interior fit with real contrast is
    not a failure at all -- the stored flag is stale, and the frame will pass
    once the run is reprocessed.
    """
    _amp, mu, sigma, offset = popt
    if not np.isfinite(mu) or not np.isfinite(sigma):
        return "no_convergence"

    # Saturation: a flat top at the sensor ceiling gives the summed profile a
    # plateau rather than a peak.
    top = float(profile.max())
    if np.count_nonzero(profile >= top * (1 - 1e-9)) > 1:
        return "saturated"

    bounds = on_bound(profile, popt)
    for b in bounds:
        if b.startswith("sigma") and "upper" in b:
            return "railed_sigma"
        if b.startswith("mu"):
            return "railed_mu"
    if bounds:
        return "railed_sigma"

    contrast = float(top - np.median(profile))
    noise = mad(residual(profile, popt))
    if noise > 0 and contrast < 5.0 * noise:
        return "dot_off_frame"

    # Interior fit, real contrast, nothing on a bound.
    if 1.0 < sigma < profile.size / 4.0:
        return "now_fits"
    return "unclassified"
