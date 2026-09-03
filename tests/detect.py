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
BOXCAR = 41


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
    """Locate the narrowest strong feature the Gaussian fit did not explain.

    High-passing the residual (subtracting a boxcar of itself) suppresses the
    broad model mismatch that swamps a plain-residual test. On postwinterbreak
    that mismatch has MAD ~250 counts against a ~600 count dot, so a plain
    residual test scores the real dot at 0.6 sigma; after high-passing it scores
    8-17 sigma and its argmax lands within 1px of the true 2D dot position.

    The first and last `boxcar` samples are excluded from the search. A boxcar
    filter has no valid output there -- whatever edge handling it uses is an
    extrapolation -- and in practice that artifact produced spurious peaks at
    index 30-36 on several runs, which is what this margin removes.

    Returns (index, significance).
    """
    resid = residual(profile, popt)
    hp = resid - uniform_filter1d(resid, boxcar, mode="nearest")
    noise = mad(hp)
    margin = min(boxcar, max(1, profile.size // 4))
    interior = hp[margin:profile.size - margin]
    if interior.size == 0:
        return int(np.argmax(hp)), (float(hp.max() / noise) if noise > 0 else np.inf)
    i = int(np.argmax(interior)) + margin
    return i, (float(hp[i] / noise) if noise > 0 else np.inf)


def second_component(px, py, popt_x, popt_y, min_sigma=5.0,
                     min_sep_sigmas=2.0, min_sep_frac=0.05):
    """Detect a fit that has locked onto the wrong one of two sources.

    Separation, not significance, is the discriminator. Measured on real frames:

        allmetal         4.3 sigma but   1.3 px separation -> single dot
        frosty           5.5 sigma and 508 px separation   -> two dots
        postwinterbreak 18.8 sigma and 375 px separation   -> two dots

    Two separate floors are needed, and neither alone works:

    * `min_sep_sigmas` alone fails on narrow dots. A sigma~5 dot whose core is
      slightly non-Gaussian leaves residual structure a few px off centre; at
      1 sigma that reads as a companion, which produced dozens of false
      positives at separations of 6-30 px.
    * `min_sep_sigmas` alone ALSO fails on broad blobs. postwinterbreak's real
      dot sits only 2.4 sigma from the blob it is being confused with, so any
      threshold loose enough to reject the narrow-dot artifacts above would
      reject the one case we have confirmed in 2D.

    Requiring the separation to clear both a multiple of the fitted sigma and a
    fraction of the frame keeps postwinterbreak (375 px, 2.4 sigma, frame 2560)
    while dropping the narrow-dot artifacts (8 px, 1.4 sigma, frame 1280).

    Returns None, or (ix, iy, separation_px, significance).
    """
    ix, sx = compact_peak(px, popt_x)
    iy, sy = compact_peak(py, popt_y)
    sig = max(sx, sy)
    if sig < min_sigma:
        return None
    mu_x, mu_y = popt_x[1], popt_y[1]
    if not (np.isfinite(mu_x) and np.isfinite(mu_y)):
        return None
    scale = max(popt_x[2], popt_y[2])
    if not np.isfinite(scale) or scale <= 0:
        return None
    sep = float(np.hypot(mu_x - ix, mu_y - iy))
    floor = max(min_sep_sigmas * scale, min_sep_frac * max(px.size, py.size))
    return (ix, iy, sep, sig) if sep > floor else None


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
