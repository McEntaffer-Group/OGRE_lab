"""Agreement with the truth pipeline (dot_movie-Copy3.ipynb).

Two levels:
  D1  per-profile -- our fit vs truth's recipe, run live on the same samples
  D2  per-run     -- our summary line vs the notebook's, via compare_pipelines
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

import fits_reprocess as fr

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.needs_fixtures

# Anchors where a single Gaussian is the right model. The two-dot anchors
# (postwinterbreak, frosty) and the saturated one (genieshots) are excluded:
# there, truth and this pipeline can agree with each other while both sit on the
# wrong component, so agreement would not mean what the test appears to claim.
SINGLE_SOURCE = ["allmetal_f001_px", "allmetal_f001_py",
                 "allmetal_f101_px", "allmetal_f101_py",
                 "allmetal_f501_px", "allmetal_f501_py",
                 "allmetal_f011_py", "allmetal_f018_py"]


def truth_fit(prof):
    """dot_movie-Copy3.ipynb cell 16: unbounded curve_fit -- Levenberg-Marquardt
    -- with p0 sigma fixed at 5."""
    x = np.arange(prof.size, dtype=np.float64)
    p, _ = curve_fit(fr.gaussian, x, prof, maxfev=10000,
                     p0=[float(prof.max()), float(prof.argmax()), 5.0,
                         float(np.median(prof))])
    amp, mu, sigma, offset = p
    return amp, mu, abs(sigma), offset


@pytest.mark.parametrize("key", SINGLE_SOURCE)
def test_matches_truth_lm_on_golden_profiles(anchors, key):
    """Agreement with truth, computed live rather than trusting a stored number.

    Asserts agreement, not correctness -- on a single-source frame those
    coincide, which is why the two-dot anchors are excluded above.
    """
    prof = anchors[key]
    try:
        _, mu_t, sigma_t, _ = truth_fit(prof)
    except Exception as exc:
        pytest.skip(f"truth recipe did not converge on {key}: {exc}")
    _, mu_n, sigma_n, _ = fr._fit_one_profile(prof)
    assert mu_n == pytest.approx(mu_t, abs=0.01), (
        f"{key}: ours mu={mu_n:.4f}, truth mu={mu_t:.4f}")
    assert sigma_n == pytest.approx(sigma_t, rel=0.01), (
        f"{key}: ours sigma={sigma_n:.4f}, truth sigma={sigma_t:.4f}")


@pytest.mark.parametrize("key", ["postwinterbreak_f001_py", "frosty_f001_py"])
def test_two_dot_anchors_are_not_silently_compared_to_truth(anchors, key):
    """Documents why the anchors above are excluded.

    On these frames both pipelines fit the broad component, so they agree with
    each other while neither measures the small dot. This test asserts that
    situation is real rather than assumed -- if a future fix makes one of them
    find the narrow source, this fails and the exclusion needs revisiting.
    """
    prof = anchors[key]
    try:
        _, mu_t, sigma_t, _ = truth_fit(prof)
    except Exception:
        pytest.skip("truth recipe did not converge")
    _, mu_n, sigma_n, _ = fr._fit_one_profile(prof)
    assert sigma_t > 50.0 or sigma_n > 50.0, (
        f"{key}: expected at least one pipeline on the broad component, "
        f"got truth sigma={sigma_t:.2f}, ours sigma={sigma_n:.2f}")


# --------------------------------------------------------------- summary level

# Runs whose divergence from truth is NOT a bug in this pipeline: the notebook's
# fwhm_max=500 gate discards frames we legitimately keep. springgenie loses
# 12920 of 14150 frames that way.
EXPECTED_VERDICT = {
    "20260306_springgenie": "DIVERGENT_TRUTH_GATE",
    "20260320_statictestgenie": "DIVERGENT_TRUTH_GATE",
    "20260814_postspiegenie": "DIVERGENT_TRUTH_GATE",
}
ACCEPTABLE = {"MATCH", "NEAR"}


@pytest.mark.skipif(
    not (REPO / "reprocess_output" / "all_runs_summary.csv").exists(),
    reason="all_runs_summary.csv not present")
def test_summary_agreement_with_truth_does_not_regress(capsys):
    """End-to-end acceptance: our per-run summary line vs the notebook's.

    Stays red until the fit is fixed AND the affected runs are reprocessed --
    a code change alone cannot turn this green, which is the correct shape for
    an acceptance test.
    """
    import compare_pipelines as cp
    with capsys.disabled():
        df = cp.report_truth()
    bad = []
    for r in df.itertuples():
        allowed = ACCEPTABLE | {EXPECTED_VERDICT.get(r.run_key)}
        if r.verdict not in allowed:
            bad.append(f"{r.run_key}: {r.verdict} "
                       f"(worst {r.worst_reldiff:.3f} on {r.worst_column})")
    assert not bad, "runs disagreeing with truth:\n  " + "\n  ".join(bad)
