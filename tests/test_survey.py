"""The survey: every run, not just the ones we already knew about.

Parametrized by run so a failure names the dataset. The point of these tests is
not that they all pass today -- it is that when one fails, you learn which run
is unusual and why.
"""

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fits_reprocess as fr
from detect import classify, on_bound, second_component

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.needs_fixtures


def _index():
    p = FIXTURES / "index.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(
        columns=["run_key", "has_healthy", "has_failing"])


_IDX = _index()
RUN_KEYS = sorted(_IDX[_IDX.has_healthy].run_key) if len(_IDX) else []
FAIL_KEYS = sorted(_IDX[_IDX.has_failing].run_key) if len(_IDX) else []

# Runs whose oddity has been looked at and understood. Each entry carries a
# reason. Anything that fails and is NOT listed here is a dataset nobody has
# examined yet, which is the whole point of these tests.
KNOWN_UNUSUAL = {
    "20251014_minutelyovernight":
        "sigma pins at the lower bound 1.0 on both axes -- dot is near-unresolved "
        "or the frame is nearly empty; not examined in 2D yet",
    "20251029_longweekend":
        "sigma pins at the lower bound 1.0 on one axis; not examined in 2D yet",
    "20251106_laser":
        "sigma pins at the upper bound (240 = size/2 on a 480px profile); the "
        "source fills the frame, so a single Gaussian has nothing to lock onto",
}

# Runs carrying a second distinct source, where the fit measures the wrong one.
#
# VERIFIED IN 2D -- blob analysis of the actual frame:
#   20260105_postwinterbreak, 20251210_frosty
#
# DETECTOR-FLAGGED ONLY -- the 1-D signature is strong and the run names are
# consistent with a laser being present, but these have not been confirmed by
# looking at the 2-D frame. Listed so the suite is usable as a gate; each still
# wants eyes on it before being treated as settled.
KNOWN_TWO_DOT = {
    "20260105_postwinterbreak": "verified in 2D: dot at (1439,626), blob at (1399,997)",
    "20251210_frosty": "verified in 2D: compact source HWHM 3px, 508px from the fit",
    "20250924_PostNate": "detector only",
    "20250926_dotrefound": "detector only",
    "20251020_morning5237": "detector only",
    "20251027_night5237": "detector only",
    "20251105_overnight": "detector only",
    "20251107_laserweekend": "detector only; laser run",
    "20251111_laserday": "detector only; laser run",
    "20251111_lasernight": "detector only; laser run",
    "20251112_laserdaytest": "detector only; laser run",
    "20251112_lasernighttest": "detector only; laser run",
    "20251117_lasersimultaneous": "detector only; laser run",
    "20251118_lasermirror": "detector only; laser run",
    "20251120_pumpsoff": "detector only",
    "20251124_prethanksgiving": "detector only",
    "20251125_thanksgiving": "detector only",
    "20251201_snowday": "detector only",
    "20251205_lowhumidity": "detector only",
    "20251217_warming": "detector only",
    "20260217_morewarming": "detector only",
    "20260624_bridgetstatic": "detector only",
}


@pytest.mark.parametrize("run_key", RUN_KEYS)
def test_healthy_frame_does_not_rail(healthy, fit_cache, run_key):
    """A frame the pipeline already marked fit_ok must not have a parameter
    sitting on a bound."""
    if run_key in KNOWN_UNUSUAL:
        pytest.xfail(KNOWN_UNUSUAL[run_key])
    problems = []
    for axis in ("px", "py"):
        prof = healthy.get(f"{run_key}_{axis}")
        if prof is None:
            continue
        popt = fit_cache(f"{run_key}_{axis}", prof)
        if not np.isfinite(popt[1]):
            problems.append(f"{axis}: fit declined on a frame marked fit_ok")
            continue
        problems += [f"{axis}: {h}" for h in on_bound(prof, popt)]
    assert not problems, f"{run_key}\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("run_key", FAIL_KEYS)
def test_failure_frame_has_a_known_cause(failing, fit_cache, run_key):
    """Failures are allowed -- some are real, e.g. the dot leaving the sensor.
    What is not allowed is a failure nobody can name. 'unclassified' means this
    dataset has not been looked at.
    """
    reasons = []
    for axis in ("px", "py"):
        prof = failing.get(f"{run_key}_{axis}")
        if prof is None:
            continue
        reasons.append(
            f"{axis}={classify(prof, fit_cache(f'fail_{run_key}_{axis}', prof))}")
    assert reasons, f"{run_key}: indexed as failing but no fixture profile"
    assert not any("unclassified" in r for r in reasons), \
        f"{run_key}: failure with no known cause -- {reasons}"


@pytest.mark.parametrize("run_key", RUN_KEYS)
def test_two_component_frames_are_flagged(healthy, fit_cache, run_key):
    """A fit that has locked onto the wrong one of two sources.

    This does NOT assert the fit is wrong -- with a laser in frame the broad fit
    may be the intended measurement. It asserts we KNOW the second source is
    there, because this failure mode is otherwise silent: the fit converges
    cleanly, nothing sits on a bound, the residual is low and fit_ok is True.
    """
    px = healthy.get(f"{run_key}_px")
    py = healthy.get(f"{run_key}_py")
    if px is None or py is None:
        pytest.skip("no healthy fixture")
    hit = second_component(px, py,
                           fit_cache(f"{run_key}_px", px),
                           fit_cache(f"{run_key}_py", py))
    if hit is None:
        return
    ix, iy, sep, sig = hit
    assert run_key in KNOWN_TWO_DOT, (
        f"{run_key}: undeclared second component at (x={ix}, y={iy}), "
        f"{sep:.0f}px from the fitted centroid, {sig:.1f} sigma")


def test_failure_census_does_not_regress(failing, fit_cache):
    """The whole picture as one number.

    Categories that are bugs must be zero. Categories that are real -- a dot
    genuinely off the sensor -- are counted and printed, not asserted away.
    'now_fits' counts frames the old code rejected that the new code fits
    cleanly; those are a result, not a problem.
    """
    census = collections.Counter()
    detail = collections.defaultdict(list)
    for key, prof in sorted(failing.items()):
        label = classify(prof, fit_cache(f"fail_{key}", prof))
        census[label] += 1
        detail[label].append(key)
    report = "\n  ".join(f"{k}: {v}" for k, v in sorted(census.items()))

    assert census["railed_sigma"] == 0, f"still railing sigma:\n  {report}"
    assert census["railed_mu"] == 0, f"still railing mu:\n  {report}"
    assert census["no_convergence"] == 0, f"fits declining outright:\n  {report}"
    assert census["unclassified"] == 0, (
        f"failures with no known cause:\n  {report}\n  "
        + ", ".join(detail["unclassified"][:10]))
