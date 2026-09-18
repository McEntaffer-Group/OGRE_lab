"""The two drift readouts on the position/FWHM plots.

`Endpoints` is the legacy chord: (y_last - y_first) / span, which is what
dot_movie-Copy3.py:564 and rvts.py:235 compute. It is kept so the legacy plots
stay comparable by eye. Note it is NOT checked by compare_pipelines --
TRUTH_PAIRS compares position and FWHM means and stds, and no slope is among
them.

`Least squares` uses every frame. The two disagree whenever the run is not
monotonic: on allmetal they report opposite signs in y (-4.49 vs +6.92 px/day)
for a run that climbs 70 px, holds for a day and a half, then comes back.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import csv_to_dotplots as c2d


def axis_for(n, span_s=None, is_time=True):
    """An _XAxis over n frames, as _resolve_x would produce."""
    if not is_time:
        return c2d._XAxis(np.arange(n, dtype=float), False, 0.0, None, None,
                          np.ones(n, dtype=bool))
    t0 = pd.Timestamp("2026-02-13 16:36:54")
    ts = pd.Series(pd.date_range(t0, periods=n, freq=pd.Timedelta(seconds=span_s / (n - 1))))
    import matplotlib.dates as mdates
    return c2d._XAxis(mdates.date2num(ts), True, float(span_s), ts.iloc[0],
                      ts.iloc[-1], np.ones(n, dtype=bool))


def run_slope(ys, span_s=None, is_time=True):
    ys = np.asarray(ys, dtype=float)
    xax = axis_for(len(ys), span_s, is_time)
    fig, ax = plt.subplots()
    try:
        return c2d._slope_line(ax, xax, ys, "red")
    finally:
        plt.close(fig)


# --- the legacy chord must not drift ---------------------------------------

def test_chord_matches_the_legacy_formula_exactly():
    """dot_movie-Copy3.py:564 -- (y[-1] - y[0]) / (x[-1] - x[0]), in frames."""
    ys = np.array([0.0, 5.0, -3.0, 11.0, 8.0])
    chord, _ = run_slope(ys, is_time=False)
    assert chord == pytest.approx((ys[-1] - ys[0]) / (len(ys) - 1))


def test_chord_uses_only_the_endpoints():
    """Anything can happen in between without moving the chord."""
    a = run_slope([0.0, 1.0, 2.0, 3.0, 10.0], is_time=False)[0]
    b = run_slope([0.0, 900.0, -900.0, 40.0, 10.0], is_time=False)[0]
    assert a == pytest.approx(b)


def test_chord_is_reported_per_day_on_a_multi_day_run():
    """allmetal: 2.72 days, so _rate_unit picks 'day'."""
    span = 2.72 * 86400
    ys = np.linspace(0.0, -64.0, 500)
    chord, fit = run_slope(ys, span_s=span)
    assert chord == pytest.approx(-64.0 / 2.72, rel=1e-6)
    assert fit == pytest.approx(chord, rel=1e-6), (
        "on a perfectly linear run the two must agree"
    )


# --- where they diverge ----------------------------------------------------

def test_they_disagree_in_SIGN_on_a_there_and_back_run():
    """The allmetal y shape: rise, plateau, return to near the start.

    The chord sees two nearly equal endpoints and reports ~no drift; the fit sees
    a run that spent most of its life far above where it began."""
    n, rise, plateau = 900, 135, 675          # 15% up, 75% held, 10% back down
    ys = np.concatenate([np.linspace(0, 70, rise),
                         np.full(plateau, 70.0),
                         np.linspace(70, -12, n - rise - plateau)])
    chord, fit = run_slope(ys, span_s=2.72 * 86400)
    assert chord < 0, "endpoints: ended below where it started"
    assert fit > 0, "least squares: spent most of the run above where it started"
    assert np.sign(chord) != np.sign(fit)


def test_one_bad_endpoint_wrecks_the_chord_but_only_dents_the_fit():
    """A blank-frame fit lands at mu=1e-10 on its lower bound. If it is frame 0,
    it sets the entire reported chord.

    Least squares is more robust, not immune: one 100 px outlier in 400 frames
    still moves it ~15%. The claim worth pinning is the ratio."""
    good = np.linspace(100.0, 110.0, 400)
    chord_ok, fit_ok = run_slope(good, span_s=86400 * 3)

    bad = good.copy()
    bad[0] = 0.0                      # the degenerate fit
    chord_bad, fit_bad = run_slope(bad, span_s=86400 * 3)

    d_chord = abs(chord_bad - chord_ok) / abs(chord_ok)
    d_fit = abs(fit_bad - fit_ok) / abs(fit_ok)
    assert d_chord == pytest.approx(10.0, rel=0.05)   # 1000%: 10 px -> 110 px
    assert d_fit < 0.20                               # ~15%
    assert d_chord > 50 * d_fit


# --- shape and degenerate cases -------------------------------------------

def test_returns_both_values():
    out = run_slope([0.0, 1.0, 2.0], is_time=False)
    assert isinstance(out, tuple) and len(out) == 2


def test_two_points_gives_a_chord_and_no_fit():
    chord, fit = run_slope([0.0, 4.0], is_time=False)
    assert chord == pytest.approx(4.0)
    assert fit == 0.0


def test_fewer_than_two_valid_points_is_zero():
    assert run_slope([np.nan], is_time=False) == (0.0, 0.0)


def test_nan_frames_are_excluded_from_both():
    ys = np.array([0.0, np.nan, 2.0, np.nan, 4.0])
    chord, fit = run_slope(ys, is_time=False)
    assert chord == pytest.approx(1.0)     # (4-0)/4, endpoints are finite
    assert fit == pytest.approx(1.0)


# --- worst-case drift (peak-to-peak) --------------------------------------

def summary_for(mu_x, mu_y, ts=None):
    n = len(mu_x)
    d = pd.DataFrame({"mu_x_rel": np.asarray(mu_x, float),
                      "mu_y_rel": np.asarray(mu_y, float),
                      "fwhm_x": np.full(n, 5.0), "fwhm_y": np.full(n, 5.0)})
    if ts is not None:
        d["timestamp"] = ts
    return c2d.build_summary(d, "t", frame_rate=1 / 60, pixel_scale=0.15)


def test_range_is_peak_to_peak_not_endpoint_difference():
    """The number the two drift rates cannot express."""
    ys = [0.0, 70.0, -20.0, 5.0]
    s = summary_for([0.0] * 4, ys)
    assert s["y min (px)"] == pytest.approx(-20.0)
    assert s["y max (px)"] == pytest.approx(70.0)
    assert s["y range (px)"] == pytest.approx(90.0)
    # the endpoints only span 5 px; the dot travelled 90
    assert abs(ys[-1] - ys[0]) == pytest.approx(5.0)


def test_range_is_also_reported_in_arcsec():
    s = summary_for([0.0, 10.0], [0.0, 0.0])
    assert s["x range (as)"] == pytest.approx(10.0 * 0.15)


def test_range_records_when_each_extreme_happened():
    ts = pd.to_datetime(["2026-02-13 16:00", "2026-02-14 04:00",
                         "2026-02-15 09:00", "2026-02-16 02:00"])
    s = summary_for([0.0, 0.0, 0.0, 0.0], [0.0, -20.0, 70.0, 5.0], ts=ts)
    assert s["y min time"].startswith("2026-02-14 04:00")
    assert s["y max time"].startswith("2026-02-15 09:00")


def test_range_ignores_nan_frames():
    s = summary_for([0.0, np.nan, 12.0], [0.0, np.nan, -3.0])
    assert s["x range (px)"] == pytest.approx(12.0)
    assert s["y range (px)"] == pytest.approx(3.0)


def test_range_is_nan_on_an_empty_run():
    s = c2d.build_summary(pd.DataFrame(), "t", frame_rate=1.0, pixel_scale=0.15)
    assert np.isnan(s["x range (px)"]) and np.isnan(s["y range (px)"])
    assert s["x min time"] == ""


# --- against the real run --------------------------------------------------

from pathlib import Path

E = Path("E:/Reverse Telescope Test Data")


@pytest.mark.skipif(not E.is_dir(), reason="E: not mounted")
def test_allmetal_reproduces_the_documented_disagreement():
    d = pd.read_csv(E / "20260213_data/allmetal/allmetal_frames.csv",
                    parse_dates=["timestamp"])
    d = c2d.apply_filter(d)
    d = c2d.compute_derived(d)
    xax = c2d._resolve_x(d)

    fig, ax = plt.subplots()
    try:
        chord_y, fit_y = c2d._slope_line(ax, xax, d.mu_y_rel.to_numpy(), "red")
        chord_x, fit_x = c2d._slope_line(ax, xax, d.mu_x_rel.to_numpy(), "red")
    finally:
        plt.close(fig)

    # y: opposite signs -- the case the second line exists for
    assert chord_y == pytest.approx(-4.49, abs=0.05)
    assert fit_y == pytest.approx(6.92, abs=0.05)
    # x: monotonic, so they broadly agree
    assert chord_x == pytest.approx(-23.75, abs=0.05)
    assert fit_x == pytest.approx(-25.97, abs=0.05)

    # and the worst case both rates miss: the dot covered 93 px in y while its
    # reported drift was a few px/day either way.
    s = c2d.build_summary(d, "allmetal", frame_rate=1 / 60, pixel_scale=0.15)
    assert s["y range (px)"] == pytest.approx(93.20, abs=0.05)
    assert s["x range (px)"] == pytest.approx(69.83, abs=0.05)
