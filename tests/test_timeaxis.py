"""Sub-second reconstruction for runs whose filenames stamp whole seconds.

31 of 96 runs record more than two frames per timestamp (nightvideo: 99). Plotted
against that clock they draw one vertical stroke per second, and any px/s slope is
fitted against a 1 Hz grid.

The delicate part is the edge buckets. Capture begins and ends part-way through a
second, so the first and last buckets are partial; spreading them across a full
second they never occupied misplaces the two points with the most leverage on a
slope fit. On short runs that is not a rounding error -- maxtest2's y slope moves
23.5% -- while on the 180 s runs it is ~0.1%.
"""

import numpy as np
import pandas as pd
import pytest

import csv_to_dotplots as c2d


def stamps(spec, start="2025-09-17 09:05:30", unit="us"):
    """Build a timestamp column from [(second_offset, count), ...]."""
    out = []
    t0 = pd.Timestamp(start)
    for off, k in spec:
        out += [t0 + pd.Timedelta(seconds=off)] * k
    return pd.Series(pd.DatetimeIndex(out).as_unit(unit))


def frame(ts):
    return pd.DataFrame({"timestamp": ts,
                         "frame_num": np.arange(len(ts)),
                         "mu_x": np.linspace(100, 110, len(ts)),
                         "mu_y": np.linspace(200, 190, len(ts))})


# --- the invariant that matters most --------------------------------------

def test_every_frame_stays_inside_its_stamped_second():
    """The stamped second is the only thing the filename actually tells us; the
    reconstruction may subdivide it but must never move a frame out of it."""
    ts = stamps([(0, 40), (1, 50), (2, 50), (3, 50), (4, 30)])
    out = c2d._subsecond_times(ts)
    assert (out >= ts).all()
    assert (out < ts + pd.Timedelta(seconds=1)).all()


def test_order_is_preserved():
    ts = stamps([(0, 10), (1, 50), (2, 50), (3, 50), (4, 20)])
    out = c2d._subsecond_times(ts)
    assert (out.diff().dropna() >= pd.Timedelta(0)).all()


# --- the edge-bucket correction -------------------------------------------

def test_sparse_first_bucket_sits_at_the_END_of_its_second():
    """A run that began mid-second has few frames in bucket 0, and they belong at
    the end of it. Spreading them across the whole second puts the leftmost point
    up to a second early, at maximum leverage on the slope."""
    ts = stamps([(0, 4), (1, 50), (2, 50), (3, 50), (4, 50)])
    out = c2d._subsecond_times(ts)
    first = out[:4]

    assert first.min() > ts.iloc[0] + pd.Timedelta(seconds=0.9), (
        "4 frames at a 50 Hz rate occupy the last ~0.08 s, not the whole second"
    )
    assert first.max() < ts.iloc[0] + pd.Timedelta(seconds=1)
    # and they are still spaced at the interior rate, not squeezed arbitrarily
    step = (first.iloc[1] - first.iloc[0]).total_seconds()
    assert step == pytest.approx(1 / 50, rel=0.05)


def test_sparse_last_bucket_sits_at_the_START_of_its_second():
    ts = stamps([(0, 50), (1, 50), (2, 50), (3, 50), (4, 3)])
    out = c2d._subsecond_times(ts)
    last = out[-3:]
    assert last.max() < ts.iloc[-1] + pd.Timedelta(seconds=0.1)


def test_a_full_first_bucket_gives_the_same_answer_either_way():
    """The backwards anchoring must not introduce a seam when bucket 0 is full.

    With k == rate, `1 - (k - i)/r` and `i/r` are the same expression, so a full
    first bucket starts exactly on its second just like every other bucket."""
    ts = stamps([(0, 50), (1, 50), (2, 50), (3, 50), (4, 50)])
    out = c2d._subsecond_times(ts)
    assert out.iloc[0] == pd.Timestamp("2025-09-17 09:05:30")
    forward = out[50:100].iloc[0]
    assert forward == pd.Timestamp("2025-09-17 09:05:31")


def test_rate_comes_from_interior_buckets_only():
    """If the edge buckets counted toward the rate, a sparse start would drag the
    estimate down and stretch every interior bucket past its own second."""
    ts = stamps([(0, 2), (1, 50), (2, 50), (3, 50), (4, 2)])
    out = c2d._subsecond_times(ts)
    interior = out[2:52]
    step = np.median(np.diff(interior.astype("int64"))) / 1e6
    assert step == pytest.approx(1 / 50, rel=0.05)
    assert (out < ts + pd.Timedelta(seconds=1)).all()


def test_bucket_fuller_than_the_rate_does_not_spill():
    """max(rate, k) keeps an over-full bucket inside its second."""
    ts = stamps([(0, 50), (1, 50), (2, 200), (3, 50), (4, 50)])
    out = c2d._subsecond_times(ts)
    assert (out < ts + pd.Timedelta(seconds=1)).all()


# --- when not to guess -----------------------------------------------------

def test_too_few_buckets_returns_none():
    """Two buckets cannot determine a rate; the caller falls back to frame index
    rather than inventing a clock. This is maxtest1 (100 frames, 2 buckets)."""
    assert c2d._subsecond_times(stamps([(0, 50), (1, 50)])) is None


def test_resolve_x_falls_back_to_frame_index_when_rate_undetermined():
    xax = c2d._resolve_x(frame(stamps([(0, 50), (1, 50)])))
    assert not xax.is_time
    assert list(xax.values) == list(range(100))


def test_resolve_x_leaves_well_resolved_timestamps_alone():
    """One frame per second: nothing to subdivide, and the values must be the
    original timestamps."""
    ts = stamps([(i, 1) for i in range(60)])
    df = frame(ts)
    xax = c2d._resolve_x(df)
    assert xax.is_time
    import matplotlib.dates as mdates
    assert np.allclose(xax.values, mdates.date2num(ts))


def test_resolve_x_subdivides_a_stacked_run():
    ts = stamps([(0, 40), (1, 50), (2, 50), (3, 50), (4, 50)])
    xax = c2d._resolve_x(frame(ts))
    assert xax.is_time
    assert len(np.unique(xax.values)) == len(ts), (
        "every frame should land on its own x after subdivision"
    )


def test_gapped_bursts_keep_their_dead_time():
    """maxtest2/nightvideo/stability are duty-cycled: bursts with real gaps. Each
    bucket is laid out from its own second, so the gaps survive."""
    ts = stamps([(0, 50), (2, 50), (4, 50), (6, 50), (8, 50)])
    out = c2d._subsecond_times(ts)
    d = out.diff().dropna().dt.total_seconds()
    assert (d.max() > 1.0), "the dead time between bursts must remain"
    assert (out < ts + pd.Timedelta(seconds=1)).all()


# --- the real corpus -------------------------------------------------------

from pathlib import Path

E = Path("E:/Reverse Telescope Test Data")
needs_e = pytest.mark.skipif(not E.is_dir(), reason="E: not mounted")


@needs_e
@pytest.mark.parametrize("rel", [
    "20250917/maxtest2_frames.csv",
    "20250918/stability_frames.csv",
    "20250918/nightvideo_frames.csv",
])
def test_real_runs_stay_inside_their_stamped_seconds(rel):
    d = pd.read_csv(E / rel, parse_dates=["timestamp"])
    d = d.sort_values(["timestamp", "frame_num"]).reset_index(drop=True)
    out = c2d._subsecond_times(d.timestamp)
    assert out is not None
    assert (out >= d.timestamp).all()
    assert (out < d.timestamp + pd.Timedelta(seconds=1)).all()
    assert (out.diff().dropna() >= pd.Timedelta(0)).all()
