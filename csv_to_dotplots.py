"""
csv_to_dotplots.py

Consume a per-run {runname}_frames.csv produced by fits_reprocess.py and emit the
plot + summary artifacts that dot_movie-Copy3.ipynb would have written for the same run:

    {runname}_position_reprocess.png    # X and Y centroid shift over time (with slope line)
    {runname}_FWHM_reprocess.png        # X and Y FWHM over time (with slope line)
    {runname}_FFT_reprocess.png         # 2x2 FFT panel: centroid X/Y, FWHM X/Y
    {runname}_summary_reprocess.csv     # one-row summary (means and stds, pixel + arcsec)

The `_reprocess` suffix keeps these separate from the dot_movie-Copy3 outputs so both
pipelines can coexist in the same run folder for direct side-by-side comparison. The
movie (.mp4) is NOT produced here because it requires the underlying image data;
fits_reprocess.py handles that.

The position and FWHM plots are drawn against wall-clock time, styled after
run_bridget_comparison.ipynb (solid major gridlines, dashed minor ones). The tick
intervals adapt to the run's duration -- days/6 hours for a multi-day run, down to
seconds for a high-speed burst -- so a run that collects thousands of frames in a
couple of hours stays readable. The date is always recoverable: it is in the tick
labels for runs of two days or more, stacked under the clock time for shorter runs
that cross midnight, and in the x-axis label otherwise. Runs whose timestamps are
missing or identical fall back to the old frame-number axis.

Frame rate is auto-detected from the CSV timestamps when median dt >= 5 s (the
per-minute regime); otherwise the caller's --frame-rate value is used (default 1/60,
matching dot_movie-Copy3.ipynb's default). Pixel scale is looked up by runname in
pixel_scales.csv, falling back to the caller's --pixel-scale (default 0.15).

CLI:
    python csv_to_dotplots.py path/to/foo_frames.csv
    python csv_to_dotplots.py foo_frames.csv --out-dir plots/ --pixel-scale 0.15
    python csv_to_dotplots.py foo_frames.csv --frame-rate 52.37 --notes "vibration test"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; fits_reprocess runs this in worker context
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

DEFAULT_PIXEL_SCALE = 0.15          # arcsec/pixel; dot_movie-Copy3.ipynb hardcoded value
DEFAULT_FRAME_RATE  = 1.0 / 60.0    # Hz; dot_movie-Copy3.ipynb default (one frame per minute)
DEFAULT_FWHM_MIN_PX = 1.0
DEFAULT_FWHM_MAX_PX = 1000.0

# When median timestamp dt is this large, trust it as the frame period.
TIMESTAMP_FRAME_RATE_THRESHOLD_S = 5.0

PIXEL_SCALES_PATH = Path(__file__).parent / "pixel_scales.csv"


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------

def load_frames(csv_path: Path) -> pd.DataFrame:
    """Read a _frames.csv, parse timestamps, and put rows in chronological order.

    Prefers timestamp for ordering (authoritative), falling back to frame_num if
    timestamps are missing/degenerate. This is defensive because the BMP/PNG code
    path in fits_reprocess.py has historically produced non-unique frame_num values
    when the filename tail contains a HH-MM-SS block."""
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    ts_usable = ("timestamp" in df.columns
                 and df["timestamp"].notna().sum() >= 2
                 and df["timestamp"].nunique() > 1)
    if ts_usable:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    elif "frame_num" in df.columns:
        df = df.sort_values("frame_num", kind="stable").reset_index(drop=True)
    return df


def apply_filter(
    df: pd.DataFrame,
    fwhm_min: float = DEFAULT_FWHM_MIN_PX,
    fwhm_max: float = DEFAULT_FWHM_MAX_PX,
    require_fit_ok: bool = True,
) -> pd.DataFrame:
    """Drop failed fits, NaNs, and rows with FWHM outside [fwhm_min, fwhm_max]."""
    mask = np.isfinite(df["mu_x"]) & np.isfinite(df["mu_y"]) \
        & np.isfinite(df["sigma_x"]) & np.isfinite(df["sigma_y"]) \
        & np.isfinite(df["fwhm_x"]) & np.isfinite(df["fwhm_y"])
    mask &= (df["fwhm_x"] > fwhm_min) & (df["fwhm_x"] < fwhm_max)
    mask &= (df["fwhm_y"] > fwhm_min) & (df["fwhm_y"] < fwhm_max)
    if require_fit_ok and "fit_ok" in df.columns:
        mask &= df["fit_ok"].astype(bool)
    return df.loc[mask].reset_index(drop=True)


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add mu_x_rel, mu_y_rel columns (relative to first frame)."""
    if df.empty:
        df = df.copy()
        df["mu_x_rel"] = []
        df["mu_y_rel"] = []
        return df
    df = df.copy()
    df["mu_x_rel"] = df["mu_x"] - df["mu_x"].iloc[0]
    df["mu_y_rel"] = df["mu_y"] - df["mu_y"].iloc[0]
    return df


# ---------------------------------------------------------------------------
# Frame-rate + pixel-scale resolution
# ---------------------------------------------------------------------------

def estimate_frame_rate_from_timestamps(df: pd.DataFrame) -> Optional[float]:
    """Return frame rate in Hz if timestamps span multiple frames at >=5s spacing."""
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return None
    ts = df["timestamp"].dropna().sort_values()
    if len(ts) < 2:
        return None
    dts = ts.diff().dropna().dt.total_seconds()
    dts = dts[dts > 0]
    if dts.empty:
        return None
    median_dt = float(dts.median())
    if median_dt < TIMESTAMP_FRAME_RATE_THRESHOLD_S:
        return None
    return 1.0 / median_dt


def lookup_pixel_scale(runname: str, path: Path = PIXEL_SCALES_PATH,
                       run_key: Optional[str] = None) -> Optional[float]:
    """Return the pixel scale (arcsec/px) from pixel_scales.csv, or None if empty/missing.

    Run names repeat across dates, so prefer an exact run_key ('{date}_{runname}')
    match when the caller knows it. Falls back to the first row matching the bare
    runname, which is all that pre-run_key files and standalone CLI use can offer."""
    if not path.exists():
        return None
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    match = None
    if run_key:
        match = next((r for r in rows if r.get("run_key") == run_key), None)
    if match is None:
        match = next((r for r in rows if r.get("runname") == runname), None)
    if match is None:
        return None

    v = (match.get("pixel_scale_arcsec_per_pixel") or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# The x-axis is wall-clock time, styled after run_bridget_comparison.ipynb: solid
# major gridlines, dashed minor ones. Which interval each gets is picked from the
# run's duration, because runs range from a couple of seconds of high-speed frames
# to nearly two weeks of once-a-minute frames. Tiers are (min span in seconds,
# major locator, minor locator, tick format), tried longest-first, and each locator
# comes from a factory because a Locator instance cannot be shared across axes.
#
# Sub-day locators pass an explicit by* list rather than interval=N: interval
# anchors the ticks on the first frame's clock time (a run starting at 13:51 would
# be gridded at 14:13, 14:43, ...), while a by* list pins them to round clock
# values, which is what makes two runs comparable side by side.
_HOURS = lambda step: list(range(0, 24, step))
_SIXTY = lambda step: list(range(0, 60, step))

_TIME_TIERS = [
    (8 * 86400, lambda: mdates.DayLocator(interval=2),
                lambda: mdates.DayLocator(),                        "%b %d"),
    (2 * 86400, lambda: mdates.DayLocator(),
                lambda: mdates.HourLocator(byhour=_HOURS(6)),       "%b %d"),
    (12 * 3600, lambda: mdates.HourLocator(byhour=_HOURS(6)),
                lambda: mdates.HourLocator(byhour=_HOURS(1)),       "%H:%M"),
    (4 * 3600,  lambda: mdates.HourLocator(byhour=_HOURS(1)),
                lambda: mdates.MinuteLocator(byminute=_SIXTY(15)),  "%H:%M"),
    (2 * 3600,  lambda: mdates.MinuteLocator(byminute=_SIXTY(30)),
                lambda: mdates.MinuteLocator(byminute=_SIXTY(10)),  "%H:%M"),
    (3600,      lambda: mdates.MinuteLocator(byminute=_SIXTY(15)),
                lambda: mdates.MinuteLocator(byminute=_SIXTY(5)),   "%H:%M"),
    (20 * 60,   lambda: mdates.MinuteLocator(byminute=_SIXTY(5)),
                lambda: mdates.MinuteLocator(byminute=_SIXTY(1)),   "%H:%M"),
    (8 * 60,    lambda: mdates.MinuteLocator(byminute=_SIXTY(2)),
                lambda: mdates.SecondLocator(bysecond=_SIXTY(30)),  "%H:%M"),
    (3 * 60,    lambda: mdates.MinuteLocator(byminute=_SIXTY(1)),
                lambda: mdates.SecondLocator(bysecond=_SIXTY(15)),  "%H:%M"),
    (90,        lambda: mdates.SecondLocator(bysecond=_SIXTY(30)),
                lambda: mdates.SecondLocator(bysecond=_SIXTY(10)),  "%H:%M:%S"),
    (40,        lambda: mdates.SecondLocator(bysecond=_SIXTY(15)),
                lambda: mdates.SecondLocator(bysecond=_SIXTY(5)),   "%H:%M:%S"),
    (15,        lambda: mdates.SecondLocator(bysecond=_SIXTY(5)),
                lambda: mdates.SecondLocator(bysecond=_SIXTY(1)),   "%H:%M:%S"),
    # Sub-15s bursts need sub-second ticks; None picks the millisecond formatter.
    (0,         lambda: mdates.AutoDateLocator(minticks=3, maxticks=7),
                lambda: mdates.AutoDateLocator(minticks=6, maxticks=15), None),
]

# At or above this span the tick format already carries the date.
DATE_IN_TICKS_MIN_SPAN_S = 2 * 86400


class _XAxis(NamedTuple):
    """Resolved x-axis for one run's plots."""
    values: np.ndarray   # matplotlib date numbers when is_time, else frame indices
    is_time: bool
    span_s: float        # first -> last, in seconds (0.0 when not is_time)
    t0: Optional[pd.Timestamp]
    t1: Optional[pd.Timestamp]
    valid: np.ndarray    # bool mask of rows with a usable x value


def _resolve_x(df: pd.DataFrame) -> _XAxis:
    """Plot against timestamps when they exist and actually advance, else frame index.

    Falling back matters for runs whose frames all share one timestamp -- the
    high-speed BMP runs stamp whole seconds, so a two-second burst can have every
    frame at the same instant -- and for CSVs with no timestamp column at all."""
    n = len(df)
    if "timestamp" in df.columns and n >= 2:
        ts = df["timestamp"]
        valid = ts.notna().to_numpy()
        if valid.sum() >= 2:
            tv = ts[valid]
            t0, t1 = tv.iloc[0], tv.iloc[-1]
            span = float((t1 - t0).total_seconds())
            if span > 0:
                # NaT rows become NaN date numbers, which matplotlib draws as gaps.
                return _XAxis(mdates.date2num(ts), True, span, t0, t1, valid)
    return _XAxis(np.arange(n), False, 0.0, None, None, np.ones(n, dtype=bool))


def _rate_unit(span_s: float):
    """Pick the denominator for the slope readout: (label, seconds per unit)."""
    if span_s >= 2 * 86400:
        return "day", 86400.0
    if span_s >= 3600:
        return "hr", 3600.0
    if span_s >= 60:
        return "min", 60.0
    return "s", 1.0


def _date_span_label(t0: pd.Timestamp, t1: pd.Timestamp) -> str:
    if t0.date() == t1.date():
        return t0.strftime("%b %d, %Y")
    if t0.year == t1.year:
        return f"{t0.strftime('%b %d')}–{t1.strftime('%b %d, %Y')}"
    return f"{t0.strftime('%b %d, %Y')}–{t1.strftime('%b %d, %Y')}"


def _ticks_carry_date(xax: _XAxis) -> bool:
    """True when the major tick labels name the date themselves.

    Long runs get date-formatted ticks outright. Shorter ones get clock-only ticks,
    which repeat once the run crosses midnight ("17:00" twice in a 40-hour run), so
    those get the date stacked under the time instead."""
    return (xax.span_s >= DATE_IN_TICKS_MIN_SPAN_S
            or xax.t0.date() != xax.t1.date())


def _style_axis(ax, xax: _XAxis) -> None:
    """Apply the locators, formatter, and two-level grid for this run's x-axis."""
    if not xax.is_time:
        ax.grid(True)
        return
    major, minor, fmt = next(t[1:] for t in _TIME_TIERS if xax.span_s >= t[0])
    major_loc = major()
    ax.xaxis.set_major_locator(major_loc)
    if fmt is None:
        # AutoDateFormatter would print six decimal places and drop the hour.
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _pos: mdates.num2date(x).strftime("%H:%M:%S.%f")[:-3]))
    else:
        if xax.span_s < DATE_IN_TICKS_MIN_SPAN_S and _ticks_carry_date(xax):
            fmt += "\n%b %d"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.xaxis.set_minor_locator(minor())
    ax.grid(which="major", linestyle="-", alpha=0.5)
    ax.grid(which="minor", linestyle="--", alpha=0.25)


def _finish_x_axis(fig, bottom_ax, xax: _XAxis) -> None:
    """Label the shared x-axis and rotate its ticks."""
    if not xax.is_time:
        bottom_ax.set_xlabel("Frame")
        return
    # Ticks that show clock time only get the run's date named here instead.
    label = ("Time" if _ticks_carry_date(xax)
             else f"Time — {_date_span_label(xax.t0, xax.t1)}")
    bottom_ax.set_xlabel(label)
    fig.autofmt_xdate(rotation=45, ha="right")


def _slope_line(ax, xax: _XAxis, ys, color: str, alpha: float = 0.5):
    """Draw the first-to-last chord and label it with the slope in px per time unit."""
    xs = np.asarray(xax.values, dtype=float)
    yv = np.asarray(ys, dtype=float)
    ok = xax.valid & np.isfinite(xs) & np.isfinite(yv)
    if ok.sum() < 2:
        return 0.0
    i, j = np.flatnonzero(ok)[[0, -1]]
    line, = ax.plot([xs[i], xs[j]], [yv[i], yv[j]], color=color, alpha=alpha)
    if xax.is_time:
        unit, per_unit = _rate_unit(xax.span_s)
        dx = xax.span_s / per_unit
    else:
        unit, dx = "frame", xs[j] - xs[i]
    slope = (yv[j] - yv[i]) / dx if dx else 0.0
    ax.legend([line], [f"Slope = {slope:.4f} px/{unit}"])
    return slope


def _plot_pair(df: pd.DataFrame, out_path: Path, cols, ylabels) -> Path:
    """Two stacked panels sharing a time x-axis, each with its own slope chord."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    xax = _resolve_x(df)
    for ax, col, ylabel, color in zip(axs, cols, ylabels, ("blue", "green")):
        if len(df):
            ax.plot(xax.values, df[col], color=color)
            _slope_line(ax, xax, df[col].to_numpy(), "red")
        ax.set_ylabel(ylabel)
        _style_axis(ax, xax)
    _finish_x_axis(fig, axs[1], xax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_position(df: pd.DataFrame, out_path: Path) -> Path:
    """X and Y centroid shift over time with slope line."""
    return _plot_pair(df, out_path, ("mu_x_rel", "mu_y_rel"),
                      ("X centroid shift (pixels)", "Y centroid shift (pixels)"))


def plot_fwhm(df: pd.DataFrame, out_path: Path) -> Path:
    """X and Y FWHM over time with slope line."""
    return _plot_pair(df, out_path, ("fwhm_x", "fwhm_y"),
                      ("FWHM X (pixels)", "FWHM Y (pixels)"))


def _fft_power(signal: np.ndarray, dt: float):
    n = len(signal)
    fft = np.fft.fft(signal - np.mean(signal))
    freqs = np.fft.fftfreq(n, d=dt)
    mask = freqs > 0
    return freqs[mask], np.abs(fft[mask]) ** 2


def plot_fft(df: pd.DataFrame, dt: float, out_path: Path) -> Path:
    """2x2 FFT of centroid X/Y and FWHM X/Y — matches dot_movie-Copy3."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    if len(df) >= 2 and dt and dt > 0:
        panels = [
            (axs[0, 0], df["mu_x_rel"].to_numpy(), "FFT of centroid (relative x)"),
            (axs[0, 1], df["mu_y_rel"].to_numpy(), "FFT of centroid (relative y)"),
            (axs[1, 0], df["fwhm_x"].to_numpy(),   "FFT of FWHM (x)"),
            (axs[1, 1], df["fwhm_y"].to_numpy(),   "FFT of FWHM (y)"),
        ]
        for ax, sig, title in panels:
            freqs, power = _fft_power(sig, dt)
            ax.plot(freqs, power)
            ax.set_title(title)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Power")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(
    df: pd.DataFrame,
    runname: str,
    frame_rate: float,
    pixel_scale: float,
    notes: str = "",
    n_frames_total: Optional[int] = None,
) -> dict:
    """Build a single-row summary dict matching dot_movie-Copy3's columns
    (with both px and arcsec variants). Values are NaN if df is empty."""
    if df.empty:
        base = {"mu_x_rel": [np.nan], "mu_y_rel": [np.nan],
                "fwhm_x":   [np.nan], "fwhm_y":   [np.nan]}
    else:
        base = df

    def _mean_std(col):
        s = np.asarray(base[col], dtype=float) if not df.empty else np.array([np.nan])
        return float(np.nanmean(s)), float(np.nanstd(s))

    def _first_ts(): return str(df["timestamp"].iloc[0])  if not df.empty and "timestamp" in df else ""
    def _last_ts():  return str(df["timestamp"].iloc[-1]) if not df.empty and "timestamp" in df else ""

    mux_m,  mux_s  = _mean_std("mu_x_rel")
    muy_m,  muy_s  = _mean_std("mu_y_rel")
    fwx_m,  fwx_s  = _mean_std("fwhm_x")
    fwy_m,  fwy_s  = _mean_std("fwhm_y")

    return {
        "filename":         runname,
        "runname":          runname,
        "total frames":     int(n_frames_total) if n_frames_total is not None else int(len(df)),
        "good frames":      int(len(df)),
        "start time":       _first_ts(),
        "stop time":        _last_ts(),
        "notes":            notes,
        "frame rate":       frame_rate,
        "frame interval (s)": (1.0 / frame_rate) if frame_rate else np.nan,
        "pixel scale (arcsec/px)": pixel_scale,
        "x position (px)":  mux_m,  "x pos std (px)":  mux_s,
        "y position (px)":  muy_m,  "y pos std (px)":  muy_s,
        "FWHM x (px)":      fwx_m,  "FWHM x std (px)": fwx_s,
        "FWHM y (px)":      fwy_m,  "FWHM y std (px)": fwy_s,
        "x position (as)":  mux_m * pixel_scale,  "x pos std (as)":  mux_s * pixel_scale,
        "y position (as)":  muy_m * pixel_scale,  "y pos std (as)":  muy_s * pixel_scale,
        "FWHM x (as)":      fwx_m * pixel_scale,  "FWHM x std (as)": fwx_s * pixel_scale,
        "FWHM y (as)":      fwy_m * pixel_scale,  "FWHM y std (as)": fwy_s * pixel_scale,
    }


def write_summary(summary: dict, out_path: Path) -> Path:
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def _resolve_runname(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[:-len("_frames")] if stem.endswith("_frames") else stem


def run(
    csv_path: Path,
    out_dir: Optional[Path] = None,
    pixel_scale: Optional[float] = None,
    frame_rate: Optional[float] = None,
    fwhm_min: float = DEFAULT_FWHM_MIN_PX,
    fwhm_max: float = DEFAULT_FWHM_MAX_PX,
    notes: str = "",
    require_fit_ok: bool = True,
    run_key: Optional[str] = None,
) -> dict:
    """Produce plots + summary for one _frames.csv. Returns a dict of output paths."""
    csv_path = Path(csv_path)
    if out_dir is None:
        out_dir = csv_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runname = _resolve_runname(csv_path)

    df_all = load_frames(csv_path)
    n_total = len(df_all)
    df = apply_filter(df_all, fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                      require_fit_ok=require_fit_ok)
    df = compute_derived(df)

    if pixel_scale is None:
        looked_up = lookup_pixel_scale(runname, run_key=run_key)
        pixel_scale = looked_up if looked_up is not None else DEFAULT_PIXEL_SCALE

    if frame_rate is None:
        est = estimate_frame_rate_from_timestamps(df if not df.empty else df_all)
        frame_rate = est if est is not None else DEFAULT_FRAME_RATE

    dt = 1.0 / frame_rate if frame_rate else 0.0

    pos_path = out_dir / f"{runname}_position_reprocess.png"
    fwhm_path = out_dir / f"{runname}_FWHM_reprocess.png"
    fft_path  = out_dir / f"{runname}_FFT_reprocess.png"
    summary_path = out_dir / f"{runname}_summary_reprocess.csv"

    plot_position(df, pos_path)
    plot_fwhm(df, fwhm_path)
    plot_fft(df, dt, fft_path)

    summary = build_summary(df, runname, frame_rate=frame_rate,
                            pixel_scale=pixel_scale, notes=notes,
                            n_frames_total=n_total)
    write_summary(summary, summary_path)

    return {
        "runname":     runname,
        "n_total":     n_total,
        "n_kept":      int(len(df)),
        "pixel_scale": pixel_scale,
        "frame_rate":  frame_rate,
        "position_png": pos_path,
        "fwhm_png":     fwhm_path,
        "fft_png":      fft_path,
        "summary_csv":  summary_path,
    }


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("csv", type=Path, help="path to a {runname}_frames.csv")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: alongside the CSV)")
    p.add_argument("--pixel-scale", type=float, default=None,
                   help=f"arcsec/pixel (default: lookup in pixel_scales.csv, "
                        f"else {DEFAULT_PIXEL_SCALE})")
    p.add_argument("--frame-rate", type=float, default=None,
                   help=f"Hz (default: auto-detect from timestamps, "
                        f"else {DEFAULT_FRAME_RATE:.6f})")
    p.add_argument("--fwhm-min", type=float, default=DEFAULT_FWHM_MIN_PX)
    p.add_argument("--fwhm-max", type=float, default=DEFAULT_FWHM_MAX_PX)
    p.add_argument("--notes", default="")
    p.add_argument("--no-fit-ok-filter", action="store_true",
                   help="skip the fit_ok filter (useful for Genie CSVs where fit_ok is always False)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run(
        args.csv,
        out_dir=args.out_dir,
        pixel_scale=args.pixel_scale,
        frame_rate=args.frame_rate,
        fwhm_min=args.fwhm_min,
        fwhm_max=args.fwhm_max,
        notes=args.notes,
        require_fit_ok=not args.no_fit_ok_filter,
    )
    print(f"[{result['runname']}] {result['n_kept']}/{result['n_total']} frames kept "
          f"(pixel_scale={result['pixel_scale']} as/px, "
          f"frame_rate={result['frame_rate']:.6f} Hz)")
    print(f"  {result['position_png']}")
    print(f"  {result['fwhm_png']}")
    print(f"  {result['fft_png']}")
    print(f"  {result['summary_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())