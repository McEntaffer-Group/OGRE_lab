"""
plot_run_environment.py

Generate a bridget_comparison_environment.png-style figure for one or more runs
that have no primary-only (Genie) counterpart. Where run_bridget_comparison.ipynb
stacks four panels (primary-only drift, full-system drift, temperature, humidity),
a solo run has three:

    1. Centroid drift X/Y (pixels)
    2. Temperature  (SHT / MCP / HDC)
    3. Relative humidity (SHT / HDC)

Styling matches the notebook: Proxima Nova, Okabe-Ito colours, solid major
gridlines with dashed minors, transparent 300-dpi PNG. The x-axis tick intervals
come from csv_to_dotplots._TIME_TIERS rather than the notebook's fixed
DayLocator, so a 13-day run does not end up with 13 crowded day labels.

CLI:
    python plot_run_environment.py thanksgiving realwinterbreak allmetal
    python plot_run_environment.py allmetal --out-dir figures/
    python plot_run_environment.py thanksgiving --no-fit-ok-filter
    python plot_run_environment.py postspie --with-run postspiegenie
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv_to_dotplots as cdp
import fits_reprocess as fr

PROJECT_ROOT = Path(__file__).parent
TEMP_DIR = PROJECT_ROOT / "temperature"
# The parent temperature/ folder holds a legacy CSV with a NaT row that crashes
# temp_functions._date_range_from_csv; daily/ is the clean per-day archive.
TEMP_DAILY = TEMP_DIR / "daily"

ENV_RESAMPLE = "30s"
OUTLIER_IQR_MULT = 1.5

# Okabe-Ito palette -- colourblind-safe, 4:1+ contrast on white (as in the notebook)
_BLUE       = "#0072B2"
_VERMILLION = "#D55E00"
_GREEN      = "#009E73"
_PURPLE     = "#CC79A7"

_TEMP_SERIES = [
    ("SHT_Temperature_C", _VERMILLION, "SHT"),
    ("MCP_Temperature_C", _GREEN,      "MCP"),
    ("HDC_Temperature_C", _PURPLE,     "HDC"),
]
_HUM_SERIES = [
    ("SHT_Relative_Humidity", _VERMILLION, "SHT"),
    ("HDC_Relative_Humidity", _PURPLE,     "HDC"),
]

_FONT_DIR = Path(r"D:\Users\jad507\OneDrive - The Pennsylvania State University"
                 r"\Documents\AstroStats\Fonts")


def apply_notebook_style() -> None:
    """Load Proxima Nova and the notebook rcParams; fall back silently if absent."""
    try:
        from matplotlib import font_manager
        if _FONT_DIR.is_dir():
            for f in _FONT_DIR.glob("*.ttf"):
                if " It.ttf" not in f.name:
                    font_manager.fontManager.addfont(str(f))
            font_manager.fontManager.ttflist = [
                e for e in font_manager.fontManager.ttflist
                if not ("Proxima" in e.name and " It.ttf" in Path(e.fname).name)
            ]
            plt.rcParams["font.family"] = "Proxima Nova"
    except Exception:
        pass  # styling is cosmetic; never block a figure on a missing font
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "figure.titlesize": 15, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "grid.color": "0.75", "legend.facecolor": "white",
        "legend.edgecolor": "black", "legend.framealpha": 1.0,
    })


# ---------------------------------------------------------------------------
# Locating and loading a run
# ---------------------------------------------------------------------------

def find_run(name: str, root: Path = None) -> Path:
    """Resolve a bare runname to its run directory, searching both layouts."""
    root = root or fr.ROOT
    matches = [r for r in fr._discover_fits_runs(root) + fr._discover_image_runs(root)
               if r.name == name]
    if not matches:
        raise FileNotFoundError(f"No run directory named {name!r} under {root}")
    if len(matches) > 1:
        # Run names repeat across dates; newest wins, but say so.
        matches.sort(key=lambda p: p.parent.name)
        print(f"    note: {len(matches)} runs named {name!r}; using {matches[-1].parent.name}")
    return matches[-1]


def frames_csv(run_path: Path) -> Path:
    """Locate {runname}_frames.csv under either output layout.

    fits_reprocess.output_dir_for() writes a FITS run's artifacts into the run dir
    but an image run's up into {date}/, so check both."""
    runname = run_path.name
    for cand in (run_path / f"{runname}_frames.csv",
                 run_path.parent / f"{runname}_frames.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No {runname}_frames.csv beside or above {run_path}")


def _despike_mask(df: pd.DataFrame, cols, window: int = 11,
                  n_mad: float = 8.0) -> pd.Series:
    """Flag isolated single-frame excursions, preserving sustained drift.

    A global Tukey/IQR fence (what run_bridget_comparison uses) is wrong for these
    runs: on thanksgiving it deletes 74% of a contiguous 26-hour window where the
    dot genuinely drifted out and stayed there -- the very event the plot exists to
    show. Comparing each point to a centred rolling MEDIAN instead means a real
    excursion drags the median with it and survives, while a lone bad fit does not."""
    keep = pd.Series(True, index=df.index)
    for c in cols:
        s = df[c].astype(float)
        resid = (s - s.rolling(window, center=True, min_periods=1).median()).abs()
        mad = float(np.median(resid.dropna()))
        if mad <= 0:
            continue  # flat/quiet series: nothing to scale against, keep everything
        keep &= resid <= n_mad * 1.4826 * mad
    return keep


def load_frames(run_path: Path, require_fit_ok: bool = True,
                iqr_mult: float = 0.0, despike: bool = True) -> pd.DataFrame:
    """Load a run's frames, drop failed fits, optionally despike, zero the start.

    The multi-peak filter that run_environment_plot.ipynb applies is deliberately
    NOT used here: on these runs n_peaks>1 for effectively every frame, so it would
    empty the dataframe. The global IQR fence is off by default for the reason given
    in _despike_mask; pass iqr_mult>0 to re-enable it."""
    df = pd.read_csv(frames_csv(run_path), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)

    finite = np.isfinite(df["mu_x"]) & np.isfinite(df["mu_y"])
    keep = finite & df["fit_ok"].astype(bool) if require_fit_ok else finite
    n_fitdrop = int((~keep).sum())
    df = df[keep].copy()
    if df.empty:
        raise ValueError(
            f"No frames survive the fit_ok filter for {run_path.name} "
            f"({n_total:,} rows). Re-run with --no-fit-ok-filter."
        )

    n_spike = n_out = 0
    if despike:
        m = _despike_mask(df, ("mu_x", "mu_y"))
        n_spike = int((~m).sum())
        df = df[m].copy()
    if iqr_mult > 0:
        def fence(s, m):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            return q1 - m * (q3 - q1), q3 + m * (q3 - q1)
        lo_x, hi_x = fence(df["mu_x"], iqr_mult)
        lo_y, hi_y = fence(df["mu_y"], iqr_mult)
        outlier = ((df["mu_x"] < lo_x) | (df["mu_x"] > hi_x)
                   | (df["mu_y"] < lo_y) | (df["mu_y"] > hi_y))
        n_out = int(outlier.sum())
        df = df[~outlier].copy()

    df["mu_x_rel"] = df["mu_x"] - df["mu_x"].iloc[0]
    df["mu_y_rel"] = df["mu_y"] - df["mu_y"].iloc[0]
    df = df.set_index("timestamp")
    detail = f"{n_fitdrop:,} failed fit/non-finite, {n_spike:,} isolated spikes"
    if iqr_mult > 0:
        detail += f", {n_out:,} IQR outliers at {iqr_mult}x"
    print(f"    {len(df):,} of {n_total:,} frames kept ({detail})")
    return df


def break_gaps(df: pd.DataFrame, cols, gap_factor: float = 5.0) -> pd.DataFrame:
    """Insert NaN rows wherever the cadence jumps, so gaps render as gaps.

    Both the acquisition itself and the IQR fence can remove a contiguous block of
    frames. Without this, matplotlib joins the surviving neighbours with a straight
    line that reads as real, slow, linear drift -- the most misleading thing this
    plot could show."""
    if len(df) < 3:
        return df
    dt = df.index.to_series().diff()
    cadence = dt.median()
    if pd.isna(cadence) or cadence <= pd.Timedelta(0):
        return df
    breaks = df.index[dt > gap_factor * cadence]
    if len(breaks) == 0:
        return df
    filler = pd.DataFrame(np.nan, index=breaks - cadence / 2, columns=list(cols))
    out = pd.concat([df[list(cols)], filler]).sort_index()
    print(f"    {len(breaks)} gap(s) > {gap_factor:g}x cadence "
          f"({cadence.total_seconds():.0f}s) drawn as breaks")
    return out


def load_environment(start, end, resample: str = ENV_RESAMPLE) -> pd.DataFrame:
    sys.path.insert(0, str(TEMP_DIR))
    import temp_functions as tf
    return tf.builder(start, end, source_dir=str(TEMP_DAILY), resample_freq=resample)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _style_time_axis(axes, t0: pd.Timestamp, t1: pd.Timestamp):
    """Apply csv_to_dotplots' duration-matched locators to every panel.

    Returns the x-axis label, which carries the date when the ticks do not."""
    span_s = float((t1 - t0).total_seconds())
    major, minor, fmt = next(t[1:] for t in cdp._TIME_TIERS if span_s >= t[0])
    ticks_have_date = (span_s >= cdp.DATE_IN_TICKS_MIN_SPAN_S
                       or t0.date() != t1.date())
    if fmt is not None and span_s < cdp.DATE_IN_TICKS_MIN_SPAN_S and ticks_have_date:
        fmt += "\n%b %d"
    for ax in axes:
        ax.xaxis.set_major_locator(major())
        if fmt is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.xaxis.set_minor_locator(minor())
        ax.grid(which="major", linestyle="-", alpha=0.5)
        ax.grid(which="minor", linestyle="--", alpha=0.25)
    return "Time" if ticks_have_date else f"Time \u2014 {cdp._date_span_label(t0, t1)}"


def plot_run(drift_panels, env: pd.DataFrame, title: str, out_path: Path,
             t0=None, t1=None) -> Path:
    """Stacked panels sharing a time x-axis: one per drift series, then temp + humidity.

    drift_panels is a list of (frames, ylabel). A solo run passes one and gets the
    three-panel layout; a run with a companion camera passes two and gets the
    four-panel run_bridget_comparison layout."""
    n_drift = len(drift_panels)
    fig, axes = plt.subplots(n_drift + 2, 1, figsize=(13, 3 + 2.7 * (n_drift + 1)),
                             sharex=True, gridspec_kw={"hspace": 0.07})
    ax_t, ax_h = axes[-2], axes[-1]

    for ax_d, (frames, ylabel) in zip(axes, drift_panels):
        drift = break_gaps(frames, ("mu_x_rel", "mu_y_rel"))
        ax_d.plot(drift.index, drift["mu_x_rel"], color=_BLUE, label="X", lw=1.5)
        ax_d.plot(drift.index, drift["mu_y_rel"], color=_VERMILLION, label="Y", lw=1.5)
        ax_d.axhline(0, color="0.5", lw=0.5)
        ax_d.set_ylabel(ylabel)
        ax_d.legend(loc="upper right")

    for col, color, label in _TEMP_SERIES:
        if col in env.columns and env[col].notna().any():
            ax_t.plot(env.index, env[col], color=color, label=label, lw=1.5)
    ax_t.set_ylabel(r"Temperature ($^\circ$C)")
    ax_t.legend(loc="upper right")

    for col, color, label in _HUM_SERIES:
        if col in env.columns and env[col].notna().any():
            ax_h.plot(env.index, env[col], color=color, label=label, lw=1.5)
    ax_h.set_ylabel("Relative humidity (%)")
    ax_h.legend(loc="upper right")

    if t0 is None:
        t0 = min(f.index.min() for f, _ in drift_panels)
        t1 = max(f.index.max() for f, _ in drift_panels)
    xlabel = _style_time_axis(axes, t0, t1)
    ax_h.set_xlabel(xlabel)
    # Rotate in place rather than via fig.autofmt_xdate, whose subplots_adjust
    # call fights tight_layout below (same approach as the notebook).
    for label in ax_h.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    fig.suptitle(title)
    # No tight_layout: it silently overrides the gridspec hspace that gives the
    # panels their tight stacking. bbox_inches="tight" at save time prevents any
    # clipping of the rotated tick labels.
    fig.subplots_adjust(top=0.93, left=0.07, right=0.99, bottom=0.13)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------

DEFAULT_COMPANION_LABEL = "Primary + secondary (px)"
DEFAULT_MAIN_LABEL = "Full system drift (px)"
SOLO_LABEL = "Centroid drift (px)"


def build(runname: str, out_dir: Optional[Path] = None,
          require_fit_ok: bool = True, iqr_mult: float = 0.0,
          despike: bool = True, companion: Optional[str] = None,
          companion_label: str = DEFAULT_COMPANION_LABEL,
          main_label: str = DEFAULT_MAIN_LABEL) -> Path:
    print(f"[{runname}]")
    run_path = find_run(runname)
    frames = load_frames(run_path, require_fit_ok=require_fit_ok,
                         iqr_mult=iqr_mult, despike=despike)

    comp_frames = None
    if companion:
        print(f"  [{companion}]")
        comp_frames = load_frames(find_run(companion), require_fit_ok=require_fit_ok,
                                  iqr_mult=iqr_mult, despike=despike)

    # Environment window spans both cameras when there is a companion.
    starts = [frames.index.min()] + ([comp_frames.index.min()] if comp_frames is not None else [])
    ends   = [frames.index.max()] + ([comp_frames.index.max()] if comp_frames is not None else [])
    t0, t1 = min(starts), max(ends)
    hours = (t1 - t0).total_seconds() / 3600.0
    print(f"    window {t0}  ->  {t1}  ({hours:.1f} h)")
    env = load_environment(t0, t1)
    print(f"    env rows: {len(env):,}")

    if comp_frames is not None:
        panels = [(comp_frames, companion_label), (frames, main_label)]
        heading = f"{companion_label.replace(' (px)', '')} vs {main_label.replace(' drift (px)', '')}"
        title = (f"{runname} \u2014 {heading}\n"
                 f"{t0:%Y-%m-%d %H:%M}  \u2192  {t1:%Y-%m-%d %H:%M}  ({hours:.1f} h)")
    else:
        panels = [(frames, SOLO_LABEL)]
        title = (f"{runname} \u2014 Centroid Drift vs Environment\n"
                 f"{t0:%Y-%m-%d %H:%M}  \u2192  {t1:%Y-%m-%d %H:%M}  ({hours:.1f} h)")

    out_dir = Path(out_dir) if out_dir else run_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{runname}_environment_comparison.png"
    plot_run(panels, env, title, out_path, t0=t0, t1=t1)
    print(f"    saved {out_path}")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("runs", nargs="+", help="run names, e.g. thanksgiving realwinterbreak")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: the run's own folder)")
    p.add_argument("--no-fit-ok-filter", action="store_true",
                   help="keep frames whose fit_ok is False (centroids may still be valid)")
    p.add_argument("--no-despike", action="store_true",
                   help="keep isolated single-frame excursions")
    p.add_argument("--iqr-mult", type=float, default=0.0,
                   help="also apply a global Tukey fence at this multiple "
                        "(default 0 = off; it deletes sustained real drift)")
    p.add_argument("--with-run", default=None, metavar="RUNNAME",
                   help="companion camera run for a second drift panel "
                        "(e.g. postspiegenie); only valid with a single run")
    p.add_argument("--companion-label", default=DEFAULT_COMPANION_LABEL,
                   help=f"y-label for the companion panel (default: {DEFAULT_COMPANION_LABEL!r})")
    p.add_argument("--main-label", default=DEFAULT_MAIN_LABEL,
                   help=f"y-label for the main panel (default: {DEFAULT_MAIN_LABEL!r})")
    args = p.parse_args(argv)
    if args.with_run and len(args.runs) != 1:
        p.error("--with-run pairs one companion to one main run")

    apply_notebook_style()
    failed = []
    for runname in args.runs:
        try:
            build(runname, out_dir=args.out_dir,
                  require_fit_ok=not args.no_fit_ok_filter,
                  iqr_mult=args.iqr_mult, despike=not args.no_despike,
                  companion=args.with_run,
                  companion_label=args.companion_label,
                  main_label=args.main_label)
        except Exception as exc:
            print(f"    !! {runname} failed: {type(exc).__name__}: {exc}")
            failed.append(runname)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
