"""
csv_to_dotplots.py

Consume a per-run {runname}_frames.csv produced by fits_reprocess.py and emit the
plot + summary artifacts that dot_movie-Copy3.ipynb would have written for the same run:

    {runname}_position_reprocess.png    # X and Y centroid shift over frames (with slope line)
    {runname}_FWHM_reprocess.png        # X and Y FWHM over frames (with slope line)
    {runname}_FFT_reprocess.png         # 2x2 FFT panel: centroid X/Y, FWHM X/Y
    {runname}_summary_reprocess.csv     # one-row summary (means and stds, pixel + arcsec)

The `_reprocess` suffix keeps these separate from the dot_movie-Copy3 outputs so both
pipelines can coexist in the same run folder for direct side-by-side comparison. The
movie (.mp4) is NOT produced here because it requires the underlying image data;
fits_reprocess.py handles that.

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
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; fits_reprocess runs this in worker context
import matplotlib.pyplot as plt
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


def lookup_pixel_scale(runname: str, path: Path = PIXEL_SCALES_PATH) -> Optional[float]:
    """Return the pixel scale (arcsec/px) from pixel_scales.csv, or None if empty/missing."""
    if not path.exists():
        return None
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("runname") != runname:
                    continue
                v = (row.get("pixel_scale_arcsec_per_pixel") or "").strip()
                if not v:
                    return None
                return float(v)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _slope_line(ax, xs, ys, color: str, alpha: float = 0.5):
    line, = ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color=color, alpha=alpha)
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if (xs[-1] - xs[0]) else 0.0
    ax.legend([line], [f"Slope = {slope:.4f} px/frame"])
    return slope


def plot_position(df: pd.DataFrame, out_path: Path) -> Path:
    """X and Y centroid shift over frames with slope line — matches dot_movie-Copy3."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    frames = np.arange(len(df))
    if len(df):
        axs[0].plot(frames, df["mu_x_rel"], color="blue")
        _slope_line(axs[0], frames, df["mu_x_rel"].to_numpy(), "red")
        axs[1].plot(frames, df["mu_y_rel"], color="green")
        _slope_line(axs[1], frames, df["mu_y_rel"].to_numpy(), "red")
    axs[0].set_ylabel("X centroid shift (pixels)")
    axs[0].grid(True)
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Y centroid shift (pixels)")
    axs[1].grid(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fwhm(df: pd.DataFrame, out_path: Path) -> Path:
    """X and Y FWHM over frames with slope line — matches dot_movie-Copy3."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    frames = np.arange(len(df))
    if len(df):
        axs[0].plot(frames, df["fwhm_x"], color="blue")
        _slope_line(axs[0], frames, df["fwhm_x"].to_numpy(), "red")
        axs[1].plot(frames, df["fwhm_y"], color="green")
        _slope_line(axs[1], frames, df["fwhm_y"].to_numpy(), "red")
    axs[0].set_ylabel("FWHM X (pixels)")
    axs[0].grid(True)
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("FWHM Y (pixels)")
    axs[1].grid(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
        looked_up = lookup_pixel_scale(runname)
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