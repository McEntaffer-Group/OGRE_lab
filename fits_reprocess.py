"""
fits_reprocess.py

Walk E:/Reverse Telescope Test Data and process all run data:
  - {date}_data/{runname}/{runname}_fits/*.fits  (FITS files)
  - {date}/{runname}/*.bmp or *.png              ({date} folders with no {date}_data sibling)

For each run, write {runname}_frames.csv with per-frame dot data:

    frame_num, filename, timestamp,
    mu_x, mu_y, sigma_x, sigma_y, fwhm_x, fwhm_y,
    amp_x, amp_y, offset_x, offset_y,
    fit_ok, n_peaks_x, n_peaks_y

All spatial values are in PIXELS. No arcsec conversion happens here; downstream code
looks up per-run pixel scale in pixel_scales.csv when needed.

Behavior:
  * Always reprocesses runs (never skips). If a _frames.csv already exists, it is copied
    as {runname}_frames_prev.csv to reprocess_output/ before being overwritten.
  * Results written to original location AND mirrored to reprocess_output/ for easy review.
  * After the CSV is written, csv_to_dotplots.py runs on it (unless --no-plots),
    producing {runname}_{position,FWHM,FFT}_reprocess.png and
    {runname}_summary_reprocess.csv (mirrored to reprocess_output/).
    The `_reprocess` suffix lets these coexist with the dot_movie-Copy3.ipynb outputs
    for side-by-side comparison; no existing files are overwritten.
  * A movie is written alongside them as {runname}_reprocess.mp4 (unless --no-movie).
    Frames are streamed lazily so runs of any size fit in memory. Movies are NOT
    mirrored to reprocess_output/ (they can be large).
  * OUTPUT LOCATION follows dot_movie-Copy3: artifacts sit one level above the
    frames, never inside the folder holding them (see output_dir_for).
      FITS  {date}_data/{runname}/       <- outputs, beside {runname}_fits/
      IMAGE {date}/                      <- outputs, beside {runname}/
    Keeping them out of the scanned folder is also what stops a re-run from
    reading its own plots back as input frames. Image runs written under the
    older layout can be swept up with --migrate-image-outputs.
  * All *_summary.csv truth files under ROOT are concatenated into
    reprocess_output/all_runs_summary.csv (read-only pass, originals untouched).
    All *_summary_reprocess.csv from this pipeline go into all_runs_summary_reprocess.csv.
  * run_timings.csv is APPENDED to (never overwritten) with a batch_timestamp column
    so historical timings are preserved across runs.
  * pixel_scales.csv is UPDATED IN PLACE preserving every prior row; only rows for
    runs in the current batch are refreshed (and user-filled pixel_scale/notes are
    always kept).
  * Single Gaussian always fit; n_peaks_x/n_peaks_y are informational only and do not
    gate fit_ok. Camera inferred from name: "genie" -> dalsa_genie, else main_camera.
  * Parallel fits via ProcessPoolExecutor (bypasses GIL; curve_fit is CPU-bound).

Run from the repo root:
    python fits_reprocess.py                       # all runs, plots + movie on
    python fits_reprocess.py --run zoeysecondary   # one run by name (substring match)
    python fits_reprocess.py --no-movie            # skip .mp4 generation
    python fits_reprocess.py --no-plots            # skip plots + summary CSV
    python fits_reprocess.py --dry-run             # list work, don't process
    python fits_reprocess.py --migrate-image-outputs          # preview the sweep
    python fits_reprocess.py --migrate-image-outputs --apply  # perform it
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# csv_to_dotplots lives alongside this file; import lazily-safe (no side effects at import).
import csv_to_dotplots

ROOT = Path("E:/Reverse Telescope Test Data")
PIXEL_SCALES_PATH = Path(__file__).parent / "pixel_scales.csv"
OUTPUT_DIR = Path(__file__).parent / "reprocess_output"

WORKERS = max(1, (os.cpu_count() or 16) - 2)

MOVIE_FPS = 20
MOVIE_IMSHOW_VMIN = 0
# Frames are 8-bit greyscale (0-255) end to end, so the color scale is fixed
# to the full native range instead of being guessed from a percentile sample
# -- a percentile gets fooled whenever the bright spot is a small fraction of
# the frame (it ends up measuring background, not signal).
# The original dot_movie-Copy3.ipynb used a hardcoded vmax=100 with no
# explanation; best guess is that early runs drove the LED at low voltage, so
# the camera never approached saturation and 100 happened to sit above the
# real peak -- fine for those runs, but not a real ceiling for this data.
MOVIE_IMSHOW_VMAX = 255
# Number of frames sampled up-front to derive vmax + profile-axis limits without
# holding the whole run in memory.
MOVIE_SAMPLE_FRAMES = 10
# Ask ffmpeg for the NVIDIA hardware H.264 encoder; falls back to libx264
# automatically if the encoder isn't available or the call fails.
MOVIE_USE_NVENC = True

PEAK_HEIGHT_FRAC = 0.50
PEAK_MIN_DISTANCE_PX = 50

FWHM_MIN_PX = 1.0
FWHM_MAX_PX = 1000.0

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548; sigma -> FWHM

# FITS filename pattern: "runname0001 26-04-20 15-16-47.fits"
_TS_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})\.fits$", re.IGNORECASE)
_FRAMENUM_RE = re.compile(r"_?(\d{3,6})\s+\d{2}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}\.fits$", re.IGNORECASE)

# BMP/PNG filename pattern: "runname1661 26-08-15 17-40-00.bmp" or
# "runname_01102 26-08-15 08-26-00.png". Frame number is the 3-6 digit run before
# the space-separated YY-MM-DD HH-MM-SS timestamp block (NOT any digits inside the
# timestamp, which is why we can't just grab the last digit run in the stem).
_IMG_TS_RE = re.compile(
    r"(\d{2})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})\.(?:bmp|png)$", re.IGNORECASE)
_IMG_FRAMENUM_RE = re.compile(
    r"_?(\d{3,6})\s+\d{2}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}\.(?:bmp|png)$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"(\d+)")

# Plots written INTO the run directory, next to the data. Image runs are
# discovered by globbing *.png there, so without this filter a second pass over
# the same run ingests its own plots as three extra "frames".
#   *_reprocess.png  - csv_to_dotplots (this pipeline)
#   bare *.png       - dot_movie-Copy3.ipynb, same three plots without the suffix
# Only image runs have ever had the reprocess variants, but dot_movie writes to
# the run directory too, so both are excluded.
#
# This is deliberately a denylist of known generated names rather than an
# allowlist of camera-frame names: 13,872 real frames (e.g.
# "1-833-stability3054.bmp") carry no timestamp and would be silently dropped by
# a pattern match on the usual "name#### YY-MM-DD HH-MM-SS" convention.
_GENERATED_PNG_RE = re.compile(
    r"_(?:FFT|FWHM|position)(?:_reprocess)?\.png$", re.IGNORECASE)

_CSV_COLS = [
    "frame_num", "filename", "timestamp",
    "mu_x", "mu_y", "sigma_x", "sigma_y", "fwhm_x", "fwhm_y",
    "amp_x", "amp_y", "offset_x", "offset_y",
    "fit_ok", "n_peaks_x", "n_peaks_y",
]


def _worker_init():
    """Ignore Ctrl-C in worker processes; the main process handles shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Fitting primitives
# ---------------------------------------------------------------------------

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def _estimate_sigma(profile: np.ndarray, x: np.ndarray, mu_guess: float) -> float:
    """Estimate sigma via weighted second moment of background-subtracted profile."""
    baseline = np.median(profile)
    shifted = np.clip(profile - baseline, 0.0, None)
    total = float(shifted.sum())
    if total <= 0.0:
        return 50.0
    weights = shifted / total
    var = float(np.sum(weights * (x - mu_guess) ** 2))
    return float(np.clip(np.sqrt(max(var, 1.0)), 2.0, profile.size / 4.0))


def _fit_one_profile(profile: np.ndarray):
    """Fit a single Gaussian to a 1D profile. Returns (amp, mu, sigma, offset) or NaNs."""
    try:
        x = np.arange(profile.size, dtype=np.float64)
        mu_guess = float(profile.argmax())
        sigma_est = _estimate_sigma(profile, x, mu_guess)
        bounds = (
            [0.0, 0.0, 1.0, -np.inf],
            [np.inf, float(profile.size), profile.size / 2.0, np.inf],
        )
        p0_amp = float(profile.max())
        p0_off = float(np.median(profile))
        popt = None
        for s in [sigma_est, sigma_est / 2.0, sigma_est * 2.0, 10.0, 50.0, 100.0]:
            try:
                popt, _ = curve_fit(
                    gaussian, x, profile,
                    p0=[p0_amp, mu_guess, s, p0_off],
                    bounds=bounds, maxfev=5000,
                )
                break
            except Exception:
                continue
        if popt is None:
            return np.nan, np.nan, np.nan, np.nan
        amp, mu, sigma, offset = popt
        return amp, mu, abs(sigma), offset
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _count_peaks(profile: np.ndarray) -> int:
    """Count comparable bright peaks (informational; does not gate fit_ok)."""
    pmax = float(profile.max())
    pmin = float(profile.min())
    if not np.isfinite(pmax) or pmax <= pmin:
        return 0
    height = pmin + PEAK_HEIGHT_FRAC * (pmax - pmin)
    peaks, _ = find_peaks(profile, height=height, distance=PEAK_MIN_DISTANCE_PX)
    return int(len(peaks))


def _empty_row(filename: str, frame_num: int, timestamp: str) -> dict:
    return {
        "filename": filename, "frame_num": frame_num, "timestamp": timestamp,
        "mu_x": np.nan, "mu_y": np.nan,
        "sigma_x": np.nan, "sigma_y": np.nan,
        "fwhm_x": np.nan, "fwhm_y": np.nan,
        "amp_x": np.nan, "amp_y": np.nan,
        "offset_x": np.nan, "offset_y": np.nan,
        "fit_ok": False, "n_peaks_x": 0, "n_peaks_y": 0,
    }


def _fill_fit_results(out: dict, px_profile: np.ndarray, py_profile: np.ndarray) -> dict:
    amp_x, mu_x, sigma_x, offset_x = _fit_one_profile(px_profile)
    amp_y, mu_y, sigma_y, offset_y = _fit_one_profile(py_profile)
    out.update({
        "mu_x": mu_x, "mu_y": mu_y,
        "sigma_x": sigma_x, "sigma_y": sigma_y,
        "amp_x": amp_x, "amp_y": amp_y,
        "offset_x": offset_x, "offset_y": offset_y,
        "n_peaks_x": _count_peaks(px_profile),
        "n_peaks_y": _count_peaks(py_profile),
    })
    if np.isfinite(sigma_x) and np.isfinite(sigma_y):
        fwhm_x = sigma_x * FWHM_FACTOR
        fwhm_y = sigma_y * FWHM_FACTOR
        out["fwhm_x"] = fwhm_x
        out["fwhm_y"] = fwhm_y
        out["fit_ok"] = bool(
            FWHM_MIN_PX < fwhm_x < FWHM_MAX_PX
            and FWHM_MIN_PX < fwhm_y < FWHM_MAX_PX
            and np.isfinite(mu_x) and np.isfinite(mu_y)
        )
    return out


def fit_frame(path: str) -> dict:
    """Process one FITS file. Returns a dict ready to write as a CSV row."""
    out = _empty_row(
        filename=os.path.basename(path),
        frame_num=_extract_frame_num(path),
        timestamp=_extract_timestamp(path),
    )
    try:
        with fits.open(path) as hdul:
            img = np.flip(hdul[0].data, axis=(0, 1)).astype(np.float64)
        px_profile = np.sum(img, axis=0)
        py_profile = np.sum(img, axis=1)
    except Exception:
        return out
    return _fill_fit_results(out, px_profile, py_profile)


def fit_frame_image(path: Path, frame_num: int) -> dict:
    """Process one BMP/PNG file. Returns a dict ready to write as a CSV row."""
    ts = _extract_timestamp_image(path)
    if not ts:
        # Fall back to file mtime only if the filename carries no timestamp.
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = ""
    out = _empty_row(filename=path.name, frame_num=frame_num, timestamp=ts)
    try:
        # Same 180 deg flip the FITS path applies (see fit_frame). PIL returns a
        # top-down array and the converters write it into FITS verbatim, so the
        # BMP/PNG array is identical to the FITS one and needs identical handling.
        img = np.flip(np.array(Image.open(path).convert("L")), axis=(0, 1)).astype(np.float64)
        px_profile = np.sum(img, axis=0)
        py_profile = np.sum(img, axis=1)
    except Exception:
        return out
    return _fill_fit_results(out, px_profile, py_profile)


# ---------------------------------------------------------------------------
# Filename / metadata parsing
# ---------------------------------------------------------------------------

def _extract_timestamp(path: str) -> str:
    """Parse " YY-MM-DD HH-MM-SS.fits" tail. Returns ISO string or empty."""
    name = os.path.basename(path)
    m = _TS_RE.search(name)
    if not m:
        return ""
    yy, mo, dd, hh, mm, ss = m.groups()
    return f"20{yy}-{mo}-{dd} {hh}:{mm}:{ss}"


def _extract_frame_num(path: str) -> int:
    name = os.path.basename(path)
    m = _FRAMENUM_RE.search(name)
    return int(m.group(1)) if m else -1


def _extract_frame_num_image(path: Path) -> int:
    """Extract frame number from BMP/PNG filename. Anchored to the digit run just
    before the ' YY-MM-DD HH-MM-SS.(bmp|png)' timestamp tail so we don't
    accidentally grab digits from the timestamp itself. Falls back to the last
    digit run in the stem if the timestamp tail isn't present."""
    m = _IMG_FRAMENUM_RE.search(path.name)
    if m:
        return int(m.group(1))
    groups = _DIGITS_RE.findall(path.stem)
    return int(groups[-1]) if groups else -1


def _extract_timestamp_image(path: Path) -> str:
    """Parse ' YY-MM-DD HH-MM-SS.(bmp|png)' tail. Returns ISO string or empty."""
    m = _IMG_TS_RE.search(path.name)
    if not m:
        return ""
    yy, mo, dd, hh, mm, ss = m.groups()
    return f"20{yy}-{mo}-{dd} {hh}:{mm}:{ss}"


def _detect_camera(runname: str) -> str:
    return "dalsa_genie" if "genie" in runname.lower() else "main_camera"


def list_run_images(run_dir: Path) -> list[Path]:
    """Input frames in a BMP/PNG run folder, excluding this pipeline's own plots.

    Always use this instead of globbing the run directory directly — see
    _GENERATED_PNG_RE for why."""
    images = list(run_dir.glob("*.bmp")) + list(run_dir.glob("*.png"))
    return sorted((p for p in images if not _GENERATED_PNG_RE.search(p.name)),
                  key=lambda p: p.name)


def date_prefix(run_dir: Path) -> str:
    """The {date} folder a run lives under: '20250922_data/foo' -> '20250922'."""
    date = run_dir.parent.name
    return date[: -len("_data")] if date.endswith("_data") else date


def output_dir_for(run_dir: Path, is_image_run: bool) -> Path:
    """Where a run's artifacts (CSV, plots, summary, movie) are written.

    Follows dot_movie-Copy3's convention: outputs sit one level above the frames,
    never inside the folder holding them. A FITS run keeps its data in
    {runname}_fits/, so its outputs belong in the run dir -- which is where
    dot_movie already puts {runname}_position.png and {runname}_summary.csv. An
    image run keeps frames in the run dir itself, so its outputs go up into the
    {date}/ folder rather than being mixed in with the camera data.

    Keeping outputs out of the scanned folder is also what stops a re-run from
    reading its own plots back as input frames."""
    return run_dir.parent if is_image_run else run_dir


def run_key(run_dir: Path) -> str:
    """Unique id for a run: '{date}_{runname}'.

    Run names repeat across dates -- 'morning5237' is six separate runs, and
    'minutely' is seven. Mirrored artifacts and pixel_scales rows are keyed on
    this rather than the bare runname, so a later run cannot silently overwrite
    an earlier one that happens to share its name."""
    return f"{date_prefix(run_dir)}_{run_dir.name}"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_frames_csv(df: pd.DataFrame, path: Path) -> None:
    df[_CSV_COLS].to_csv(path, index=False)


def _mirror(src: Path, dest_name: str) -> None:
    """Copy src into OUTPUT_DIR under dest_name."""
    shutil.copy2(src, OUTPUT_DIR / dest_name)


def _mirror_for_run(src: Path, run_dir: Path, suffix: str | None = None) -> None:
    """Mirror a run artifact into OUTPUT_DIR, date-prefixed so same-named runs
    on different dates do not overwrite each other.

    '{runname}_frames.csv' -> '{date}_{runname}_frames.csv'. Pass suffix to
    rename the tail, e.g. suffix='_frames_prev.csv'."""
    name = f"{run_dir.name}{suffix}" if suffix else src.name
    _mirror(src, f"{date_prefix(run_dir)}_{name}")


# ---------------------------------------------------------------------------
# Post-CSV artifacts: plots (via csv_to_dotplots) and movie
# ---------------------------------------------------------------------------

def _generate_plots_and_summary(out_csv: Path, out_dir: Path, run_dir: Path,
                                runname: str) -> None:
    """Run csv_to_dotplots on out_csv and mirror the artifacts to OUTPUT_DIR.

    Failures are logged but non-fatal — a broken plot pass should not lose the
    frames CSV that was just written."""
    try:
        result = csv_to_dotplots.run(out_csv, out_dir=out_dir, run_key=run_key(run_dir))
    except Exception as exc:
        print(f"    !! csv_to_dotplots failed for {runname}: {exc}")
        return
    for key in ("position_png", "fwhm_png", "fft_png", "summary_csv"):
        src = Path(result[key])
        try:
            _mirror_for_run(src, run_dir)
        except Exception as exc:
            print(f"    !! could not mirror {src.name}: {exc}")
    print(f"    plots: {result['n_kept']}/{result['n_total']} frames kept "
          f"(pixel_scale={result['pixel_scale']} as/px, "
          f"frame_rate={result['frame_rate']:.6f} Hz)")


def _sample_movie_stats(loader, paths, n_sample: int = MOVIE_SAMPLE_FRAMES) -> dict:
    """Sample a few frames spread across the run to derive the profile-plot
    axis ceiling without holding the whole run in memory. vmax and the
    profile floor are both fixed, cosmetic constants (see MOVIE_IMSHOW_VMAX)
    rather than anything derived from data."""
    if not paths:
        return {"vmax": MOVIE_IMSHOW_VMAX,
                "px_min": 0.0, "px_max": 1.0,
                "py_min": 0.0, "py_max": 1.0}
    idxs = np.linspace(0, len(paths) - 1, num=min(n_sample, len(paths)), dtype=int)
    samples = []
    for i in idxs:
        try:
            samples.append(loader(paths[i]))
        except Exception:
            continue
    if not samples:
        samples = [loader(paths[0])]
    px_sums = [np.sum(img, axis=0) for img in samples]
    py_sums = [np.sum(img, axis=1) for img in samples]
    return {
        "vmax":   MOVIE_IMSHOW_VMAX,
        "px_max": float(max(p.max() for p in px_sums)),
        "px_min": 0.0,
        "py_max": float(max(p.max() for p in py_sums)),
        "py_min": 0.0,
    }


def _load_fits_frame(path) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.flip(hdul[0].data, axis=(0, 1))


def _load_image_frame(path) -> np.ndarray:
    return np.flip(np.array(Image.open(path).convert("L")), axis=(0, 1))


def _write_movie(paths: list, loader, out_path: Path, timestamps: list[str] | None = None) -> None:
    """Stream frames from disk one at a time into an ffmpeg-encoded .mp4.

    Layout matches dot_movie-Copy3.ipynb: main image bottom-left, X-profile line
    plot above (shares x-axis with image), Y-profile line plot right (shares y),
    timestamp text overlay in image corner. Uses matplotlib's FuncAnimation with
    blit=True. Only one frame is held in memory at any time, so the movie step
    scales to arbitrarily long runs.

    Requests NVIDIA h264_nvenc when MOVIE_USE_NVENC is True; falls back to
    libx264 if the nvenc save fails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if not paths:
        return
    first = loader(paths[0])
    stats = _sample_movie_stats(loader, paths)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.0, hspace=0.0)
    ax_img = fig.add_subplot(gs[1, 0])
    ax_x   = fig.add_subplot(gs[0, 0], sharex=ax_img)
    ax_y   = fig.add_subplot(gs[1, 1], sharey=ax_img)
    fig.add_subplot(gs[0, 1]).axis("off")

    im = ax_img.imshow(first, cmap="viridis", origin="lower",
                       vmin=MOVIE_IMSHOW_VMIN, vmax=stats["vmax"], aspect="auto")

    profile_x = np.sum(first, axis=0)
    profile_y = np.sum(first, axis=1)
    line_x, = ax_x.plot(np.arange(first.shape[1]), profile_x)
    line_y, = ax_y.plot(profile_y, np.arange(first.shape[0]))

    ax_x.set_xlim(ax_img.get_xlim())
    ax_y.set_ylim(ax_img.get_ylim())
    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)
    ax_x.set_ylabel("counts")
    ax_y.set_xlabel("counts")
    ax_x.grid(True)
    ax_y.grid(True)
    ax_x.set_ylim(stats["px_min"], stats["px_max"])
    ax_y.set_xlim(stats["py_min"], stats["py_max"])

    ts_text = ax_img.text(0.02, 0.98, "", transform=ax_img.transAxes,
                          color="white", fontsize=12, verticalalignment="top",
                          bbox=dict(facecolor="black", alpha=0.5,
                                    edgecolor="none", pad=3))

    def _update(i):
        try:
            arr = loader(paths[i])
            im.set_array(arr)
            line_x.set_ydata(np.sum(arr, axis=0))
            line_y.set_xdata(np.sum(arr, axis=1))
        except Exception:
            pass
        if timestamps and i < len(timestamps) and timestamps[i]:
            ts_text.set_text(timestamps[i])
        return [im, line_x, line_y, ts_text]

    ani = animation.FuncAnimation(fig, _update, frames=len(paths),
                                  interval=50, blit=True)
    Writer = animation.writers["ffmpeg"]

    def _save(codec: str | None) -> None:
        kwargs = {"fps": MOVIE_FPS, "bitrate": 1800}
        if codec is not None:
            kwargs["codec"] = codec
        ani.save(str(out_path), writer=Writer(**kwargs))

    if MOVIE_USE_NVENC:
        try:
            _save("h264_nvenc")
        except Exception as exc:
            print(f"    !! h264_nvenc failed ({exc}); retrying with libx264")
            _save(None)
    else:
        _save(None)

    plt.close(fig)


def _generate_movie_from_fits(fits_files: list[str], out_path: Path) -> None:
    ts = [_extract_timestamp(p) for p in fits_files]
    try:
        _write_movie(fits_files, _load_fits_frame, out_path, timestamps=ts)
        print(f"    movie -> {out_path}")
    except Exception as exc:
        print(f"    !! movie generation failed: {exc}")


def _generate_movie_from_images(image_paths: list[Path], out_path: Path) -> None:
    ts = [_extract_timestamp_image(p) for p in image_paths]
    try:
        _write_movie(image_paths, _load_image_frame, out_path, timestamps=ts)
        print(f"    movie -> {out_path}")
    except Exception as exc:
        print(f"    !! movie generation failed: {exc}")


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _run_output_names(runname: str) -> list[str]:
    """Exact artifact filenames this pipeline writes for a run.

    Listed explicitly rather than pattern-matched so the migration sweep can
    never touch a camera frame that happens to have an unusual name."""
    return [
        f"{runname}_frames.csv",
        f"{runname}_summary_reprocess.csv",
        f"{runname}_reprocess.mp4",
        f"{runname}_FFT_reprocess.png",
        f"{runname}_FWHM_reprocess.png",
        f"{runname}_position_reprocess.png",
    ]


def migrate_image_outputs(root: Path, apply: bool = False) -> int:
    """Move image-run artifacts out of the frame folder and up into {date}/.

    Image runs used to write their outputs into the same directory as the .bmp
    frames. output_dir_for now puts them one level up, matching how FITS runs
    (and dot_movie-Copy3) keep outputs beside the data rather than inside it.
    This sweeps the leftovers from the old layout.

    Where the new location already holds a copy, the stale one is deleted --
    the parent copy was written by the current code. Otherwise the file moves.
    Dry-run unless apply=True. Returns the number of files affected."""
    moved = deleted = 0
    for run_dir in _discover_image_runs(root):
        dest_dir = output_dir_for(run_dir, is_image_run=True)
        for name in _run_output_names(run_dir.name):
            src = run_dir / name
            if not src.exists():
                continue
            dest = dest_dir / name
            if dest.exists():
                print(f"  {'delete ' if apply else 'would delete '}{src}"
                      f"\n      (superseded by {dest})")
                if apply:
                    src.unlink()
                deleted += 1
            else:
                print(f"  {'move   ' if apply else 'would move   '}{src}\n      -> {dest}")
                if apply:
                    shutil.move(str(src), str(dest))
                moved += 1

    total = moved + deleted
    if not total:
        print("  nothing to migrate; image-run outputs are already in {date}/.")
    else:
        verb = "Migrated" if apply else "Would migrate"
        print(f"\n  {verb} {total} file(s): {moved} moved, {deleted} deleted as superseded.")
        if not apply:
            print("  Re-run with --apply to perform it.")
    return total


def _discover_fits_runs(root: Path) -> list[Path]:
    """Find run folders with a *_fits subfolder under *_data/ directories."""
    runs = []
    for data_dir in sorted(root.glob("*_data")):
        if not data_dir.is_dir():
            continue
        for run_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            fits_dir = run_dir / f"{run_dir.name}_fits"
            if fits_dir.is_dir():
                runs.append(run_dir)
    return runs


def _discover_image_runs(root: Path) -> list[Path]:
    """Find run folders with BMP/PNG files under {date}/ dirs that have no {date}_data sibling."""
    runs = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or date_dir.name.endswith("_data"):
            continue
        if (root / (date_dir.name + "_data")).exists():
            continue
        for run_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if list_run_images(run_dir):
                runs.append(run_dir)
    return runs


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_fits_run(run_dir: Path, make_plots: bool = True,
                     make_movie: bool = True) -> dict:
    """Process one FITS run folder. Returns a stats dict."""
    runname = run_dir.name
    fits_dir = run_dir / f"{runname}_fits"
    out_dir = output_dir_for(run_dir, is_image_run=False)
    out_csv = out_dir / f"{runname}_frames.csv"

    stats = {
        "runname": runname, "run_key": run_key(run_dir),
        "source": str(fits_dir), "out_csv": str(out_csv),
        "n_files": 0, "n_ok": 0, "elapsed_s": 0.0,
        "camera": _detect_camera(runname), "shape": "",
    }

    if out_csv.exists():
        _mirror_for_run(out_csv, run_dir, "_frames_prev.csv")

    files = sorted(glob.glob(str(fits_dir / "*.fits")))
    stats["n_files"] = len(files)
    if not files:
        return stats

    try:
        with fits.open(files[0]) as hdul:
            sh = hdul[0].data.shape
        stats["shape"] = f"{sh[0]}x{sh[1]}"
    except Exception:
        pass

    t0 = time.time()
    results = [None] * len(files)
    print(f"  -> {runname}: {len(files)} FITS files, camera={stats['camera']}")
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init) as exe:
        futures = {exe.submit(fit_frame, f): i for i, f in enumerate(files)}
        try:
            for n_done, fut in enumerate(as_completed(futures)):
                results[futures[fut]] = fut.result()
                if n_done and n_done % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (len(files) - n_done) / max(rate, 1e-9)
                    print(f"    {n_done}/{len(files)}  ({rate:.0f} f/s, eta {eta/60:.1f} min)")
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise

    df = pd.DataFrame(results).sort_values("frame_num").reset_index(drop=True)
    _write_frames_csv(df, out_csv)
    _mirror_for_run(out_csv, run_dir)

    stats["n_ok"] = int(df["fit_ok"].sum())
    stats["elapsed_s"] = time.time() - t0

    if make_plots:
        _generate_plots_and_summary(out_csv, out_dir, run_dir, runname)
    if make_movie:
        _generate_movie_from_fits(files, out_dir / f"{runname}_reprocess.mp4")

    return stats


def process_image_run(run_dir: Path, make_plots: bool = True,
                      make_movie: bool = True) -> dict:
    """Process one BMP/PNG run folder (no FITS conversion). Returns a stats dict."""
    runname = run_dir.name
    images = list_run_images(run_dir)
    out_dir = output_dir_for(run_dir, is_image_run=True)
    out_csv = out_dir / f"{runname}_frames.csv"

    stats = {
        "runname": runname, "run_key": run_key(run_dir),
        "source": str(run_dir), "out_csv": str(out_csv),
        "n_files": len(images), "n_ok": 0, "elapsed_s": 0.0,
        "camera": _detect_camera(runname), "shape": "",
    }

    if not images:
        return stats

    if out_csv.exists():
        _mirror_for_run(out_csv, run_dir, "_frames_prev.csv")

    try:
        arr = np.array(Image.open(images[0]).convert("L"))
        stats["shape"] = f"{arr.shape[0]}x{arr.shape[1]}"
    except Exception:
        pass

    t0 = time.time()
    results = [None] * len(images)
    print(f"  -> {runname}: {len(images)} images, camera={stats['camera']}")
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_worker_init) as exe:
        futures = {
            exe.submit(fit_frame_image, p, _extract_frame_num_image(p)): i
            for i, p in enumerate(images)
        }
        try:
            for n_done, fut in enumerate(as_completed(futures)):
                results[futures[fut]] = fut.result()
                if n_done and n_done % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (len(images) - n_done) / max(rate, 1e-9)
                    print(f"    {n_done}/{len(images)}  ({rate:.0f} f/s, eta {eta/60:.1f} min)")
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise

    df = pd.DataFrame(results).sort_values("frame_num").reset_index(drop=True)
    _write_frames_csv(df, out_csv)
    _mirror_for_run(out_csv, run_dir)

    stats["n_ok"] = int(df["fit_ok"].sum())
    stats["elapsed_s"] = time.time() - t0

    if make_plots:
        _generate_plots_and_summary(out_csv, out_dir, run_dir, runname)
    if make_movie:
        _generate_movie_from_images(images, out_dir / f"{runname}_reprocess.mp4")

    return stats


# ---------------------------------------------------------------------------
# Summaries and pixel scales
# ---------------------------------------------------------------------------

def _artifact_run_key(path: Path, suffix: str) -> tuple[str, str]:
    """(run_key, runname) for a collected artifact named {runname}{suffix}.

    Handles both output layouts. A FITS run's artifacts sit inside the run dir
    ({date}_data/{runname}/), while an image run's sit one level up in {date}/.
    The runname always comes from the filename, so it is right either way; only
    where to find the date differs."""
    runname = path.name[: -len(suffix)]
    parent = path.parent
    date = date_prefix(parent) if parent.name == runname else parent.name
    if date.endswith("_data"):
        date = date[: -len("_data")]
    return f"{date}_{runname}", runname


def _collect_summaries(root: Path) -> None:
    """Collect all dot_movie-style *_summary.csv truth files into OUTPUT_DIR.
    Never modifies originals. Excludes files ending in _summary_reprocess.csv
    (those come from this pipeline and are collected separately)."""
    frames = []
    for p in sorted(root.rglob("*_summary.csv")):
        if p.name.endswith("_summary_reprocess.csv"):
            continue
        try:
            df = pd.read_csv(p)
            key, runname = _artifact_run_key(p, "_summary.csv")
            if "runname" not in df.columns:
                df.insert(0, "runname", runname)
            # runname alone is ambiguous across dates; stamp the unique key so
            # the concatenated file can distinguish same-named runs.
            df.insert(0, "run_key", key)
            _mirror(p, f"{key}_summary.csv")
            frames.append(df)
        except Exception as exc:
            print(f"  warning: could not read {p}: {exc}")
    if not frames:
        print("  no summary CSVs found under", root)
        return
    out_path = OUTPUT_DIR / "all_runs_summary.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"  collected {len(frames)} summary CSV(s) -> {out_path}")


def _collect_reprocess_summaries(root: Path) -> None:
    """Collect all *_summary_reprocess.csv files (produced by csv_to_dotplots) into
    OUTPUT_DIR / all_runs_summary_reprocess.csv. Never modifies originals."""
    frames = []
    for p in sorted(root.rglob("*_summary_reprocess.csv")):
        try:
            df = pd.read_csv(p)
            key, runname = _artifact_run_key(p, "_summary_reprocess.csv")
            if "runname" not in df.columns:
                df.insert(0, "runname", runname)
            df.insert(0, "run_key", key)
            frames.append(df)
        except Exception as exc:
            print(f"  warning: could not read {p}: {exc}")
    if not frames:
        print("  no reprocess summary CSVs found under", root)
        return
    out_path = OUTPUT_DIR / "all_runs_summary_reprocess.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"  collected {len(frames)} reprocess summary CSV(s) -> {out_path}")


_TIMING_FIELDNAMES = ["batch_timestamp", "run_key", "runname", "camera",
                      "n_files", "n_ok", "elapsed_s", "fps", "source"]


def write_timing_summary(stats_list: list[dict], grand_elapsed_s: float) -> None:
    """APPEND per-run timing rows to reprocess_output/run_timings.csv.

    Each row is tagged with the current batch_timestamp so that repeated runs
    build up a full history rather than overwriting the previous file. If the
    existing file has an older schema (no batch_timestamp column), it is
    upgraded in place — historical rows get an empty batch_timestamp string."""
    if not stats_list:
        return
    path = OUTPUT_DIR / "run_timings.csv"
    batch_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_rows: list[dict] = []
    if path.exists():
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("runname") == "_TOTAL_":
                        continue  # drop historical totals; a new one is written below
                    for k in _TIMING_FIELDNAMES:
                        row.setdefault(k, "")
                    existing_rows.append({k: row.get(k, "") for k in _TIMING_FIELDNAMES})
        except Exception as exc:
            print(f"  warning: could not read existing run_timings.csv: {exc}")

    new_rows = []
    for s in stats_list:
        fps = s["n_files"] / max(s["elapsed_s"], 1e-9) if s["n_files"] > 0 else 0.0
        new_rows.append({
            "batch_timestamp": batch_ts,
            "run_key":   s.get("run_key", ""),
            "runname":   s["runname"],
            "camera":    s["camera"],
            "n_files":   s["n_files"],
            "n_ok":      s["n_ok"],
            "elapsed_s": round(s["elapsed_s"], 2),
            "fps":       round(fps, 1),
            "source":    s["source"],
        })

    total_files = sum(s["n_files"] for s in stats_list)
    total_ok = sum(s["n_ok"] for s in stats_list)
    overall_fps = total_files / max(grand_elapsed_s, 1e-9)
    total_row = {
        "batch_timestamp": batch_ts,
        "run_key": "", "runname": "_TOTAL_", "camera": "",
        "n_files": total_files, "n_ok": total_ok,
        "elapsed_s": round(grand_elapsed_s, 2),
        "fps": round(overall_fps, 1),
        "source": "",
    }

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TIMING_FIELDNAMES)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        for r in new_rows:
            w.writerow(r)
        w.writerow(total_row)
    print(f"  run timings appended ({len(new_rows)} row(s) added) -> {path}")


def write_pixel_scales(stats_list: list[dict]) -> None:
    """Update pixel_scales.csv IN PLACE, preserving EVERY prior row.

    For runs that appear in the current batch, camera + image_shape are refreshed
    (the underlying data may have changed shape between runs). User-filled fields
    (pixel_scale_arcsec_per_pixel, notes) are preserved even for current-batch runs.
    Runs NOT in the current batch are kept exactly as they were."""
    fieldnames = ["run_key", "runname", "camera", "image_shape",
                  "pixel_scale_arcsec_per_pixel", "notes"]
    rows: dict[str, dict] = {}
    if PIXEL_SCALES_PATH.exists():
        try:
            with open(PIXEL_SCALES_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    # Rows written before run_key existed are keyed on the bare
                    # runname. Keep them under that key so nothing the user typed
                    # is lost; they are superseded once that run is processed again.
                    key = row.get("run_key") or row.get("runname")
                    if not key:
                        continue
                    rows[key] = {k: row.get(k, "") for k in fieldnames}
        except Exception as e:
            print(f"  warning: could not read existing pixel_scales.csv: {e}")

    for s in stats_list:
        key = s.get("run_key") or s["runname"]
        # Carry over anything the user filled in under the pre-run_key name.
        prev = rows.get(key) or rows.pop(s["runname"], {})
        rows[key] = {
            "run_key": key,
            "runname": s["runname"],
            "camera": s["camera"] or prev.get("camera", ""),
            "image_shape": s["shape"] or prev.get("image_shape", ""),
            "pixel_scale_arcsec_per_pixel": prev.get("pixel_scale_arcsec_per_pixel", ""),
            "notes": prev.get("notes", ""),
        }

    with open(PIXEL_SCALES_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    print(f"  pixel_scales.csv updated ({len(rows)} unique runs preserved) "
          f"-> {PIXEL_SCALES_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--root", default=str(ROOT), help="data root (default: %(default)s)")
    parser.add_argument("--run", help="process only this run name (substring match)")
    parser.add_argument("--dry-run", action="store_true", help="list discovered work, don't process")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip csv_to_dotplots plots + summary (default: on)")
    parser.add_argument("--no-movie", action="store_true",
                        help="skip .mp4 movie generation (default: on)")
    parser.add_argument("--migrate-image-outputs", action="store_true",
                        help="move image-run artifacts out of the frame folder up "
                             "into {date}/, then exit (dry-run unless --apply)")
    parser.add_argument("--apply", action="store_true",
                        help="with --migrate-image-outputs, actually move/delete files")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.migrate_image_outputs:
        migrate_image_outputs(root, apply=args.apply)
        return 0

    fits_runs = _discover_fits_runs(root)
    image_runs = _discover_image_runs(root)

    if args.run:
        fits_runs = [r for r in fits_runs if args.run in r.name]
        image_runs = [r for r in image_runs if args.run in r.name]
        if not fits_runs and not image_runs:
            print(f"No runs match '{args.run}'.", file=sys.stderr)
            return 1

    total = len(fits_runs) + len(image_runs)
    print(f"Discovered {len(fits_runs)} FITS run(s) and {len(image_runs)} image run(s) under {root}.")

    if args.dry_run:
        for r in fits_runs:
            tag = "EXISTS" if (r / f"{r.name}_frames.csv").exists() else "pending"
            print(f"  [FITS/{tag}] {r}")
        for r in image_runs:
            tag = "EXISTS" if (r / f"{r.name}_frames.csv").exists() else "pending"
            print(f"  [IMG /{tag}] {r}")
        return 0

    make_plots = not args.no_plots
    make_movie = not args.no_movie

    all_stats = []
    grand_t0 = time.time()
    try:
        for i, run in enumerate(fits_runs, 1):
            print(f"[{i}/{total}] FITS  {run.parent.name}/{run.name}")
            stats = process_fits_run(run, make_plots=make_plots, make_movie=make_movie)
            all_stats.append(stats)
            if stats["n_files"] == 0:
                print(f"  -- no FITS files found in {stats['source']}")
            else:
                rate = stats["n_files"] / max(stats["elapsed_s"], 1e-9)
                print(f"  -- {stats['n_ok']}/{stats['n_files']} ok in {stats['elapsed_s']:.1f}s ({rate:.0f} f/s)")

        for i, run in enumerate(image_runs, len(fits_runs) + 1):
            print(f"[{i}/{total}] IMAGE {run.parent.name}/{run.name}")
            stats = process_image_run(run, make_plots=make_plots, make_movie=make_movie)
            all_stats.append(stats)
            if stats["n_files"] == 0:
                print(f"  -- no images found in {stats['source']}")
            else:
                rate = stats["n_files"] / max(stats["elapsed_s"], 1e-9)
                print(f"  -- {stats['n_ok']}/{stats['n_files']} ok in {stats['elapsed_s']:.1f}s ({rate:.0f} f/s)")

    except KeyboardInterrupt:
        print("\nInterrupted — partial results may have been written to reprocess_output/.",
              file=sys.stderr)
        if all_stats:
            grand_elapsed = time.time() - grand_t0
            write_timing_summary(all_stats, grand_elapsed)
            print(f"Partial timings written to {OUTPUT_DIR / 'run_timings.csv'}.", file=sys.stderr)
        return 130

    grand_elapsed = time.time() - grand_t0
    write_pixel_scales(all_stats)
    write_timing_summary(all_stats, grand_elapsed)
    _collect_summaries(root)
    if make_plots:
        _collect_reprocess_summaries(root)
    print(f"\nDone. Total wall time: {grand_elapsed / 60:.1f} min.")
    print(f"pixel_scales.csv updated at {PIXEL_SCALES_PATH}.")
    print(f"Output folder: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
