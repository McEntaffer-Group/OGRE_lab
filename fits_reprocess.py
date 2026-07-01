"""
fits_reprocess.py

Walk E:/Reverse Telescope Test Data, locate every {date}_data/{runname}/{runname}_fits/
folder, and for each one write {runname}_frames.csv with per-frame dot data:

    frame_num, filename, timestamp,
    mu_x, mu_y, sigma_x, sigma_y, fwhm_x, fwhm_y,
    amp_x, amp_y, offset_x, offset_y,
    fit_ok, n_peaks_x, n_peaks_y

All spatial values are in PIXELS. No arcsec conversion happens here; downstream code
looks up per-run pixel scale in pixel_scales.csv when needed.

Behavior:
  * Skip runs whose _frames.csv already exists (resumable).
  * Parallel Gaussian fits via ThreadPoolExecutor (USB-read latency is the bottleneck).
  * Multi-dot detection: scipy.signal.find_peaks on each 1D profile, count is stored
    per frame so phase-4 plotting can filter `(n_peaks_x == 1) & (n_peaks_y == 1)`.
  * Builds pixel_scales.csv at repo root, mapping each run to its camera type
    (detected from image dimensions) so the user can fill in real scale values later.

Run from the repo root:
    python fits_reprocess.py                       # all runs
    python fits_reprocess.py --run zoeysecondary   # one run by name
    python fits_reprocess.py --dry-run             # list work, don't process
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

ROOT = Path("E:/Reverse Telescope Test Data")
PIXEL_SCALES_PATH = Path(__file__).parent / "pixel_scales.csv"

WORKERS = 16

# Multi-dot detection: a "peak" must be at least PEAK_HEIGHT_FRAC of the profile max
# (so a second peak has to be a real comparable spike, not noise) and at least
# PEAK_MIN_DISTANCE_PX away from any other peak.
PEAK_HEIGHT_FRAC = 0.50
PEAK_MIN_DISTANCE_PX = 50

# Reject obviously bad fits. Bumped above the 500px timelapse default because the Genie
# sensor (1936x1216) admits wider point spreads than the main camera (1280x1024).
FWHM_MIN_PX = 1.0
FWHM_MAX_PX = 1000.0

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548; sigma -> FWHM

# Filenames look like:
#   main:   zoeysecondary0001 26-04-20 15-16-47.fits
#   genie:  zoeysecondarygenie_0001 26-04-20 15-16-01.fits
# Capture trailing " YY-MM-DD HH-MM-SS.fits"; the frame number is whatever comes
# before that, suffixed to the runname (with or without an underscore separator).
_TS_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})\.fits$", re.IGNORECASE)
_FRAMENUM_RE = re.compile(r"_?(\d{3,6})\s+\d{2}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}\.fits$", re.IGNORECASE)

CAMERA_BY_SHAPE = {
    (1024, 1280): "main_camera",
    (1216, 1936): "dalsa_genie",
}


# ---------------------------------------------------------------------------
# Fitting primitives
# ---------------------------------------------------------------------------

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def _fit_one_profile(profile):
    """Fit a single Gaussian to a 1D profile. Returns (amp, mu, sigma, offset) or NaNs."""
    try:
        x = np.arange(profile.size)
        popt, _ = curve_fit(
            gaussian, x, profile,
            p0=[profile.max(), float(profile.argmax()), 50.0, float(np.median(profile))],
            maxfev=3000,
        )
        amp, mu, sigma, offset = popt
        return amp, mu, abs(sigma), offset
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _count_peaks(profile):
    """Count comparable bright peaks in a 1D profile. A peak qualifies only if it is
    >= PEAK_HEIGHT_FRAC of the profile maximum and >= PEAK_MIN_DISTANCE_PX from any
    other peak. This is biased toward returning 1 for normal single-dot frames; we
    only want to flag a frame as multi-dot when there genuinely are two comparably
    bright peaks well separated in the projection."""
    pmax = float(profile.max())
    pmin = float(profile.min())
    if not np.isfinite(pmax) or pmax <= pmin:
        return 0
    # Height threshold measured above the baseline (min of the profile), so a wide
    # bright plateau still counts as one peak rather than zero.
    height = pmin + PEAK_HEIGHT_FRAC * (pmax - pmin)
    peaks, _ = find_peaks(profile, height=height, distance=PEAK_MIN_DISTANCE_PX)
    return int(len(peaks))


def fit_frame(path: str) -> dict:
    """Process one FITS file. Returns a dict ready to write as a CSV row."""
    out = {
        "filename": os.path.basename(path),
        "frame_num": _extract_frame_num(path),
        "timestamp": _extract_timestamp(path),
        "mu_x": np.nan, "mu_y": np.nan,
        "sigma_x": np.nan, "sigma_y": np.nan,
        "fwhm_x": np.nan, "fwhm_y": np.nan,
        "amp_x": np.nan, "amp_y": np.nan,
        "offset_x": np.nan, "offset_y": np.nan,
        "fit_ok": False,
        "n_peaks_x": 0, "n_peaks_y": 0,
    }
    try:
        with fits.open(path) as hdul:
            img = np.flip(hdul[0].data, axis=(0, 1)).astype(np.float64)
        px_profile = np.sum(img, axis=0)
        py_profile = np.sum(img, axis=1)
    except Exception:
        return out

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
    # 2-digit year -> 2000s. Lab data all post-2020 so safe.
    return f"20{yy}-{mo}-{dd} {hh}:{mm}:{ss}"


def _extract_frame_num(path: str) -> int:
    name = os.path.basename(path)
    m = _FRAMENUM_RE.search(name)
    return int(m.group(1)) if m else -1


def _detect_camera(fits_dir: Path) -> tuple[str, str]:
    """Return (camera_label, shape_str) from the first FITS in the directory."""
    sample = next(iter(sorted(fits_dir.glob("*.fits"))), None)
    if sample is None:
        return "no_fits", ""
    try:
        with fits.open(sample) as hdul:
            shape = hdul[0].data.shape
    except Exception as e:
        return f"unreadable ({e})", ""
    shape_str = f"{shape[0]}x{shape[1]}"
    return CAMERA_BY_SHAPE.get(tuple(shape), "unknown"), shape_str


# ---------------------------------------------------------------------------
# Run discovery and execution
# ---------------------------------------------------------------------------

def _discover_runs(root: Path) -> list[Path]:
    """Find every run folder that contains a *_fits subfolder."""
    runs = []
    for data_dir in sorted(root.glob("*_data")):
        if not data_dir.is_dir():
            continue
        for run_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            fits_dir = run_dir / f"{run_dir.name}_fits"
            if fits_dir.is_dir():
                runs.append(run_dir)
    return runs


def process_run(run_dir: Path, force: bool = False) -> dict:
    """Process one run folder. Returns a stats dict."""
    runname = run_dir.name
    fits_dir = run_dir / f"{runname}_fits"
    out_csv = run_dir / f"{runname}_frames.csv"

    stats = {
        "runname": runname,
        "fits_dir": str(fits_dir),
        "out_csv": str(out_csv),
        "n_files": 0,
        "n_ok": 0,
        "elapsed_s": 0.0,
        "skipped": False,
        "camera": "",
        "shape": "",
    }

    camera, shape_str = _detect_camera(fits_dir)
    stats["camera"] = camera
    stats["shape"] = shape_str

    if out_csv.exists() and not force:
        stats["skipped"] = True
        return stats

    files = sorted(glob.glob(str(fits_dir / "*.fits")))
    stats["n_files"] = len(files)
    if not files:
        return stats

    t0 = time.time()
    results = [None] * len(files)
    print(f"  -> {runname}: {len(files)} files, camera={camera} ({shape_str})")
    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(fit_frame, f): i for i, f in enumerate(files)}
        for n_done, fut in enumerate(as_completed(futures)):
            results[futures[fut]] = fut.result()
            if n_done and n_done % 1000 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta = (len(files) - n_done) / max(rate, 1e-9)
                print(f"    {n_done}/{len(files)}  ({rate:.0f} files/s, eta {eta/60:.1f} min)")

    df = pd.DataFrame(results)
    df = df.sort_values("frame_num").reset_index(drop=True)
    cols = [
        "frame_num", "filename", "timestamp",
        "mu_x", "mu_y", "sigma_x", "sigma_y", "fwhm_x", "fwhm_y",
        "amp_x", "amp_y", "offset_x", "offset_y",
        "fit_ok", "n_peaks_x", "n_peaks_y",
    ]
    df = df[cols]
    df.to_csv(out_csv, index=False)

    stats["n_ok"] = int(df["fit_ok"].sum())
    stats["elapsed_s"] = time.time() - t0
    return stats


def write_pixel_scales(stats_list: list[dict]) -> None:
    """Write/update pixel_scales.csv. Preserves any existing pixel_scale_arcsec_per_pixel
    values the user has filled in by hand."""
    existing = {}
    if PIXEL_SCALES_PATH.exists():
        try:
            with open(PIXEL_SCALES_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    existing[row["runname"]] = row
        except Exception as e:
            print(f"  warning: could not read existing pixel_scales.csv: {e}")

    fieldnames = ["runname", "camera", "image_shape", "pixel_scale_arcsec_per_pixel", "notes"]
    with open(PIXEL_SCALES_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Merge: keep user-filled scale/notes, overwrite camera/shape from fresh detection.
        for s in stats_list:
            prev = existing.get(s["runname"], {})
            w.writerow({
                "runname": s["runname"],
                "camera": s["camera"],
                "image_shape": s["shape"],
                "pixel_scale_arcsec_per_pixel": prev.get("pixel_scale_arcsec_per_pixel", ""),
                "notes": prev.get("notes", ""),
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--root", default=str(ROOT), help="data root (default: %(default)s)")
    parser.add_argument("--run", help="process only this run name (substring match)")
    parser.add_argument("--force", action="store_true", help="reprocess runs even if _frames.csv exists")
    parser.add_argument("--dry-run", action="store_true", help="list discovered work, don't process")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    runs = _discover_runs(root)
    if args.run:
        runs = [r for r in runs if args.run in r.name]
        if not runs:
            print(f"No runs match '{args.run}'.", file=sys.stderr)
            return 1

    print(f"Discovered {len(runs)} run(s) under {root}.")
    if args.dry_run:
        for r in runs:
            existing = "EXISTS" if (r / f"{r.name}_frames.csv").exists() else "pending"
            print(f"  [{existing}] {r}")
        return 0

    all_stats = []
    grand_t0 = time.time()
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.parent.name}/{run.name}")
        stats = process_run(run, force=args.force)
        all_stats.append(stats)
        if stats["skipped"]:
            print(f"  -- skipped (already processed)")
        elif stats["n_files"] == 0:
            print(f"  -- no FITS files found in {stats['fits_dir']}")
        else:
            print(
                f"  -- {stats['n_ok']}/{stats['n_files']} ok in {stats['elapsed_s']:.1f}s "
                f"({stats['n_files']/max(stats['elapsed_s'],1e-9):.0f} f/s) -> {stats['out_csv']}"
            )

    write_pixel_scales(all_stats)
    print(f"\nDone. Total wall time: {(time.time()-grand_t0)/60:.1f} min.")
    print(f"pixel_scales.csv updated at {PIXEL_SCALES_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
