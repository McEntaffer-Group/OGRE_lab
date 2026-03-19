#!/usr/bin/env python
# coding: utf-8
"""
Streaming version of dot_movie:

- Reads FITS frames one-by-one from disk (no giant data list in memory)
- Computes Gaussian fits to collapsed profiles (x, y)
- Filters bad fits and computes centroid drift, FWHM, FFT
- Writes plots and a summary CSV
- Optionally builds an animation (movie) from the FITS sequence

Folder/layout matches the existing pipeline, e.g.:

rootpath / "20260306_data" / "springgenie" / "springgenie_fits"  (FITS)
rootpath / "20260306_data" / "springgenie" / "springgenie.txt"   (log)

This is compatible with genieshots_converter's output layout.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.optimize import curve_fit
import os
import glob
from pylab import *
from math import e
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from scipy.ndimage import rotate
from PIL import Image
import pandas as pd
import pathlib
from IPython.display import HTML
from datetime import datetime


# ------------------------
# Low-level helpers
# ------------------------

def openfits(path: Path) -> np.ndarray:
    """
    Open a FITS file, apply the same flips as the original dot_movie code,
    and return a 2D numpy array.

    We don't keep the HDU object around; this is streaming-friendly.
    """
    with fits.open(path) as hdu:
        data = hdu[0].data
        # Original code flips both axes (equivalent to 180 deg rotation)
        data = np.flip(data, axis=(0, 1))
        # Explicit copy to avoid lazy IO surprises
        return data.copy()


def gaussian(x, amp, mu, sigma, offset):
    """1D Gaussian with constant offset."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


@dataclass
class FitResults:
    amps_x: np.ndarray
    mus_x: np.ndarray
    sigmas_x: np.ndarray
    offsets_x: np.ndarray

    amps_y: np.ndarray
    mus_y: np.ndarray
    sigmas_y: np.ndarray
    offsets_y: np.ndarray

    min_x: float
    max_x: float
    min_y: float
    max_y: float


# ------------------------
# Core analysis
# ------------------------

def analyze_fits_sequence(fits_list):
    """
    Streaming pass over all FITS files:
    - For each frame, collapse along x and y to get 1D profiles
    - Fit Gaussians in x and y
    - Track global min/max of profiles (for consistent plot scaling)

    Returns a FitResults dataclass.
    """
    amps_x, mus_x, sigmas_x, offsets_x = [], [], [], []
    amps_y, mus_y, sigmas_y, offsets_y = [], [], [], []

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    for idx, fits_file in enumerate(fits_list):
        img = openfits(Path(fits_file))

        # Collapse profiles
        profile_x = np.sum(img, axis=0)  # along y
        profile_y = np.sum(img, axis=1)  # along x

        # Update global min/max for plots
        min_x = min(min_x, profile_x.min())
        max_x = max(max_x, profile_x.max())
        min_y = min(min_y, profile_y.min())
        max_y = max(max_y, profile_y.max())

        # --- Fit X ---
        x_vals = np.arange(profile_x.size)
        p0_x = [profile_x.max(), profile_x.argmax(), 5.0, np.median(profile_x)]
        try:
            popt_x, _ = curve_fit(gaussian, x_vals, profile_x, p0=p0_x)
        except RuntimeError:
            popt_x = [np.nan] * 4

        amps_x.append(popt_x[0])
        mus_x.append(popt_x[1])
        sigmas_x.append(popt_x[2])
        offsets_x.append(popt_x[3])

        # --- Fit Y ---
        y_vals = np.arange(profile_y.size)
        p0_y = [profile_y.max(), profile_y.argmax(), 5.0, np.median(profile_y)]
        try:
            popt_y, _ = curve_fit(gaussian, y_vals, profile_y, p0=p0_y)
        except RuntimeError:
            popt_y = [np.nan] * 4

        amps_y.append(popt_y[0])
        mus_y.append(popt_y[1])
        sigmas_y.append(popt_y[2])
        offsets_y.append(popt_y[3])

    return FitResults(
        amps_x=np.array(amps_x),
        mus_x=np.array(mus_x),
        sigmas_x=np.array(sigmas_x),
        offsets_x=np.array(offsets_x),
        amps_y=np.array(amps_y),
        mus_y=np.array(mus_y),
        sigmas_y=np.array(sigmas_y),
        offsets_y=np.array(offsets_y),
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )


def filter_fits(amps_x, mus_x, sigmas_x, offsets_x,
                amps_y, mus_y, sigmas_y, offsets_y,
                fwhm_min, fwhm_max):
    """
    Same logic as your original filter_fits: remove NaNs/infs and
    out-of-range FWHM fits.
    """
    fwhm_x = 2.355 * sigmas_x
    fwhm_y = 2.355 * sigmas_y

    finite_mask = (
        np.isfinite(amps_x) & np.isfinite(mus_x) & np.isfinite(sigmas_x) & np.isfinite(offsets_x) &
        np.isfinite(amps_y) & np.isfinite(mus_y) & np.isfinite(sigmas_y) & np.isfinite(offsets_y)
    )

    fwhm_mask = (
        (fwhm_x > fwhm_min) & (fwhm_x < fwhm_max) &
        (fwhm_y > fwhm_min) & (fwhm_y < fwhm_max)
    )

    mask = finite_mask & fwhm_mask

    return (
        amps_x[mask], mus_x[mask], sigmas_x[mask], offsets_x[mask],
        amps_y[mask], mus_y[mask], sigmas_y[mask], offsets_y[mask],
        fwhm_x[mask], fwhm_y[mask], mask
    )


def compute_relative_centroids(mus_x_f, mus_y_f):
    """Recenter to the first frame."""
    mu_x_rel = mus_x_f - mus_x_f[0]
    mu_y_rel = mus_y_f - mus_y_f[0]
    frames = np.arange(len(mu_x_rel))
    return frames, mu_x_rel, mu_y_rel


def compute_fwhm(sigmas_x_f, sigmas_y_f):
    """Compute FWHM from sigma (Gaussian)."""
    FWHM_factor = 2 * np.sqrt(2 * np.log(2))
    FWHM_x = sigmas_x_f * FWHM_factor
    FWHM_y = sigmas_y_f * FWHM_factor
    frames = np.arange(len(FWHM_x))
    return frames, FWHM_x, FWHM_y


def compute_fft(signal, dt):
    """
    FFT of a 1D signal with sampling interval dt (seconds between frames).
    Returns positive frequencies and squared amplitude (power).
    """
    N = len(signal)
    fft_vals = np.fft.fft(signal - np.mean(signal))  # remove DC offset
    freqs = np.fft.fftfreq(N, d=dt)
    mask = freqs > 0
    return freqs[mask], np.abs(fft_vals[mask]) ** 2


# ------------------------
# Plotting & animation
# ------------------------

def plot_centroid_drift(frames, mu_x_rel, mu_y_rel, out_path: Path):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # X
    axs[0].plot(frames, mu_x_rel, color='blue')
    axs[0].set_ylabel("X centroid shift (pixels)")
    axs[0].grid(True)
    linex = axs[0].plot(
        [frames[0], frames[-1]],
        [mu_x_rel[0], mu_x_rel[-1]],
        color='red'
    )[0]
    slopex = (mu_x_rel[-1] - mu_x_rel[0]) / (frames[-1] - frames[0])
    axs[0].legend([linex], [f"Slope = {slopex:.4f} px/frame"])

    # Y
    axs[1].plot(frames, mu_y_rel, color='green')
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Y centroid shift (pixels)")
    axs[1].grid(True)
    liney = axs[1].plot(
        [frames[0], frames[-1]],
        [mu_y_rel[0], mu_y_rel[-1]],
        color='red', alpha=0.5
    )[0]
    slopey = (mu_y_rel[-1] - mu_y_rel[0]) / (frames[-1] - frames[0])
    axs[1].legend([liney], [f"Slope = {slopey:.4f} px/frame"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fwhm(frames, FWHM_x, FWHM_y, out_path: Path):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # X FWHM
    axs[0].plot(frames, FWHM_x, color='blue')
    axs[0].set_ylabel("FWHM X (pixels)")
    axs[0].grid(True)
    linex = axs[0].plot(
        [frames[0], frames[-1]],
        [FWHM_x[0], FWHM_x[-1]],
        color='red', alpha=0.5
    )[0]
    slopex = (FWHM_x[-1] - FWHM_x[0]) / (frames[-1] - frames[0])
    axs[0].legend([linex], [f"Slope = {slopex:.4f} px/frame"])

    # Y FWHM
    axs[1].plot(frames, FWHM_y, color='green')
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("FWHM Y (pixels)")
    axs[1].grid(True)
    liney = axs[1].plot(
        [frames[0], frames[-1]],
        [FWHM_y[0], FWHM_y[-1]],
        color='red', alpha=0.5
    )[0]
    slopey = (FWHM_y[-1] - FWHM_y[0]) / (frames[-1] - frames[0])
    axs[1].legend([liney], [f"Slope = {slopey:.4f} px/frame"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fft(mu_x_rel, mu_y_rel, FWHM_x, FWHM_y, dt, out_path: Path):
    freqs_x, power_mu_x = compute_fft(mu_x_rel, dt)
    freqs_y, power_mu_y = compute_fft(mu_y_rel, dt)
    freqs_fx, power_fwhm_x = compute_fft(FWHM_x, dt)
    freqs_fy, power_fwhm_y = compute_fft(FWHM_y, dt)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(freqs_x, power_mu_x)
    axs[0, 0].set_title("FFT of centroid (relative x)")
    axs[0, 0].set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel("Power")

    axs[0, 1].plot(freqs_y, power_mu_y)
    axs[0, 1].set_title("FFT of centroid (relative y)")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Power")

    axs[1, 0].plot(freqs_fx, power_fwhm_x)
    axs[1, 0].set_title("FFT of FWHM (x)")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Power")

    axs[1, 1].plot(freqs_fy, power_fwhm_y)
    axs[1, 1].set_title("FFT of FWHM (y)")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Power")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_animation(fits_list, df_times, profile_limits, out_path: Path,
                   vmin=0, vmax=100):
    """
    Optional animation: re-opens each FITS on demand.
    This is still streaming (no big in-memory stack), but does
    one disk read per frame while building the movie.
    """
    min_x, max_x, min_y, max_y = profile_limits

    # Load first frame once to initialize figure
    img0 = openfits(Path(fits_list[0]))

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1],
                          height_ratios=[1, 4], wspace=0.0, hspace=0.0)
    ax_img = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_img)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_img)
    fig.add_subplot(gs[0, 1]).axis("off")

    im = ax_img.imshow(img0, cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax, aspect="auto")

    profile_x0 = np.sum(img0, axis=0)
    profile_y0 = np.sum(img0, axis=1)
    line_x, = ax_x.plot(np.arange(img0.shape[1]), profile_x0)
    line_y, = ax_y.plot(profile_y0, np.arange(img0.shape[0]))

    ax_x.set_xlim(ax_img.get_xlim())
    ax_y.set_ylim(ax_img.get_ylim())
    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)
    ax_x.set_ylabel("counts")
    ax_y.set_xlabel("counts")
    ax_x.grid()
    ax_y.grid()

    # Use global limits from the analysis pass
    ax_x.set_ylim(min_x - 200, max_x - 200)
    ax_y.set_xlim(min_y - 200, max_y)

    timestamp_text = ax_img.text(
        0.02, 0.98, "",
        transform=ax_img.transAxes,
        color="white",
        fontsize=12,
        va="top",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3)
    )

    def update(frame):
        img = openfits(Path(fits_list[frame]))
        im.set_array(img)

        profile_x = np.sum(img, axis=0)
        profile_y = np.sum(img, axis=1)
        line_x.set_ydata(profile_x)
        line_y.set_xdata(profile_y)

        if frame < len(df_times):
            timestamp = df_times.iloc[frame]["Creation Date"]
            timestamp_text.set_text(f"{timestamp}")
        else:
            timestamp_text.set_text("")

        return [im, line_x, line_y, timestamp_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(fits_list), interval=200, blit=True
    )

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='dot_movie_streaming'), bitrate=1800)
    ani.save(out_path, writer=writer)
    plt.close(fig)


# ------------------------
# IO / orchestration
# ------------------------

def get_fits_list(rootpath: Path, foldername_date: str, foldername: str):
    """
    Builds the same fits_list as the original dot_movie:

    rootpath / (foldername_date + "_data") / foldername / (foldername + "_fits")/*.fits
    """
    folderpath_date = rootpath / (foldername_date + "_data")
    filepath = folderpath_date / foldername
    fits_dir = filepath / f"{foldername}_fits"
    pattern = str(fits_dir / "*.fits")

    fits_list = sorted(Path().glob(pattern))  # pattern is absolute string
    if not fits_list:
        raise FileNotFoundError(f"No FITS found for pattern: {pattern}")

    return folderpath_date, filepath, [str(p) for p in fits_list]


def load_timestamps(log_path: Path) -> pd.DataFrame:
    """
    Load the timestamp log produced by bmp_to_fits / genieshots_converter.
    Matches the parsing you used in the notebook: tab-separated, skip header lines.
    """
    df_times = pd.read_csv(
        log_path,
        sep=r'\t+',
        engine='python',
        skiprows=2,
        names=["Filename", "Creation Date"]
    )
    df_times["Filename"] = df_times["Filename"].str.strip()
    df_times["Creation Date"] = df_times["Creation Date"].str.strip()
    return df_times


def summarize_run(filepath: Path, fits_list, notes: str,
                  mu_x_rel, mu_y_rel, FWHM_x, FWHM_y,
                  frame_period_s: float, pixel_scale: float):
    """
    Build the summary dict and write a CSV in the same style as the original code.
    """
    filename = filepath.name
    num_data = len(fits_list)

    creation_date_start = str(Path(fits_list[0]).name[-22:-5])
    creation_date_stop = str(Path(fits_list[-1]).name[-22:-5])

    # In original script this column was 'frame rate'; here we store
    # the *period* as seconds per frame, but you could store 1/period instead.
    frame_rate = 1.0 / frame_period_s

    new_row = {
        "filename": filename,
        "number of frames": num_data,
        "start time": creation_date_start,
        "stop time": creation_date_stop,
        "notes": notes,
        "frame rate": frame_rate,
        "x position": float(np.mean(mu_x_rel * pixel_scale)),
        "x position std": float(np.std(mu_x_rel * pixel_scale)),
        "y position": float(np.mean(mu_y_rel * pixel_scale)),
        "y position std": float(np.std(mu_y_rel * pixel_scale)),
        "FWHM x": float(np.mean(FWHM_x * pixel_scale)),
        "FWHM x std": float(np.std(FWHM_x * pixel_scale)),
        "FWHM y": float(np.mean(FWHM_y * pixel_scale)),
        "FWHM y std": float(np.std(FWHM_y * pixel_scale)),
    }

    out_csv = filepath / f"{filename}_summary.csv"
    pd.DataFrame([new_row]).to_csv(out_csv, index=False)
    print(f"Summary written to: {out_csv}")


def process_run(rootpath: Path,
                foldername_date: str,
                foldername: str,
                notes: str,
                frame_period_s: float,
                pixel_scale: float = 0.15,
                make_animation_flag: bool = False):
    """
    High-level orchestrator for a single run.

    Parameters
    ----------
    rootpath : Path
        Base directory (e.g. D:/Reverse Telescope Test).
    foldername_date : str
        Date string like "20260306".
    foldername : str
        Run name like "springgenie".
    notes : str
        Free-form notes about the dataset.
    frame_period_s : float
        Time between frames in seconds (dt).
    pixel_scale : float
        Spatial scale per pixel (e.g., arcsec/pixel or mm/pixel).
    make_animation_flag : bool
        If True, also build an mp4 movie.
    """
    # Locate files
    folderpath_date, filepath, fits_list = get_fits_list(
        rootpath, foldername_date, foldername
    )

    print(f"Found {len(fits_list)} FITS files in {filepath}")

    # Load timestamp log (same location/naming as original code)
    log_path = filepath / f"{foldername}.txt"
    df_times = load_timestamps(log_path)

    # Streaming analysis
    print("Analyzing FITS sequence (streaming)...")
    fit_results = analyze_fits_sequence(fits_list)

    # Filter fits
    (amps_x_f, mus_x_f, sigmas_x_f, offsets_x_f,
     amps_y_f, mus_y_f, sigmas_y_f, offsets_y_f,
     fwhm_x_f, fwhm_y_f, mask) = filter_fits(
        fit_results.amps_x, fit_results.mus_x, fit_results.sigmas_x, fit_results.offsets_x,
        fit_results.amps_y, fit_results.mus_y, fit_results.sigmas_y, fit_results.offsets_y,
        fwhm_min=1, fwhm_max=1000
    )

    # Centroid drift
    frames_pos, mu_x_rel, mu_y_rel = compute_relative_centroids(mus_x_f, mus_y_f)

    # FWHM
    frames_fwhm, FWHM_x, FWHM_y = compute_fwhm(sigmas_x_f, sigmas_y_f)

    # Plots
    pos_plot_path = folderpath_date / foldername / f"{foldername}_position.png"
    fwhm_plot_path = folderpath_date / foldername / f"{foldername}_FWHM.png"
    fft_plot_path = folderpath_date / foldername / f"{foldername}_FFT.png"

    plot_centroid_drift(frames_pos, mu_x_rel, mu_y_rel, pos_plot_path)
    plot_fwhm(frames_fwhm, FWHM_x, FWHM_y, fwhm_plot_path)
    plot_fft(mu_x_rel, mu_y_rel, FWHM_x, FWHM_y,
             dt=frame_period_s, out_path=fft_plot_path)

    # Summary CSV (in same folder as original)
    summarize_run(filepath, fits_list, notes,
                  mu_x_rel, mu_y_rel, FWHM_x, FWHM_y,
                  frame_period_s=frame_period_s,
                  pixel_scale=pixel_scale)

    print("Analysis complete.")

    # Optional animation
    if make_animation_flag:
        print("Building animation (this may take a bit; disk IO-bound)...")
        movie_path = folderpath_date / foldername / f"{foldername}.mp4"
        profile_limits = (
            fit_results.min_x, fit_results.max_x,
            fit_results.min_y, fit_results.max_y
        )
        make_animation(fits_list, df_times, profile_limits, movie_path)
        print(f"Animation saved to: {movie_path}")


# ------------------------
# Main entry point
# ------------------------

if __name__ == "__main__":


    # # change to your path
    # filepath = rootpath / (foldername_date + "_data") / foldername
    # folderpath_date = rootpath / (foldername_date + "_data")
    # path1 = (filepath / (foldername + "_fits")).__str__() + "/*.fits"
    #
    # filepath = rootpath / foldername_date / foldername
    # path1 = filepath.__str__() + "/*.bmp"
    #
    # bmp_list = glob.glob(path1)
    # bmp_list = np.sort(bmp_list, kind='standardsort')

    # EDIT THESE FOR EACH RUN
    rootpath = Path(r"D:\Reverse Telescope Test")
    foldername_date = "20260306"
    foldername = "springgenie"
    notes = "spring break data"

    # Frame timing:
    # - If your camera is 52.37 Hz -> frame_period_s = 1.0 / 52.37
    # - If you're taking one frame every 60 s -> frame_period_s = 60.0
    frame_period_s = 60.0

    # Spatial scale (whatever units you want in the summary)
    pixel_scale = 0.15  # e.g. 0.15 arcsec/pixel

    # Toggle animation here:
    make_animation_flag = True  # or False

    process_run(
        rootpath=rootpath,
        foldername_date=foldername_date,
        foldername=foldername,
        notes=notes,
        frame_period_s=frame_period_s,
        pixel_scale=pixel_scale,
        make_animation_flag=make_animation_flag,
    )