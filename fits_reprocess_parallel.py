"""
fits_reprocess_parallel.py

Same output as fits_reprocess.py — {runname}_frames.csv, plots, summary, movie —
but restructured so fitting and movie generation share a single pass over each
image. Each worker returns (fit_row, image, px_sum, py_sum); the main process
consumes futures in submission order, writes the CSV row list, and feeds the
same image straight to the movie writer via matplotlib's grab_frame() streaming
API. No frame is loaded from disk twice.

Trade-offs vs fits_reprocess.py:
  + Skips the second disk-read pass the sequential pipeline did during movie
    generation (~4 min saved for a 5000-frame main-camera run).
  + Fitting and movie rasterization overlap, so total wall time approaches
    max(fit_time, matplotlib_time) rather than their sum.
  + Sum profiles computed in the worker are reused, avoiding an extra np.sum
    per frame in the movie loop.
  - Workers pickle the raw image back to the main process each frame. For
    uint8 480x640 frames that's ~300 KB per shot; for uint16 1920x2560 FITS
    that's ~10 MB per shot. IPC overhead is real but usually still faster
    than the disk read it replaces.
  - Order-preserving consumption means one slow worker throttles the whole
    pipeline temporarily; other workers keep producing but their outputs sit
    in the executor queue.

Memory guardrail:
  --max-in-flight caps how many frames can be in the ProcessPool pipeline at
  once (default 2 * cpu_count). Combined with the executor's internal queue,
  that's the maximum image count the process holds simultaneously. For 16
  workers with 20 MB frames that caps overhead at ~640 MB, so 10k-frame runs
  at 1920x2560 don't OOM.

Two pipeline modes; --pipeline chooses:
  shmem      Pre-allocates a (N, H, W) numpy memmap file (in OUTPUT_DIR)
             at the raw pixel dtype. Workers write image bytes directly into
             slot i via zero-copy; only the ~200-byte fit-result dict
             crosses IPC. Movie push reads the same buffer with no copy.
             Fastest when the run fits in RAM (OS page cache keeps it hot).
             File-backed rather than anonymous shared memory so it works on
             Windows without hitting the per-worker commit-charge limit
             (WinError 1450). On Linux, point TMPDIR at /dev/shm to force
             RAM-backing instead of disk.
  streaming  Workers pickle the whole image back over IPC. Slower per-frame
             but bounded to --max-in-flight images at once — the only path
             that safely handles runs larger than RAM.
  auto       (default) Compare total image bytes * 1.2 to psutil available
             memory; pick shmem when it fits, streaming otherwise.

Movie frame selection:
  --video-fps N        default 20 (dot_movie-Copy3 parity). Use 24 to hit
                       exactly 60 s of movie per real day at 1-frame-per-min.
  --seconds-per-day X  when set, decimates the MOVIE only (CSV keeps every
                       fit) so the video runs at ~X seconds per real day.
                       Formula matches timelapse_dot_movie.ipynb:
                       stride = round(real_frames_per_day / (fps * X)).
                       real_frames_per_day is auto-derived from the median
                       Δt between filenames' timestamps.

CLI mirrors fits_reprocess.py plus pipeline knobs:
    python fits_reprocess_parallel.py                      # all runs
    python fits_reprocess_parallel.py --run postspie       # substring match
    python fits_reprocess_parallel.py --no-movie           # fit only
    python fits_reprocess_parallel.py --max-in-flight 32
    python fits_reprocess_parallel.py --pipeline shmem     # force shmem
    python fits_reprocess_parallel.py --pipeline streaming # force streaming
    python fits_reprocess_parallel.py --video-fps 24 --seconds-per-day 15
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image

import csv_to_dotplots

# Reuse everything portable from the sequential module.
import fits_reprocess as fr
from fits_reprocess import (
    ROOT, PIXEL_SCALES_PATH, OUTPUT_DIR, WORKERS,
    MOVIE_FPS, MOVIE_IMSHOW_VMIN,
    MOVIE_USE_NVENC,
    _worker_init,
    _fill_fit_results, _empty_row,
    _extract_timestamp, _extract_frame_num,
    _extract_timestamp_image, _extract_frame_num_image,
    _detect_camera, _write_frames_csv, _mirror_for_run,
    list_run_images, run_key, output_dir_for,
    _discover_fits_runs, _discover_image_runs,
    _collect_summaries, _collect_reprocess_summaries,
    write_timing_summary, write_pixel_scales,
    _load_fits_frame, _load_image_frame,
    _sample_movie_stats,
    _generate_plots_and_summary,
)


DEFAULT_MAX_IN_FLIGHT = max(2 * WORKERS, 8)
DEFAULT_VIDEO_FPS = 20            # matches dot_movie-Copy3
DEFAULT_SECONDS_PER_DAY = None    # None = movie keeps every frame
# Safety headroom when deciding whether the run's image set fits in RAM.
SHMEM_HEADROOM_MULT = 1.2


# ---------------------------------------------------------------------------
# Streaming workers
# ---------------------------------------------------------------------------

def _fit_worker_fits(path: str, need_image: bool):
    """Fit one FITS frame. Returns (row_dict, image, px_sum, py_sum).
    When need_image is False, image/px_sum/py_sum are all None to avoid IPC cost."""
    out = _empty_row(
        filename=os.path.basename(path),
        frame_num=_extract_frame_num(path),
        timestamp=_extract_timestamp(path),
    )
    try:
        img = _load_fits_frame(path)
    except Exception:
        return out, None, None, None
    img_f = img.astype(np.float64)
    px = np.sum(img_f, axis=0)
    py = np.sum(img_f, axis=1)
    _fill_fit_results(out, px, py)
    if not need_image:
        return out, None, None, None
    return out, img, px.astype(np.float32), py.astype(np.float32)


def _fit_worker_image(path: Path, need_image: bool):
    """Fit one BMP/PNG frame. Returns (row_dict, image, px_sum, py_sum).
    When need_image is False, image/px_sum/py_sum are all None to avoid IPC cost."""
    frame_num = _extract_frame_num_image(path)
    ts = _extract_timestamp_image(path)
    if not ts:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = ""
    out = _empty_row(filename=path.name, frame_num=frame_num, timestamp=ts)
    try:
        img = _load_image_frame(path)
    except Exception:
        return out, None, None, None
    img_f = img.astype(np.float64)
    px = np.sum(img_f, axis=0)
    py = np.sum(img_f, axis=1)
    _fill_fit_results(out, px, py)
    if not need_image:
        return out, None, None, None
    return out, img, px.astype(np.float32), py.astype(np.float32)


# ---------------------------------------------------------------------------
# Movie writer setup (shared by FITS and IMAGE paths)
# ---------------------------------------------------------------------------

class _StreamingMovie:
    """Wraps a matplotlib composite-layout figure and an ffmpeg writer's
    saving() context so callers can push one frame at a time. Layout matches
    dot_movie-Copy3.ipynb: main image + X profile above + Y profile right +
    timestamp overlay."""

    def __init__(self, sample_paths, loader, out_path: Path,
                 fps: int = DEFAULT_VIDEO_FPS):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        self._fps = fps

        self._plt = plt
        self._animation = animation

        stats = _sample_movie_stats(loader, sample_paths)
        first = loader(sample_paths[0])

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                              wspace=0.0, hspace=0.0)
        ax_img = fig.add_subplot(gs[1, 0])
        ax_x   = fig.add_subplot(gs[0, 0], sharex=ax_img)
        ax_y   = fig.add_subplot(gs[1, 1], sharey=ax_img)
        fig.add_subplot(gs[0, 1]).axis("off")

        im = ax_img.imshow(first, cmap="viridis", origin="lower",
                           vmin=MOVIE_IMSHOW_VMIN, vmax=stats["vmax"],
                           aspect="auto")
        line_x, = ax_x.plot(np.arange(first.shape[1]), np.sum(first, axis=0))
        line_y, = ax_y.plot(np.sum(first, axis=1), np.arange(first.shape[0]))
        ax_x.set_xlim(ax_img.get_xlim())
        ax_y.set_ylim(ax_img.get_ylim())
        ax_x.tick_params(labelbottom=False)
        ax_y.tick_params(labelleft=False)
        ax_x.set_ylabel("counts")
        ax_y.set_xlabel("counts")
        ax_x.grid(True); ax_y.grid(True)
        ax_x.set_ylim(stats["px_min"], stats["px_max"])
        ax_y.set_xlim(stats["py_min"], stats["py_max"])
        ts_text = ax_img.text(0.02, 0.98, "", transform=ax_img.transAxes,
                              color="white", fontsize=12,
                              verticalalignment="top",
                              bbox=dict(facecolor="black", alpha=0.5,
                                        edgecolor="none", pad=3))

        self.fig = fig
        self.im = im
        self.line_x = line_x
        self.line_y = line_y
        self.ts_text = ts_text
        self._out_path = out_path

        Writer = animation.writers["ffmpeg"]
        codec = "h264_nvenc" if MOVIE_USE_NVENC else None
        try:
            self.writer = Writer(fps=self._fps, bitrate=1800,
                                 codec=codec) if codec else Writer(
                fps=self._fps, bitrate=1800)
            self._saving = self.writer.saving(fig, str(out_path), dpi=100)
            self._saving.__enter__()
            self._nvenc_active = codec == "h264_nvenc"
        except Exception as exc:
            print(f"    !! ffmpeg writer init failed ({exc}); retrying with libx264")
            self.writer = Writer(fps=self._fps, bitrate=1800)
            self._saving = self.writer.saving(fig, str(out_path), dpi=100)
            self._saving.__enter__()
            self._nvenc_active = False
        self._closed = False

    def push(self, image, px_sum, py_sum, ts_str: str) -> None:
        if self._closed or image is None:
            return
        self.im.set_array(image)
        if px_sum is None:
            px_sum = np.sum(image, axis=0)
        if py_sum is None:
            py_sum = np.sum(image, axis=1)
        self.line_x.set_ydata(px_sum)
        self.line_y.set_xdata(py_sum)
        if ts_str:
            self.ts_text.set_text(ts_str)
        try:
            self.writer.grab_frame()
        except Exception as exc:
            print(f"    !! grab_frame failed at this frame: {exc}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._saving.__exit__(None, None, None)
        except Exception as exc:
            print(f"    !! movie finalize failed: {exc}")
        self._plt.close(self.fig)


# ---------------------------------------------------------------------------
# Shared-memory workers (zero-copy image transfer main <-> workers)
# ---------------------------------------------------------------------------

# Per-worker cache populated by _shmem_worker_init; module-level so worker
# functions can find it after fork/spawn without paying pickle cost per call.
_SHMEM_ARR = None


def _shmem_worker_init(memmap_path: str, shape, dtype_str: str) -> None:
    """ProcessPool initializer: open the parent-created memmap file and
    expose it as a plain numpy ndarray. File-backed instead of anon shmem
    to avoid Windows commit-charge accounting blowing up with many workers
    (WinError 1450). Also silences Ctrl-C so shutdown flows through the
    parent (matches _worker_init)."""
    global _SHMEM_ARR
    _SHMEM_ARR = np.memmap(memmap_path, dtype=np.dtype(dtype_str),
                           mode="r+", shape=tuple(shape))
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _fit_worker_fits_shmem(args):
    """Fit one FITS frame and write its raw pixels into shared-memory slot
    `index`. Returns (row_dict, wrote_image_ok) — no image over IPC."""
    path, index = args
    out = _empty_row(
        filename=os.path.basename(path),
        frame_num=_extract_frame_num(path),
        timestamp=_extract_timestamp(path),
    )
    try:
        img = _load_fits_frame(path)
    except Exception:
        return out, False
    try:
        _SHMEM_ARR[index] = img.astype(_SHMEM_ARR.dtype, copy=False)
    except Exception:
        return out, False
    img_f = img.astype(np.float64)
    px = np.sum(img_f, axis=0)
    py = np.sum(img_f, axis=1)
    _fill_fit_results(out, px, py)
    return out, True


def _fit_worker_image_shmem(args):
    """Fit one BMP/PNG frame and write its raw pixels into shared-memory
    slot `index`. Returns (row_dict, wrote_image_ok) — no image over IPC."""
    path, index = args
    frame_num = _extract_frame_num_image(path)
    ts = _extract_timestamp_image(path)
    if not ts:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = ""
    out = _empty_row(filename=path.name, frame_num=frame_num, timestamp=ts)
    try:
        img = _load_image_frame(path)
    except Exception:
        return out, False
    try:
        _SHMEM_ARR[index] = img.astype(_SHMEM_ARR.dtype, copy=False)
    except Exception:
        return out, False
    img_f = img.astype(np.float64)
    px = np.sum(img_f, axis=0)
    py = np.sum(img_f, axis=1)
    _fill_fit_results(out, px, py)
    return out, True


# ---------------------------------------------------------------------------
# Auto-select + video-stride helpers
# ---------------------------------------------------------------------------

def _peek_first(loader, path):
    """Load the first frame to learn shape + dtype for shmem sizing."""
    return loader(path)


def _available_ram_bytes() -> int:
    """Best-effort available-RAM figure. Falls back to a conservative 4 GB
    if psutil isn't installed."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 4 * 1024 ** 3


def _choose_pipeline(mode: str, n_files: int, first_img: np.ndarray) -> tuple[str, int, int]:
    """Return (chosen_mode, total_bytes_needed, available_ram_bytes).
    mode may be 'auto', 'shmem', or 'streaming'; the return may downgrade
    'shmem' to 'streaming' with a printed warning if the buffer would exceed
    OS-reported available memory in 'auto' mode."""
    total = int(n_files * first_img.nbytes * SHMEM_HEADROOM_MULT)
    avail = _available_ram_bytes()
    if mode == "shmem":
        return "shmem", total, avail
    if mode == "streaming":
        return "streaming", total, avail
    # auto
    if total <= avail:
        return "shmem", total, avail
    return "streaming", total, avail


def _compute_video_stride(timestamps: list[str], video_fps: int,
                          seconds_per_day) -> int:
    """Match timelapse_dot_movie.ipynb: stride = round(real_fps_per_day /
    (video_fps * seconds_per_day)). Returns 1 (no decimation) when
    seconds_per_day is falsy or timestamps are unusable."""
    if not seconds_per_day:
        return 1
    ts_parsed = []
    for t in timestamps:
        if not t:
            continue
        try:
            ts_parsed.append(pd.to_datetime(t))
        except Exception:
            continue
    if len(ts_parsed) < 2:
        return 1
    ts_parsed.sort()
    dts = np.diff([t.timestamp() for t in ts_parsed])
    dts = dts[dts > 0]
    if dts.size == 0:
        return 1
    median_dt = float(np.median(dts))
    real_frames_per_day = 86400.0 / median_dt
    video_frames_per_day = float(video_fps) * float(seconds_per_day)
    if video_frames_per_day <= 0:
        return 1
    return max(1, round(real_frames_per_day / video_frames_per_day))


# ---------------------------------------------------------------------------
# Shared-memory per-run driver
# ---------------------------------------------------------------------------

def _shmem_run(
    runname: str,
    files: list,
    worker_fn,               # _fit_worker_fits_shmem or _fit_worker_image_shmem
    loader_frame,            # for movie-stats sampling
    timestamps: list[str],
    out_csv: Path,
    run_dir: Path,
    stats: dict,
    make_plots: bool,
    make_movie: bool,
    first_img: np.ndarray,
    video_stride: int,
    video_fps: int = DEFAULT_VIDEO_FPS,
) -> dict:
    """Zero-copy pipeline: workers write images into a file-backed memmap;
    main reads back the same file for the movie. File-backed rather than
    anon shmem so it works cross-platform without hitting Windows'
    commit-charge accounting (WinError 1450 with many workers). On Linux
    users can point TMPDIR at /dev/shm to force RAM-backing."""
    import tempfile

    if out_csv.exists():
        _mirror_for_run(out_csv, run_dir, "_frames_prev.csv")

    n_files = len(files)
    if n_files == 0:
        return stats

    shape = (n_files,) + first_img.shape
    dtype = first_img.dtype
    n_bytes = int(np.prod(shape) * dtype.itemsize)

    # Temp file goes in OUTPUT_DIR (same filesystem as everything else this
    # tool writes) so cleanup is co-located and we don't blow up C:/ on
    # Windows if the system temp dir is on a small drive.
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{runname}_shmem_",
                                        suffix=".dat", dir=OUTPUT_DIR)
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    # np.memmap in 'w+' mode allocates the file to the requested size.
    arr = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=shape)

    movie = None
    if make_movie:
        try:
            movie = _StreamingMovie(files, loader_frame,
                                    out_csv.parent / f"{runname}_reprocess.mp4",
                                    fps=video_fps)
            print(f"    movie writer ready (nvenc={movie._nvenc_active}, "
                  f"fps={video_fps}, stride={video_stride})")
        except Exception as exc:
            print(f"    !! movie setup failed ({exc}); continuing without movie")
            movie = None

    print(f"  -> {runname}: {n_files} files, camera={stats['camera']}, "
          f"shmem pipeline (memmap={n_bytes/1024**3:.2f} GB "
          f"at {tmp_path.name}, stride={video_stride})")

    t0 = time.time()
    results = [None] * n_files
    ok_flags = [False] * n_files

    try:
        with ProcessPoolExecutor(
                max_workers=WORKERS,
                initializer=_shmem_worker_init,
                initargs=(str(tmp_path), list(shape), str(dtype))) as exe:
            futures = [exe.submit(worker_fn, (path, i))
                       for i, path in enumerate(files)]

            n_done = 0
            for i, fut in enumerate(futures):
                row, ok = fut.result()
                results[i] = row
                ok_flags[i] = ok
                if movie is not None and ok and (i % video_stride == 0):
                    ts_str = timestamps[i] if timestamps and i < len(timestamps) else ""
                    movie.push(arr[i], None, None, ts_str)
                n_done += 1
                if n_done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (n_files - n_done) / max(rate, 1e-9)
                    print(f"    {n_done}/{n_files}  ({rate:.0f} f/s, "
                          f"eta {eta/60:.1f} min)")
    except KeyboardInterrupt:
        if movie is not None:
            movie.close()
        try:
            del arr
        except Exception:
            pass
        try:
            tmp_path.unlink()
        except Exception:
            pass
        raise
    finally:
        if movie is not None:
            movie.close()
        try:
            arr.flush()
        except Exception:
            pass
        try:
            del arr
        except Exception:
            pass
        try:
            tmp_path.unlink()
        except Exception as exc:
            print(f"    !! could not remove shmem temp file {tmp_path}: {exc}")

    df = (pd.DataFrame([r for r in results if r is not None])
            .sort_values("frame_num").reset_index(drop=True))
    _write_frames_csv(df, out_csv)
    _mirror_for_run(out_csv, run_dir)
    stats["n_ok"] = int(df["fit_ok"].sum())
    stats["elapsed_s"] = time.time() - t0

    if make_plots:
        _generate_plots_and_summary(out_csv, out_csv.parent, run_dir, runname)

    return stats


# ---------------------------------------------------------------------------
# Streaming per-run driver
# ---------------------------------------------------------------------------

def _stream_run(
    runname: str,
    files: list,
    worker_fn,                 # partial-bound: (path) -> (row, img, px, py)
    loader_frame,              # (path) -> np.ndarray, for movie stats sampling
    timestamps: list[str],
    out_csv: Path,
    run_dir: Path,
    stats: dict,
    make_plots: bool,
    make_movie: bool,
    max_in_flight: int,
    video_fps: int = DEFAULT_VIDEO_FPS,
    video_stride: int = 1,
) -> dict:
    """Shared streaming pipeline used by both FITS and IMAGE processing."""

    if out_csv.exists():
        _mirror_for_run(out_csv, run_dir, "_frames_prev.csv")

    n_files = len(files)
    if n_files == 0:
        return stats

    movie = None
    if make_movie:
        try:
            movie = _StreamingMovie(files, loader_frame,
                                    out_csv.parent / f"{runname}_reprocess.mp4",
                                    fps=video_fps)
            print(f"    movie writer ready (nvenc={movie._nvenc_active}, "
                  f"fps={video_fps}, stride={video_stride})")
        except Exception as exc:
            print(f"    !! movie setup failed ({exc}); continuing without movie")
            movie = None

    print(f"  -> {runname}: {n_files} files, camera={stats['camera']}, "
          f"streaming (max_in_flight={max_in_flight})")

    t0 = time.time()
    results = [None] * n_files
    submit_iter = iter(enumerate(files))
    pending: deque = deque()

    try:
        with ProcessPoolExecutor(max_workers=WORKERS,
                                 initializer=_worker_init) as exe:
            # Prime the pipeline.
            for _ in range(max_in_flight):
                try:
                    i, path = next(submit_iter)
                except StopIteration:
                    break
                pending.append((i, exe.submit(worker_fn, path)))

            n_done = 0
            while pending:
                i, fut = pending.popleft()
                row, img, px_sum, py_sum = fut.result()
                results[i] = row
                if movie is not None and (i % video_stride == 0):
                    ts_str = timestamps[i] if timestamps and i < len(timestamps) else ""
                    movie.push(img, px_sum, py_sum, ts_str)
                n_done += 1
                if n_done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (n_files - n_done) / max(rate, 1e-9)
                    print(f"    {n_done}/{n_files}  ({rate:.0f} f/s, "
                          f"eta {eta/60:.1f} min)")
                # Refill.
                try:
                    j, path = next(submit_iter)
                    pending.append((j, exe.submit(worker_fn, path)))
                except StopIteration:
                    pass
    except KeyboardInterrupt:
        for _, f in pending:
            f.cancel()
        if movie is not None:
            movie.close()
        raise
    finally:
        if movie is not None:
            movie.close()

    df = (pd.DataFrame([r for r in results if r is not None])
            .sort_values("frame_num").reset_index(drop=True))
    _write_frames_csv(df, out_csv)
    _mirror_for_run(out_csv, run_dir)
    stats["n_ok"] = int(df["fit_ok"].sum())
    stats["elapsed_s"] = time.time() - t0

    if make_plots:
        _generate_plots_and_summary(out_csv, out_csv.parent, run_dir, runname)

    return stats


# ---------------------------------------------------------------------------
# Per-run entry points
# ---------------------------------------------------------------------------

def process_fits_run(run_dir: Path, make_plots: bool = True,
                     make_movie: bool = True,
                     max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
                     pipeline: str = "auto",
                     video_fps: int = DEFAULT_VIDEO_FPS,
                     seconds_per_day=None) -> dict:
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

    files = sorted(glob.glob(str(fits_dir / "*.fits")))
    stats["n_files"] = len(files)
    if not files:
        return stats

    try:
        first_img = _load_fits_frame(files[0])
        stats["shape"] = f"{first_img.shape[0]}x{first_img.shape[1]}"
    except Exception:
        first_img = None
        pipeline = "streaming"  # can't shmem without a shape/dtype peek

    timestamps = [_extract_timestamp(p) for p in files]
    video_stride = _compute_video_stride(timestamps, video_fps, seconds_per_day)

    if first_img is not None:
        chosen, need, avail = _choose_pipeline(pipeline, len(files), first_img)
        print(f"    pipeline={chosen}  "
              f"(needs {need/1024**3:.2f} GB, avail {avail/1024**3:.2f} GB)")
    else:
        chosen = "streaming"

    if chosen == "shmem":
        return _shmem_run(
            runname=runname, files=files,
            worker_fn=_fit_worker_fits_shmem,
            loader_frame=_load_fits_frame, timestamps=timestamps,
            out_csv=out_csv, run_dir=run_dir, stats=stats,
            make_plots=make_plots, make_movie=make_movie,
            first_img=first_img, video_stride=video_stride,
            video_fps=video_fps,
        )
    worker_fn = partial(_fit_worker_fits, need_image=make_movie)
    return _stream_run(
        runname=runname, files=files, worker_fn=worker_fn,
        loader_frame=_load_fits_frame, timestamps=timestamps,
        out_csv=out_csv, run_dir=run_dir, stats=stats,
        make_plots=make_plots, make_movie=make_movie,
        max_in_flight=max_in_flight,
        video_fps=video_fps, video_stride=video_stride,
    )


def process_image_run(run_dir: Path, make_plots: bool = True,
                      make_movie: bool = True,
                      max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
                      pipeline: str = "auto",
                      video_fps: int = DEFAULT_VIDEO_FPS,
                      seconds_per_day=None) -> dict:
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

    try:
        first_img = _load_image_frame(images[0])
        stats["shape"] = f"{first_img.shape[0]}x{first_img.shape[1]}"
    except Exception:
        first_img = None
        pipeline = "streaming"

    timestamps = [_extract_timestamp_image(p) for p in images]
    video_stride = _compute_video_stride(timestamps, video_fps, seconds_per_day)

    if first_img is not None:
        chosen, need, avail = _choose_pipeline(pipeline, len(images), first_img)
        print(f"    pipeline={chosen}  "
              f"(needs {need/1024**3:.2f} GB, avail {avail/1024**3:.2f} GB)")
    else:
        chosen = "streaming"

    if chosen == "shmem":
        return _shmem_run(
            runname=runname, files=images,
            worker_fn=_fit_worker_image_shmem,
            loader_frame=_load_image_frame, timestamps=timestamps,
            out_csv=out_csv, run_dir=run_dir, stats=stats,
            make_plots=make_plots, make_movie=make_movie,
            first_img=first_img, video_stride=video_stride,
            video_fps=video_fps,
        )
    worker_fn = partial(_fit_worker_image, need_image=make_movie)
    return _stream_run(
        runname=runname, files=images, worker_fn=worker_fn,
        loader_frame=_load_image_frame, timestamps=timestamps,
        out_csv=out_csv, run_dir=run_dir, stats=stats,
        make_plots=make_plots, make_movie=make_movie,
        max_in_flight=max_in_flight,
        video_fps=video_fps, video_stride=video_stride,
    )


# ---------------------------------------------------------------------------
# CLI (mirrors fits_reprocess.main with --max-in-flight added)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--root", default=str(ROOT), help="data root")
    parser.add_argument("--run", help="process only this run name (substring match)")
    parser.add_argument("--dry-run", action="store_true", help="list discovered work, don't process")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip csv_to_dotplots plots + summary")
    parser.add_argument("--no-movie", action="store_true",
                        help="skip .mp4 movie generation")
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT,
                        help=f"cap the streaming in-flight image buffer size "
                             f"(default: 2*cpu_count = {DEFAULT_MAX_IN_FLIGHT})")
    parser.add_argument("--pipeline", choices=["auto", "shmem", "streaming"],
                        default="auto",
                        help="pipeline mode: 'auto' picks shmem when the run's "
                             "images (x1.2) fit in available RAM, else streaming")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS,
                        help=f"movie frame rate (default: {DEFAULT_VIDEO_FPS}, "
                             f"matches dot_movie-Copy3; use 24 for 60 s/day)")
    parser.add_argument("--seconds-per-day", type=float, default=None,
                        help="movie decimation target — approx video seconds "
                             "per real day at the auto-detected capture rate "
                             "(default: none, movie keeps every frame)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Clean up any orphaned memmap temp files from previous crashed runs.
    for stale in OUTPUT_DIR.glob("*_shmem_*.dat"):
        try:
            stale.unlink()
            print(f"  removed orphaned shmem file: {stale.name}")
        except Exception:
            pass

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
            s = process_fits_run(run, make_plots=make_plots,
                                 make_movie=make_movie,
                                 max_in_flight=args.max_in_flight,
                                 pipeline=args.pipeline,
                                 video_fps=args.video_fps,
                                 seconds_per_day=args.seconds_per_day)
            all_stats.append(s)
            if s["n_files"] == 0:
                print(f"  -- no FITS files found in {s['source']}")
            else:
                rate = s["n_files"] / max(s["elapsed_s"], 1e-9)
                print(f"  -- {s['n_ok']}/{s['n_files']} ok in {s['elapsed_s']:.1f}s ({rate:.0f} f/s)")

        for i, run in enumerate(image_runs, len(fits_runs) + 1):
            print(f"[{i}/{total}] IMAGE {run.parent.name}/{run.name}")
            s = process_image_run(run, make_plots=make_plots,
                                  make_movie=make_movie,
                                  max_in_flight=args.max_in_flight,
                                  pipeline=args.pipeline,
                                  video_fps=args.video_fps,
                                  seconds_per_day=args.seconds_per_day)
            all_stats.append(s)
            if s["n_files"] == 0:
                print(f"  -- no images found in {s['source']}")
            else:
                rate = s["n_files"] / max(s["elapsed_s"], 1e-9)
                print(f"  -- {s['n_ok']}/{s['n_files']} ok in {s['elapsed_s']:.1f}s ({rate:.0f} f/s)")

    except KeyboardInterrupt:
        print("\nInterrupted -- partial results may have been written to reprocess_output/.",
              file=sys.stderr)
        if all_stats:
            grand_elapsed = time.time() - grand_t0
            write_timing_summary(all_stats, grand_elapsed)
            print(f"Partial timings written to {OUTPUT_DIR / 'run_timings.csv'}.",
                  file=sys.stderr)
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
