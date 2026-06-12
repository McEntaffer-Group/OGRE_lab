# Temperature/humidity ↔ dot-tracking integration

Conference-poster prep work. Joins independently-logged environmental data (SHT/MCP/HDC temperature, SHT/HDC humidity, 1 Hz, in `temperature/`) with per-frame dot centroid + FWHM extracted from FITS image runs on `E:/Reverse Telescope Test Data/`, so we can produce per-run time-aligned plots and answer questions like "how often does dot drift co-occur with a temperature swing in the same hour?"

## Status (2026-06-12)

| Phase | What | State |
|-------|------|-------|
| 1 | Normalize all temperature CSVs into per-day files in `temperature/daily/` | **Done.** 38 new daily files written via `tf.split_to_daily()`. Coverage 2025-10-27 → 2026-06-09 with two real logger-off gaps (2026-04-11 → 2026-04-26 turned out to exist after all; the only true gap is 2026-05-12 → 2026-06-03). |
| 2a | Write `fits_reprocess.py` — per-frame Gaussian fit, threaded, resumable, writes `{runname}_frames.csv` | **Done.** Smoke-tested on `20260302_data/genieshots` (2,550 frames, 10.7 min). Mean X/Y position and FWHM agree exactly with the existing `_summary.csv` to a factor of 0.15 (the old code's hardcoded pixel→arcsec). |
| 2b | Full E:/ reprocess pass (15 runs remaining) | **In flight** (kicked off 2026-06-12). Expected ~7 hours wall time at the observed 4 files/s. Skips `genieshots` (done) and `zoeysecondarygenie` (also done by the verification job). |
| 3 | Multi-dot tracking | **Deferred to v2.** Multi-peak frames are tagged via `n_peaks_x`/`n_peaks_y` columns; the v1 plot filters them out (`n_peaks_x == 1 & n_peaks_y == 1`). |
| 4a | `run_environment_plot.ipynb` — 3 stacked panels (X/Y drift, temperature, humidity) on shared time axis with day/6-hour gridlines | **Done.** End-to-end visual verification pending until the `zoeysecondarygenie` `_frames.csv` is available. |
| 4b | Event detection — rolling-window range exceedances per series, pairwise overlap counts, event windows highlighted on the plot | **Done.** Sanity-tested on a 5-day temperature window. Default thresholds (1.0 °C, 5 %RH, 1.0 px) are first-pass guesses — will need tuning against the first real plot. |
| 4c | `summarize_all_runs(root)` — cross-run descriptive statistics from concatenated `_events.csv` | **Done** (stub in notebook). Useful once multiple runs have been processed. |
| 5 | Vibration / accelerometer integration | **Deferred.** The 4 sessions in `accelerometer/` are all Nov 2025 and don't overlap with any of the recent (Apr–May 2026) image runs. Natural future path: a joint capture run + a 4th panel + a 5th event series. |

## How to use

Run all of these from the repo root using the `ReverseTelescopeDot` venv:

```
D:\Users\jad507\PycharmProjects\ReverseTelescopeDot\.venv\Scripts\python.exe
```

(That venv now has `astropy` and `pandas` installed alongside the existing numpy/scipy/matplotlib/PIL/cv2.)

### Phase 1 — normalize temperature data
```
<venv>\python.exe -c "import sys; sys.path.insert(0,'temperature'); import temp_functions as tf; tf.split_to_daily()"
```
Idempotent — only writes day files that don't already exist.

### Phase 2 — reprocess FITS into per-frame CSVs
```
<venv>\python.exe fits_reprocess.py               # all runs on E:/
<venv>\python.exe fits_reprocess.py --run NAME    # one run by substring match
<venv>\python.exe fits_reprocess.py --dry-run     # list discovered work
<venv>\python.exe fits_reprocess.py --force       # rebuild even if _frames.csv exists
```
Outputs `{run_path}/{runname}_frames.csv` next to the existing `_summary.csv` and updates `pixel_scales.csv` at the repo root with the detected camera per run (main vs Dalsa Genie, by image dimensions).

### Phase 4 — combined plot + events
Open `run_environment_plot.ipynb`, edit `RUN_PATH` in the config cell, run all cells. Outputs:
- `{run_path}/{runname}_environment.png` — the 3-panel figure
- `{run_path}/{runname}_events.csv` — every detected event window with start/end/series

Tunable in the same config cell: `EVENT_WINDOW`, `EVENT_THRESHOLDS`, `EVENT_SENSORS`, `ENV_RESAMPLE`.

## Architecture

- **All spatial values stored in pixels.** No arcsec conversion in `fits_reprocess.py`. The pixel scale is camera- and resolution-dependent and lives in `pixel_scales.csv` (currently has `camera` and `image_shape` columns auto-detected; `pixel_scale_arcsec_per_pixel` left blank for the user to fill in).
- **Per-run CSV is the join key.** `_frames.csv` has a `timestamp` column parsed from the FITS filename (` YY-MM-DD HH-MM-SS.fits` tail). The plot notebook calls `tf.builder(start, end, source_dir=temperature/daily)` to get aligned environmental data.
- **No edits to existing notebooks** (`dot_movie-Copy3.ipynb`, `timelapse_dot_movie.ipynb`). Their Gaussian fit was copied into `fits_reprocess.py` rather than imported, because notebooks aren't cleanly importable.

## Known gotchas

- **`2.355` vs `0.15`.** `2.3548` (= 2√(2 ln 2)) is the σ → FWHM factor for a Gaussian — pure math, same units in/out. `0.15 arcsec/pixel` is the camera pixel scale, hardcoded in both old notebooks and almost certainly wrong for the Dalsa Genie. The new code only stores pixels.
- **`dot_movie-Copy3.ipynb` cell `9180270b` writes ambiguous summary CSVs.** Its `_summary.csv` columns are named `'x position'`, `'FWHM x'`, etc. (no unit suffix) but the values are arcsec (multiplied by 0.15). Anyone reading them at face value will be off by ~6.67×. `timelapse_dot_movie.ipynb`'s summary is fine — suffixed `_px` and `_as` explicitly.
- **`temp_functions.builder()` against the parent `temperature/` folder crashes** with `TypeError: Cannot compare NaT with datetime.date object` because one legacy CSV has a NaT row. Workaround: pass `source_dir=temperature/daily`.
- **Camera identification has no FITS-header source.** Main camera = 1280×1024, Dalsa Genie = 1936×1216. Folder/run names containing `genie` are Genie runs. `fits_reprocess.py` infers from image shape.

## Open work / next iteration

- **Tune event-detection thresholds** against the first real plot. `1.0 °C` over 1 h is likely too strict for typical lab swings (~1 °C across 5 days observed in `data_hotandcold.csv`); something like `0.3` may catch the real events without flooding.
- **Fill in `pixel_scales.csv`** with the correct arcsec/pixel for both cameras at the resolutions used. Once populated, `plot_run_with_environment(units='arcsec')` (not yet wired) becomes a one-line addition.
- **Multi-dot tracking (v2).** For frames where `n_peaks_x > 1` or `n_peaks_y > 1`, run 2D centroiding (photutils DAOStarFinder or scipy.ndimage.label) and write a per-(frame, dot) CSV.
- **Vibration integration.** Requires a joint capture run where the accelerometer logger runs concurrently with a dot capture. Add a 4th panel and a 5th event series.
- **Fix `temp_functions._date_range_from_csv` NaT crash** — small bug, easy fix, currently routed around by pointing at `daily/`.
