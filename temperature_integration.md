# Temperature/humidity ↔ dot-tracking integration

Conference-poster prep work. Joins independently-logged environmental data (SHT/MCP/HDC temperature, SHT/HDC humidity, 1 Hz, in `temperature/`) with per-frame dot centroid + FWHM extracted from FITS image runs on `E:/Reverse Telescope Test Data/`, so we can produce per-run time-aligned plots and answer questions like "how often does dot drift co-occur with a temperature swing in the same hour?"

## Status (2026-06-15)

| Phase | What | State |
|-------|------|-------|
| 1 | Normalize all temperature CSVs into per-day files in `temperature/daily/` | **Done.** 38 new daily files written via `tf.split_to_daily()`. Coverage 2025-10-27 → 2026-06-09 with two real logger-off gaps (the only true gap is 2026-05-12 → 2026-06-03). |
| 2a | Write `fits_reprocess.py` — per-frame Gaussian fit, threaded, resumable, writes `{runname}_frames.csv` | **Done.** Smoke-tested on `genieshots` (2,550 frames, 10.7 min). Mean X/Y position and FWHM agree exactly with the existing `_summary.csv` to a factor of 0.15 (the old code's hardcoded pixel→arcsec). |
| 2b | Full E:/ reprocess pass | **Done.** All 17 runs processed in 167.3 min. Every run has a `_frames.csv`. |
| 3 | Multi-dot tracking | **Deferred to v2.** Multi-peak frames are tagged via `n_peaks_x`/`n_peaks_y` columns. |
| 4a | `run_environment_plot.ipynb` — 3 stacked panels (X/Y drift, temperature, humidity) on shared time axis with day/6-hour gridlines | **Done.** Verified end-to-end on `zoeystatic` (April 2026, 100% single-dot). Saved figure visually plausible. |
| 4b | Event detection — rolling-window range exceedances per series, pairwise overlap counts, event windows highlighted on the plot | **Done.** Thresholds retuned on real zoeystatic data: position 15 px / 1h, temp 0.4 °C / 1h, humidity 3.5 %RH / 1h gives the upper-decile interpretation (10 X-drift, 32 Y-drift, 13 temp, 13 humidity events over 121.5 h; 20 joint, 22 position-only, 1 env-only). |
| 4c | `summarize_all_runs(root)` — cross-run descriptive statistics from concatenated `_events.csv` | **Done** (stub in notebook). Will accumulate as more runs are plotted. |
| 5 | Vibration / accelerometer integration | **Deferred.** The 4 sessions in `accelerometer/` are all Nov 2025 and don't overlap with any of the recent (Apr–May 2026) image runs. Natural future path: a joint capture run + a 4th panel + a 5th event series. |

### Cross-run notes from the full pass

- **Genie systematically projects to multi-peak Y profiles** (3–8 peaks). All four Genie-camera stability runs (`zoeysecondarygenie`, `zoeystaticgenie`, `springgenie`, `statictestgenie`) show 0% single-dot frames. This is likely an optical artifact, not real two-dot data. Workaround in notebook: set `FILTER_MULTI_PEAK = False` for Genie runs and trust the dominant-peak Gaussian fit. Worth understanding the root cause before v2 multi-dot work.
- **Camera detection gaps.** `pixel_scales.csv` has 5 runs flagged `unknown` (image shapes 1920×2560, 1944×2592, 1080×1920) — these don't match my main_camera (1024×1280) or dalsa_genie (1216×1936) mapping. Likely the main camera in a different binning/resolution mode. User should classify these manually.
- **Single-dot rates per run** (main_camera = consistently clean; Genie = consistently multi-peak):
  - 100% single-dot: replacedprimary, zoeystatic
  - ~80–100%: morewarming, statictest, secondarylaser, zoeysecondary, newprimary (a Genie that's clean — interesting)
  - ~50–75%: hotandcold, springbreak
  - 0% (Genie quirk): zoeysecondarygenie, zoeystaticgenie, springgenie, statictestgenie, genieshots, postwinterbreak
  - <5%: newsecondary, allmetal

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

- **Classify the 5 `unknown` cameras** in `pixel_scales.csv` (1920×2560, 1944×2592, 1080×1920). Likely main_camera at different binning. Once labeled, `plot_run_with_environment(units='arcsec')` (not yet wired) becomes a one-line addition.
- **Investigate the Genie multi-peak Y projection.** Every Genie stability run except `newprimary` shows 3–8 peaks in the Y profile of every frame. Could be diffraction pattern, rolling-shutter artifact, or optical alignment. Affects how trustworthy the dominant-peak Gaussian centroid is for those runs.
- **Multi-dot tracking (v2).** For frames where `n_peaks_x > 1` or `n_peaks_y > 1`, run 2D centroiding (photutils DAOStarFinder or scipy.ndimage.label) and write a per-(frame, dot) CSV.
- **Vibration integration.** Requires a joint capture run where the accelerometer logger runs concurrently with a dot capture. Add a 4th panel and a 5th event series.
- **Fix `temp_functions._date_range_from_csv` NaT crash** — small bug, easy fix, currently routed around by pointing at `daily/`.
- **Per-run threshold tuning.** Current defaults work for stable runs (zoeystatic). Runs that span HVAC events (`hotandcold`, `newprimary`) may need different thresholds; cross-run statistics are only comparable if thresholds are fixed, so pick once before the poster.
