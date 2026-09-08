# Pick up here — 2026-09-08

Branch `claude/build-test-suite`. Working tree clean apart from `__pycache__`.

**The full reprocess has run.** 2026-09-04 17:18 → 2026-09-06 02:37, 33.3 hours
wall, 94/94 runs, 503,407 frames, empty stderr. Everything below reflects the
post-reprocess state.

```
D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -q
```
→ `359 passed, 1 failed, 2 skipped, 3 xfailed` in ~66s. `-X utf8` is required.

**`E:` IS READ-ONLY** by standing instruction — no writes, overwrites or deletes
— *except* during an explicitly authorised reprocess.

## Read in this order

| doc | what it is |
|---|---|
| this file | state, and what is still open |
| `POST_REPROCESS.md` | the checklist; §0–§4 and §6 are **done**, results recorded there |
| `CSV_SCHEMA.md` | the 26 frame columns and the summary columns |
| `TEST_PLAN.md` §15 | the test suite's own results |
| `FIT_DIAGNOSIS_HANDOFF.md` | the original diagnosis; **stale**, see its header |

## The one thing to fix next

**`fit_ok` no longer gates anything, and 12,107 railed frames pass it.**

Before the fix, `FWHM_MAX_PX = 1000` accidentally caught railed fits, because a
railed fit usually had an absurd width. Now the fits don't explode, so the cap
never fires:

```
mean fit_ok across all 94 runs   100.00%
on_bound rows that PASS fit_ok    12,107
on_bound rows that FAIL fit_ok         6
```

`CSV_SCHEMA.md` predicted the cap was "mostly a proxy for this fit railed". That
is now confirmed, and the corollary is that the accidental protection is gone.
`on_bound` is recorded but does not gate. Making `fit_ok` account for `on_bound`
was deferred as low priority on 2026-09-04; it is the top item now.

Note before changing it: a gate of `not on_bound and SNR > 5` removes **18,685**
frames, more than twice what the old despike removed and 2,600× what `fit_ok`
removes today. `SNR > 5` alone accounts for 17,923 of those and is an invented
threshold. It needs its own justification before it becomes a default — see the
measured table in `POST_REPROCESS.md` §3.

## What the reprocess showed

- **Railing dropped from 25 runs to 11 of 94**, 12,113 rows of 503,407 (2.41%).
  Worst: `longweekend` 69.5%, `laser` 60.0%, `minutelyovernight` 36.1%.
- **The failure mode changed.** It is now overwhelmingly the *lower* bound
  (11,642 axis-hits at σ<1.001 vs 3,031 upper) — collapse-to-a-pixel, not
  explosion.
- **The σ ≥ 1 bound is doing its job.** The railers are not sharp dots: median
  amp/noise **1.97** against 71.5 for healthy interior fits; 68.6% below SNR 3;
  12 of 11,642 above SNR 10. `longweekend`'s median is 0.67, below the noise.
  Sub-pixel σ is unmeasurable on a summed profile anyway — amplitude and width
  trade off freely below one pixel. Lowering the bound is only defensible
  alongside an SNR gate.
- **The genie question is answered.** All eight `*genie` runs sit at the top of
  `resid/noise`, and **none of them rail** — 0.0% `on_bound`, 100% `fit_ok`.
  They converge cleanly onto a broad plateau a Gaussian does not describe.

  ```
  bridgetstaticgenie 46.98   springgenie      37.07   genieshots      35.54
  newprimaryagain    35.54   newprimary       25.66   statictestgenie 18.78
  ```

  ~59,000 frames of suspect position data that every existing gate calls good.
  `newprimary` / `newprimaryagain` are in the same band at σ≈57 with no `genie`
  in the name — **unexplained, needs eyes on frames.**

## What changed on 2026-09-07/08

### `29a2594` — run discovery

`_discover_image_runs` skipped every `{date}/` that had a `{date}_data/` sibling.
But a `_data` folder only means *some* of that date was converted to FITS. Runs
never converted were invisible to both discovery paths.

Fixed by excluding runs individually, matched case-insensitively. **95 runs now,
up from 94** — `20250923/collimationtests` was the hidden one (16 frames). It was
one partially-converted date away from dropping a large run.

`discovery_anomalies()` now prints before every run so nothing is skipped
silently:

```
[mismatched_fits]    20260306_data/springgenie_big   holds springgenie_fits/ (14150 .fits)
[case_mismatch]      20250923/bigShakeB              counterpart is 'bigshakeB'
[partial_conversion] 20260320/statictestgenie        7018 images but 5676 FITS
[partial_conversion] 20260130/newsecondary           4143 images but 3999 FITS
```

Both partial conversions are clean truncated tails, not scattered loss. FITS is
preferred over images of the same name, so those 1,486 tail frames are dropped —
deliberately, and now reported.

### `b5bcfaa` — time axis, and the despike

**Sub-second reconstruction.** 31 of 96 runs stamp whole seconds, so a burst put
~50 frames on one instant (99 for `nightvideo`) and plotted as a vertical stroke
per second. `_subsecond_times()` lays each bucket out at the rate implied by the
*interior* buckets, and lays the **first** bucket out backwards from the
following second boundary because it is partial. 15 of 30 affected runs open with
a bucket under half the interior rate. Verified: no frame leaves its stamped
second, ordering preserved, and runs with too few buckets fall back to frame
index. Slope impact is 23.5% on `maxtest2` (10 s span) and ~0.02% median.

**The despike is commented out**, not deleted, with measurements in place. It
removed 8,237 frames of which **3,033 (36.8%) were clean, high-SNR, non-railed
fits**; the MAD fence collapses on quiet runs (`statictestgenie` lost 155 frames
at a median deviation of **0.2 px**); and it removed frame 0 on 17 runs, shifting
whole curves (`statictestgenie` by 105.8 px). Filtering in
`plot_run_environment` is back to "curve_fit did not error".

### `4977958` — drift reporting

- **487 of 493 bare-named files deleted** from `reprocess_output/` (204 MB). All
  predated the reprocess and shadowed the current run-keyed output. Recoverable
  from git history. This cleared `test_no_ambiguous_bare_named_mirrors`.
- **Both drift rates now shown.** The old "Slope" was an endpoint chord
  `(y_last - y_first) / span` — two frames. Kept and relabelled `Endpoints`;
  a least-squares fit is drawn alongside. On `allmetal`'s y they disagree in
  **sign**: -4.49 vs +6.92 px/day.
- **Worst-case drift** added to the summary: `x/y min`, `max`, `range` in px and
  as, plus `min time` / `max time`. `allmetal` y range is **93.20 px** against
  reported drift of a few px/day either way.

**Correction worth keeping:** the endpoint chord is **not** a truth-agreement
measure. `compare_pipelines.TRUTH_PAIRS` compares position and FWHM means and
stds; no slope is among them. It is kept for visual comparability with the
legacy plots, nothing more.

## Still open

1. **`fit_ok` should account for `on_bound`** — see the top of this file.
2. **`newprimary` / `newprimaryagain`** score 25–36 on `resid/noise` with no
   `genie` in the name. Needs a look at real frames.
3. **The last test failure**: `20260420_zoeysecondary`, `DIVERGENT_UNEXPLAINED`,
   0.286 on FWHM x std. That run is also 14.3% railed, so it is probably the same
   cause rather than a plotting issue. `POST_REPROCESS.md` §6.
4. **Everything on `E:` still has the old plots and summaries.** The new drift
   lines and range columns only exist for runs processed after `4977958`. The
   `_frames.csv` files are current, so **plots and summaries can be regenerated
   without re-fitting** — far shorter than the 33-hour run. Nothing does this in
   bulk yet.
5. **`csv_to_dotplots` has no gap-breaking.** `plot_run_environment.break_gaps()`
   inserts NaN rows at cadence jumps so dead time renders as a break; the
   production plots join across it with a straight line that reads as slow drift.
   Visible on `maxtest2` now that the vertical strokes are gone.
6. **Tick density on multi-day runs.** `allmetal` spans 2.72 days, lands in the
   `2 * 86400` tier, and gets exactly 3 major x labels (one per midnight). The
   tier below uses 6-hour majors and would give 11. Everything from 2 to 8 days
   has this.
7. **`springgenie_big`** holds 14,150 FITS, the same count as `springgenie` —
   almost certainly a duplicate, but discovery cannot tell. Renaming its
   `springgenie_fits/` to `springgenie_big_fits/` would make it process.

## Where things are

| file | what |
|---|---|
| `fits_reprocess.py` | the fit, detectors, `_CSV_COLS`, discovery, `discovery_anomalies` |
| `fits_reprocess_parallel.py` | imports all of the above; fixing one fixes both |
| `csv_to_dotplots.py` | production plots, `_subsecond_times`, `_slope_line`, `build_summary` |
| `plot_run_environment.py` | environment plots; despike commented out in `load_frames` |
| `tests/test_discovery.py` | 12 discovery tests (new) |
| `tests/test_timeaxis.py` | 15 sub-second tests (new) |
| `tests/test_slope.py` | 15 drift/range tests (new) |
| `pre_reprocess_snapshot_20260904/` | the pre-fix state, with sha256 manifest |
