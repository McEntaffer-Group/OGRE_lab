# Pick up here — 2026-09-17

Branch `claude/build-test-suite`, with `main` merged in (`1c45d92`). `main` is
now an ancestor of this branch, so the PR `claude/build-test-suite` → `main`
fast-forwards and cannot lose anything on `main`. **No PR has been opened yet.**

## Correction: the data-loss alarm this page used to carry was wrong

An earlier version of this handoff read `git diff main claude/build-test-suite`
— 891 files, ~1.7M insertions, ~2M deletions — and concluded a PR would delete
two months of temperature data. That is a **two-dot diff**: it renders every
file that exists only on `main` as a deletion. It is not what a merge does.

`git merge-tree --write-tree main claude/build-test-suite` showed the real
shape: **11 conflicted paths**, and every temperature log added on either side
after the split preserved, because none of them existed at the merge base
(`52e9528`). Both sides' additions survive a three-way merge.

If you compare two branches again, simulate the merge. Do not read a two-dot
diff as a merge preview.

## Temperature data: merged, and now actually reachable

`main` did hold twenty days this branch lacked — `2026-08-28`..`2026-09-16`,
exactly `postspiererun`'s window — but at the **repo root**, where nothing
reads them. `plot_run_environment.TEMP_DAILY` points at `temperature/daily/`,
and `load_environment()` resolves every run's series through it. The data was
committed and still invisible to the pipeline.

Ingested into `temperature/daily/` in `87457de`, CRLF → LF to match the archive
(`daily/` is LF-only; the legacy `temperature/*.csv` mirrors are CRLF). Each
file verified byte-identical to `main`'s copy modulo `\r`, and single-date.
Root-level copies kept as well, by request.

The archive now spans `2025-10-27`..`2026-09-16`, 234 files. Over the real run
window:

```
tf.builder('2026-08-28 17:16:19','2026-09-16 11:03:40', source_dir='temperature/daily')
→ 53,973 rows @30s, 0.0 NaN in all five series, no gap warnings
```

**`2026-08-19`..`08-27` is missing from every ref** — a collection gap, not a
merge artifact. It falls before `postspiererun` starts, so that run is unaffected.

## How the 11 conflicts were resolved

All one root cause: this branch migrated `reprocess_output/` to date-prefixed
`run_key`s; `main` still held the pre-migration copies.

| conflict | resolution |
|---|---|
| 8× bare `postspie*`/`postspiegenie*` (modify/delete) | took this branch's deletion; all eight have `20260814_*` twins, verified present |
| `all_runs_summary_reprocess.csv` | took this branch (95 rows vs `main`'s 2, which are covered) |
| `run_timings.csv` | took this branch (485 rows); carried `main`'s two OneDrive rows into the 9-column schema |
| `all_runs_summary.csv` | took this branch's 38 rows **+ `bridgetstatic` and `run2`** |

`run_timings.csv`: recomputing `_TOTAL_` with the dedup from `0c28dc9` gives
553382/553365/124807.29 — identical to the stored row — over 97 unique runs,
matching `pixel_scales.csv`. The `2026-09-13` row already won that dedup key,
so the carried rows are provenance only. `main`'s stale batch-local `_TOTAL_`
row was deliberately not imported.

The 163 bare-named files `main` has and this branch does not are the intentional
cleanup from `4977958`; every one was confirmed to have a run-keyed twin before
the merge was committed.

### What is left

1. Fix `_collect_summaries()` / `_collect_reprocess_summaries()` — see below.
   Still real, still unfixed.
2. Open the PR `claude/build-test-suite` → `main` (fast-forward).
3. Rerun `postspiererun`/`postspiererungenie` — temperature data is now in place
   for the full window.
4. `__pycache__` and `.idea` are still tracked on both sides. Untrack and
   `.gitignore` them; listed before, still outstanding.

**Not verified: the test suite.** `pytest` is absent from both local venvs
(`.venv`, `.venv12`), the D: interpreter this page documents is not on this
machine, and neither D: nor E: is mounted.
`test_no_ambiguous_bare_named_mirrors` — the one invariant these changes could
plausibly break — was evaluated directly and passes. Run the full suite when
you are next on a machine with the drives.

## Code bug found but NOT fixed this session: `_collect_summaries()` /
## `_collect_reprocess_summaries()` silently lose history

In `fits_reprocess.py`, these two functions (~line 1243 and ~line 1271)
build `all_runs_summary.csv` / `all_runs_summary_reprocess.csv` by doing a
fresh `root.rglob(...)` sweep and overwriting the output file from scratch
every time — unlike `write_pixel_scales()` (~line 1399), which explicitly
merges by `run_key` and preserves every row not in the current batch.

This bit hard this session: reprocessing `postspiererun`/`postspiererungenie`
against a `--root` pointed at the OneDrive copy of the data — which has raw
frames mirrored for all ~97 historical runs, but reprocessed *outputs*
(`*_summary_reprocess.csv` etc.) only for the 2 runs actually reprocessed
there — caused both collector CSVs to collapse from ~97 rows to 6. Reverted
rather than committed; **not fixed**. Fix should mirror `write_pixel_scales`'s
merge-by-`run_key` pattern.

Also found in the same investigation: a **stale duplicate** output file at
the pre-fix output layout, e.g.
`{date}/{runname}/{runname}_summary_reprocess.csv` sitting alongside the
current correct location `{date}/{runname}_summary_reprocess.csv` — both
match the `rglob` pattern, so runs with a leftover old-layout copy get
double-counted. The module docstring already promises a
`--migrate-image-outputs` flag for exactly this; check whether it exists and
actually handles it, or needs writing.

**Correction:** an earlier version of this page called two untracked files
collateral damage from this bug, with `notes` "copied from the wrong run", safe
to delete. Both claims were wrong. Their notes — "second primary test" and
"second secondary test" — agree exactly with `main`'s authoritative rows, and
`main`'s rows were collected from these very files. They were the **only**
source for the two runs `main` recorded that this branch did not:
`bridgetstatic` and `run2`. Deleting them would have dropped both runs at merge
time. Now tracked in `2e5966a` and folded into `all_runs_summary.csv`.

`run2` is a reprocess of the same `zoeysecondarygenie` data out of the OneDrive
folder the mirrored filename names `run2` — identical frame count and window to
`20260420_zoeysecondarygenie`, but carrying the reprocess column family rather
than the original one. Still worth understanding before anything derived from it
is deleted.

## What was actually fixed and verified this session (already pushed)

1. **Movie frame ordering** (`list_run_images()` in `fits_reprocess.py`, and
   the FITS `sorted(glob.glob(...))` calls in both `fits_reprocess.py` and
   `fits_reprocess_parallel.py`). Frames were sorted by filename *string*,
   not numeric frame number. Frame numbers zero-pad to a fixed width only up
   to that width (`"0001".."9999"`), then grow unpadded (`"10000"...`) — a
   string sort interleaves the two widths, which played back as the movie
   repeatedly snapping back to early frames throughout. This is why
   `postspiererun` (crossed 9999 frames, 4-digit padding) showed the bug and
   `postspiererungenie` (padded to 5 digits from frame 1, never crosses a
   width boundary within this run) didn't. Fixed to sort by numeric
   `frame_num`, filename as tiebreak.
2. **`run_timings.csv` `_TOTAL_` row**, fixed twice:
   - First bug: `_TOTAL_` was computed from only the current batch's
     `stats_list`, so a `--run`-filtered invocation (this session ran
     `--run postspie`) silently dropped every historical run's contribution
     from the total (503,423 → 49,959 files).
   - Second bug (in my own first fix): switched to summing every row in the
     file, but the dedup key (`run_key`, falling back to `runname|source`)
     failed to unify a run's pre-`run_key`-schema log rows with its later
     re-processed rows (same `source`, blank vs populated `run_key`) —
     doubled the total to 1,068,408 instead of the real 553,382. Fixed by
     recovering the run's date from `source` via regex
     (`r"(20\d{6})"`) instead of falling back to the raw path, so legacy
     rows collapse onto the same key a modern `run_key` would produce.
     Verified against `pixel_scales.csv`'s 97 rows (which uses the correct
     merge-by-key pattern already) — now matches exactly.
3. **Verified clean on the real overnight run**: `postspiererun`
   (26,981 frames) / `postspiererungenie` (22,978 frames) — 100% `fit_ok`,
   zero duplicate or out-of-order `frame_num`, zero timestamp gaps, median
   60s cadence throughout, both `.mp4`s present at expected sizes. The
   filename-sort fix holds on a real 27k/23k-frame run.

## One unanswered question from last session

`postspiererungenie` has **79% of frames flagged `two_component`** (two-source
contamination) vs **1.9%** on `postspiererun` (main camera). Asked the user
whether a second source is expected on the genie camera for this run
(similar sibling runs `postspie`/`postspiegenie` have notes mentioning laser
tests) — not yet answered as of this handoff.

## Where things are

| file | what |
|---|---|
| `fits_reprocess.py` | the fit, discovery, `write_timing_summary`, `write_pixel_scales`, `_collect_summaries`/`_collect_reprocess_summaries` (the latter two need the merge-safety fix) |
| `fits_reprocess_parallel.py` | imports all of the above; fixing one fixes both |
| `reprocess_output/run_timings.csv` | now correct; dedup logic to reuse if a similar total needs computing elsewhere |
| `reprocess_output/all_runs_summary.csv`, `all_runs_summary_reprocess.csv` | currently at `HEAD` (pre-this-session) state, reverted; do not regenerate against a partial `--root` until the merge fix lands |

---

## Older handoff (2026-09-08) — superseded above for git/branch state, items
## below are likely still open and untouched this session

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

### Read in this order

| doc | what it is |
|---|---|
| this file | state, and what is still open |
| `POST_REPROCESS.md` | the checklist; §0–§4 and §6 are **done**, results recorded there |
| `CSV_SCHEMA.md` | the 26 frame columns and the summary columns |
| `TEST_PLAN.md` §15 | the test suite's own results |
| `FIT_DIAGNOSIS_HANDOFF.md` | the original diagnosis; **stale**, see its header |

### The one thing to fix next (as of 2026-09-08)

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

### What the reprocess showed

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

### Still open (as of 2026-09-08)

1. **`fit_ok` should account for `on_bound`** — see above.
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
