# Pick up here — 2026-09-04

Branch `claude/build-test-suite`. Working tree has **uncommitted changes** to
`fits_reprocess.py`, `tests/`, and a new untracked snapshot directory.

**A full reprocess of all data, including movies, is scheduled for the weekend
of 2026-09-04.** Everything below is written on the assumption it runs. If it
has already run, go to `POST_REPROCESS.md` instead — that is the checklist for
the next stage.

## Read in this order

| doc | what it is |
|---|---|
| this file | state, and what changed today |
| `POST_REPROCESS.md` | **the checklist for after the weekend run** — start here once it finishes |
| `CSV_SCHEMA.md` | what all 26 columns mean, and which names lie |
| `TEST_PLAN.md` §15 | the test suite's own results |
| `FIT_DIAGNOSIS_HANDOFF.md` | the original diagnosis; **partly stale**, see its header |

## State in one paragraph

The centroid-railing bug is fixed. The fit now measures sigma properly, takes a
single shot from that estimate, and only re-seeds when the result rails or
declines. `{run}_frames.csv` gained 10 columns that record the *evidence* for a
fit rather than only its answer. The test suite is at 316 passing with 2
deliberate failures that cannot go green without a reprocess. **Nothing has been
reprocessed** — every CSV on `E:` and in `reprocess_output/` still holds
old-fit output, and the pre-reprocess state is snapshotted in
`pre_reprocess_snapshot_20260904/`.

```
D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -q
```
→ `316 passed, 2 failed, 2 skipped, 3 xfailed` in ~45s. `-X utf8` is required.

**`E:` IS READ-ONLY** by standing instruction — no writes, overwrites or deletes
— *except* during an explicitly authorised reprocess, which writes into run
directories there.

## What changed on 2026-09-04

### The fit

- `_fit_profile` replaces the always-three-seeds ladder with **one fit from the
  measured estimate, re-seeding only when that lands on a bound or declines**.
  Justification, measured across all 234 fixture profiles: all three seeds land
  in the same basin every time, disagreeing by at most 0.002 px on `mu` and
  2.8e-9 relative on residual, and no seed ever failed. The ladder cost 3.3× and
  bought nothing. 230/234 now take the fast path; the 4 that don't are exactly
  the `on_bound` cases.
- `_fit_one_profile` remains as a 4-tuple wrapper so existing callers are
  unchanged.
- Suite runtime dropped 62s → 45s as a side effect.

### 10 new CSV columns

`nx, ny, resid_x, resid_y, on_bound_x, on_bound_y, noise_x, noise_y,
two_component, two_comp_sep`

All additive; every existing consumer reads by name and still works. See
`CSV_SCHEMA.md` for what each means. **They cannot be backfilled** — producing
them requires re-fitting.

The one to know about: **`resid / noise` is a scale-free goodness of fit**,
calibrated on real frames at `1.0–1.2` for a correct single-Gaussian model,
`5.5–6.4` for two-dot contamination, and `19–57` when the model does not
describe the data at all. Raw `resid` cannot be compared between runs because
source brightness varies 200×.

### Things found while building it

- **A blank frame is `fit_ok=True`.** `curve_fit` converges on the degenerate
  `amp≈0` solution, leaving `mu=1e-10` and `sigma=1.0` on their lower bounds;
  FWHM 2.355 sits inside the gate. `mu=1e-10` then enters the position series as
  a real centroid at pixel 0. Only `on_bound` catches it.
- **`FWHM_MAX_PX = 1000` is unreachable on frames ≤ 640 px**, because sigma is
  already bounded at `size/2`. Same number, wildly different strictness across
  the corpus. It has mostly been acting as a proxy for "this fit railed."
- **`n_peaks` has four measured blind spots** — it goes blind exactly during a
  dot crossing (separation below ~50 px), deletes companions fainter than half
  the profile range, invents peaks on any broad spot (σ=250 → 11 peaks from one
  source), and cannot see a pair aligned on the other axis. It is not noise
  though: 15 agreements against the independent detector, including both
  2D-verified cases.
- **The bare-name collision is worse than recorded.** 10 bare names map to 26
  distinct datasets. `20260814/postspie` and `20260814_data/postspie` are
  different files with different content.
- **`_prev.csv` backups are themselves bare-named and collide**, so the
  pipeline's own backup overwrites itself for the 20 colliding runs. Hence the
  snapshot below.

### `pre_reprocess_snapshot_20260904/`

96 per-run CSVs copied off `E:` (132.5 MB) with `MANIFEST.csv` giving sha256 and
source path per file. This is the **only** reliable record of the pre-fix state
for the colliding runs. Untracked — decide whether it belongs in git or
somewhere else.

## Decisions made today, and why

| # | decision | outcome |
|---|---|---|
| 1 | `n_peaks` | **leave unchanged** — free, gates nothing, and a continuous historical record. Retune after the reprocess using `two_comp_sep` to measure how often it went blind |
| 2 | `FWHM_MAX_PX` | **keep 1000.** Not because it is right, but because `on_bound` is recorded and does not gate, so the cap is currently the only thing keeping railed fits out of the position series |
| 3 | `two_component` | **added.** Costs 1.9% of fit time (~3 min CPU across 515k frames) |
| 4 | `output_dir=` | **deferred to after the run** — the only one that could affect a multi-hour job |

Rejected: lowering `FWHM_MAX_PX` to 500 to match the truth notebook. It would
turn `test_summary_agreement_with_truth_does_not_regress` green by discarding
~26,000 frames to agree with a threshold tuned on a camera whose spot is 12 px
wide. Green for the wrong reason.

## The two test failures are still correct

Neither can go green from a code change; both need the reprocess.

| test | why it fails | clears when |
|---|---|---|
| `test_no_ambiguous_bare_named_mirrors` | 8 stale bare-named CSVs in `reprocess_output/` | affected runs reprocessed **and** stale files deleted (nothing does that yet) |
| `test_summary_agreement_with_truth_does_not_regress` | 6 runs `DIVERGENT_WITH_FAILURES` | affected runs reprocessed; some may legitimately stay red — see `POST_REPROCESS.md` §6 |

## Open questions that need a human, not a threshold

1. **Are the other seven `*genie` runs the misaligned laser?** `springgenie` is
   known to be one — skewed, migrating across the frame, and by the end its +1σ
   point sits ~90 px past the frame edge while all 14,150 frames read
   `fit_ok=True`. The other seven look identical by every available measure
   (median σ≈150 vs 5.3 elsewhere; `genieshots` scores 20–34 on the mismatch
   ratio). If they are, that is 8 runs of unusable position data.
2. **Do the three lower-bound railers have a source at all?** Synthetic work
   reproduced `sigma → 1.0` two ways: a blank frame, and a source far enough
   off-frame that the remaining ramp is nearly flat. Testable now — pull a
   railed frame from each and look. See `POST_REPROCESS.md` §5.

## Scope of what the reprocess is fixing

25 of 96 runs showed bound-railing in the pre-fix CSVs — a real count from the
per-run files on `E:`, not the contaminated `reprocess_output/` glob that
produced the handoff's "21 of ~78" floor.

```
20251106/laser              62.7%      20251029/longweekend    24.8%
20260624/bridgetstatic      45.8%      20250919/weekend        24.1%
20260130/newsecondary       43.8%      20251014/minutelyover.  23.3%
```

14,978 railed rows out of 128,773 in those runs; 515,020 frames across all 96.

## Where things are

| file | what |
|---|---|
| `fits_reprocess.py` | the fit, the detectors, `_CSV_COLS`. The parallel module imports these — fixing one fixes both |
| `tests/detect.py` | now **delegates** to the promoted functions; no second copy to drift |
| `tests/synth.py` | synthetic profiles; noise 60 is measured, do not use 174 |
| `tests/make_fixtures.py` | fixture extraction; reads `E:`, writes only `tests/fixtures/` |
| `tests/test_survey.py` | `KNOWN_UNUSUAL` and `KNOWN_TWO_DOT` live here |
| `pre_reprocess_snapshot_20260904/` | the pre-fix state, with manifest |
