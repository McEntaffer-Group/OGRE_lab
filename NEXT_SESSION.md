# Pick up here — 2026-09-03

Branch `claude/build-test-suite`, commit `24f1f9c`. Not merged, not pushed.

## State in one paragraph

The centroid-railing bug is **fixed** in `fits_reprocess.py` (both entry points —
the parallel module imports the same functions). A 304-test suite exists in
`tests/` and passes except for two failures that are deliberate. Nothing has been
reprocessed, so every CSV in `reprocess_output/` and every run directory on `E:`
still holds output from the **old, buggy** fit.

```
D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -q
```
→ `299 passed, 2 failed, 2 skipped, 3 xfailed` in ~46s. `-X utf8` is required.

**E: IS READ-ONLY.** Standing instruction: no writes, no overwrites, no deletes on
that drive. Reprocessing writes into run directories on E:, so it needs explicit
permission before anyone runs it.

## The two failures are correct

Neither can go green from a code change; both need a reprocess.

| test | why it fails | clears when |
|---|---|---|
| `test_no_ambiguous_bare_named_mirrors` | 8 stale bare-named `*_frames.csv` in `reprocess_output/` | affected runs reprocessed **and** stale files deleted (nothing does that yet) |
| `test_summary_agreement_with_truth_does_not_regress` | 6 runs `DIVERGENT_WITH_FAILURES` | affected runs reprocessed |

## Decisions waiting on you

1. **Reprocess or not, and what scope.** This is the gate on everything else. It
   writes to `E:` run directories, hours of compute, and rewrites
   `reprocess_output/`. At minimum the 6 divergent runs; the earlier scope
   estimate of "21 of ~78 runs" is a **floor**, not a count (see §1 of
   `TEST_PLAN.md`).
2. **`output_dir=` on `process_fits_run`.** Needed to build F1
   (worker-count invariance, the one planned test not written). The alternative
   is monkeypatching a module constant, which is fragile because
   `fits_reprocess_parallel.py:449` calls `OUTPUT_DIR.mkdir()` directly.
3. **Promote the detectors?** `tests/detect.py` has `on_bound`, `compact_peak`,
   `second_component`, `classify`. They are test-side only. If they move into
   `fits_reprocess.py`, `{run}_frames.csv` should gain `nx`/`ny`, `resid_x/y`,
   `on_bound_x/y`, `two_component` — which changes the CSV schema and so wants to
   happen in the *same* reprocess as (1), not a later one.
4. **`FWHM_MAX_PX`.** Still 1000. You said a dot can legitimately exceed 500 and
   see no reason it can't exceed 1000, so no test depends on its value. Open
   question: `20260306_springgenie` has **12,920 of 14,150 frames** cut by truth's
   `fwhm_max=500` and kept by us. If those are real, truth is discarding good
   data and that "divergence" is truth being wrong.

## Findings that want eyes, not code

- **3 runs still rail, on the LOWER sigma bound** — `20251014_minutelyovernight`,
  `20251029_longweekend`, `20251106_laser`. σ pinned at 1.0 (or at 240 = size/2
  for laser). This is a *different* signature from the bug just fixed: the fit
  collapsing onto something narrower than a pixel, not exploding. Listed in
  `KNOWN_UNUSUAL` in `tests/test_survey.py` with reasons; **not examined in 2D**.
- **19 runs flagged two-dot by detector only.** `KNOWN_TWO_DOT` in
  `tests/test_survey.py` marks which two are 2D-verified (`postwinterbreak`,
  `frosty`) and which are not. Seven of the 19 are `laser*` runs, which is the
  expected population — but they are unconfirmed. To verify one: threshold the 2D
  frame and look for a compact source far from the fitted centroid.
- **`n_peaks_x`/`n_peaks_y` are uninformative** — >1 on 100% of frames in 22 runs,
  catching noise ripple on broad plateaus. Either retune
  `PEAK_HEIGHT_FRAC`/`PEAK_MIN_DISTANCE_PX` or stop writing the columns.

## Traps — things that already bit this session

- **`reprocess_output/*_frames.csv` is not a per-run index.** 20 runs collapse into
  8 bare-named files, each holding whichever run finished last (`minutely` is 7
  runs, `morning5237` is 6). Any glob over that directory silently reads one
  arbitrary run as if it were all of them — that is how the handoff's §1.5 table
  ended up describing 1 of 7 `minutely` runs. Read per-run CSVs from the run's own
  directory on `E:` instead. `compare_pipelines._frames_files()` already handles
  this correctly; ad-hoc globs do not.
- **`fit_ok=False` in existing CSVs is stale.** 23 of 26 failing fixture frames now
  fit cleanly. Do not treat the stored flag as ground truth for the current code.
- **The summary CSV's "good frames" is post-FWHM-gate**, not the fit-level
  `fit_ok`. That is why it says 50 runs have failures while only 13 have
  `fit_ok=False` rows.
- **The survey samples one frame per run**, and two-dot detection is
  frame-dependent: postwinterbreak's median frame has its sources 105 px apart and
  is not flagged, while frames 1/501/5001 sit 357-411 px apart and are. G4
  therefore **undercounts**. The three postwinterbreak anchors exist to catch this.
- **Bash heredocs mangle `\n` inside Python string literals** in this environment —
  it became a real newline and produced a syntax error. Use the Write/Edit tools
  for Python containing escapes.

## Where things are

| file | what |
|---|---|
| `TEST_PLAN.md` | the plan, plus **§15 RESULTS** — read §15 first |
| `FIT_DIAGNOSIS_HANDOFF.md` | the original diagnosis. **Partly stale** — see its header |
| `tests/detect.py` | `on_bound`, `compact_peak`, `second_component`, `classify` |
| `tests/synth.py` | synthetic profiles; noise 60 is measured, do not use 174 |
| `tests/make_fixtures.py` | fixture extraction; reads `E:`, writes only `tests/fixtures/` |
| `tests/test_survey.py` | `KNOWN_UNUSUAL` and `KNOWN_TWO_DOT` live here |
