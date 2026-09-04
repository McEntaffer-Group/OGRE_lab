# After the full re-run — what to check, in order

Written 2026-09-04, before the weekend reprocess. This is the checklist for the
session that picks up once it finishes.

Interpreter for everything here (`-X utf8` is required; the default cp1252
console encoding crashes on non-ASCII output):

```
D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8
```

Run from `D:/Users/jad507/PycharmProjects/OGRE_lab`. Several snippets need
`PYTHONPATH=.` to import `fits_reprocess`.

---

## 0. Before you trust anything

**Confirm the run actually used the new code.** If any CSV lacks the 10 new
columns, that run was processed by an old build and everything below is invalid
for it.

```python
import pandas as pd, glob, os
NEW = ["nx","ny","resid_x","resid_y","on_bound_x","on_bound_y",
       "noise_x","noise_y","two_component","two_comp_sep"]
files = (sorted(glob.glob("E:/Reverse Telescope Test Data/*/*_frames.csv"))
         + sorted(glob.glob("E:/Reverse Telescope Test Data/*/*/*_frames.csv")))
stale = []
for f in files:
    cols = pd.read_csv(f, nrows=0).columns
    missing = [c for c in NEW if c not in cols]
    if missing:
        stale.append((f, missing))
print("%d files, %d stale" % (len(files), len(stale)))
for f, m in stale:
    print("  ", f, "missing", m)
```

**The pre-reprocess state is in `pre_reprocess_snapshot_20260904/`** — 96 CSVs,
132.5 MB, with `MANIFEST.csv` giving sha256 and source path per file. This
exists because `_prev.csv` mirrors are bare-named and collide, so for the 20
colliding runs the pipeline's own backup overwrites itself. Use the snapshot for
any old-vs-new comparison, not `reprocess_output/*_prev.csv`.

---

## 1. Did the fix hold at scale?

Everything so far was measured on 234 committed fixture profiles — one frame per
run. This is the first look at all ~515,020 frames.

```python
import pandas as pd, numpy as np, glob, os
files = (sorted(glob.glob("E:/Reverse Telescope Test Data/*/*_frames.csv"))
         + sorted(glob.glob("E:/Reverse Telescope Test Data/*/*/*_frames.csv")))
def key(f):
    parts = f.replace("\\","/").split("/")
    stem = os.path.basename(f)[:-len("_frames.csv")]
    parent = parts[-2]
    date = parts[-3] if parent == stem else parent
    return date + "__" + stem          # keep _data: it distinguishes real datasets
rows = []
for f in files:
    d = pd.read_csv(f)
    if "on_bound_x" not in d.columns: continue
    n = len(d)
    ob = (d.on_bound_x | d.on_bound_y)
    rows.append((key(f), n, int(ob.sum()), 100*ob.mean(),
                 int(d.mu_x.isna().sum()), 100*d.fit_ok.astype(bool).mean()))
rows.sort(key=lambda r: -r[3])
print("%-44s %8s %8s %7s %7s %8s" % ("run","rows","on_bound","pct","declined","fit_ok%"))
for r in rows[:30]:
    print("%-44s %8d %8d %6.2f%% %7d %7.1f%%" % r)
print("TOTAL on_bound: %d of %d (%.2f%%)"
      % (sum(r[2] for r in rows), sum(r[1] for r in rows),
         100*sum(r[2] for r in rows)/sum(r[1] for r in rows)))
```

**Expected:** the 25 runs that showed bound-railing before should collapse to
near zero, except the three known lower-bound cases. Pre-fix baseline for
comparison — 14,978 railed rows out of 128,773 in the affected runs:

```
20251106/laser              62.7%      20251029/longweekend    24.8%
20260624/bridgetstatic      45.8%      20250919/weekend        24.1%
20260130/newsecondary       43.8%      20251014/minutelyover.  23.3%
```

**If bound-railing is still widespread**, the fix did not generalise off the
fixtures — stop and investigate before drawing any physics conclusions.

---

## 2. Was the ladder fallback ever needed?

The fit now takes one shot from the measured estimate and only re-seeds when
that lands on a bound or declines. On the fixtures 230/234 took the fast path.
`on_bound` is the recorded trigger, so the fallback's necessity is now
measurable:

```python
# fraction of frames where the fast path was NOT trusted
ob = (d.on_bound_x | d.on_bound_y).mean()
```

**If this is ~0 across the whole corpus, delete the fallback.** The measured
justification for keeping it was thin to begin with: across all 234 fixture
profiles the three seeds agreed to within 0.002 px on `mu` and 2.8e-9 relative
on residual, and no seed ever failed. It costs 3.3× when it runs.

---

## 3. Calibrate `resid / noise` on real data

This is the highest-value new thing in the file and the numbers below come from
**twelve hand-picked frames**. Re-derive them from the full corpus.

```python
import pandas as pd, numpy as np, glob, os
rows = []
for f in files:
    d = pd.read_csv(f)
    if "noise_x" not in d.columns: continue
    ok = d[d.fit_ok.astype(bool)]
    if not len(ok): continue
    rx = (ok.resid_x/ok.noise_x).replace([np.inf,-np.inf], np.nan)
    ry = (ok.resid_y/ok.noise_y).replace([np.inf,-np.inf], np.nan)
    r = pd.concat([rx, ry])
    rows.append((key(f), len(ok), float(r.median()), float(r.quantile(.9)),
                 float(ok.noise_x.isna().mean()*100)))
rows.sort(key=lambda t: -t[2])
print("%-44s %8s %9s %9s %9s" % ("run","good","med ratio","p90","noise NaN%"))
for r in rows:
    print("%-44s %8d %9.2f %9.2f %8.1f%%" % r)
```

Hand-measured anchors to check the distribution against:

```
allmetal, springbreak    single dot, model correct        1.0 - 1.2
postwinterbreak          two dots, fit on the wrong one   5.5 - 6.4
genieshots, springgenie  truncated skewed laser          19   - 57
```

**What to produce from this:** a defensible threshold, replacing the provisional
`<2 good / 2-10 suspect / >10 model-is-wrong`. Look for whether the distribution
is genuinely bimodal or a continuum — that decides whether a threshold is the
right tool at all.

**Also check `noise NaN%`** — a high rate means the source fills the frame for
that run, which is the springgenie condition and disqualifies the run from
position analysis regardless of what any gate says.

---

## 4. Settle the genie question

**The open physical question.** `20260306_springgenie` is known (from the user,
not from the data) to be a misaligned laser pointer — skewed, and migrating
across the frame. The other seven `*genie` runs are unconfirmed but look
identical by every available measure:

```
genie      8 runs   median sigma_x=150.5  sigma_y=155.3   n_peaks fires 100%
non-genie 88 runs   median sigma_x=  5.3  sigma_y=  6.5   n_peaks fires 33%
```

`genieshots` scores 20–34 on the mismatch ratio, i.e. as bad as springgenie.

**If all 8 are the laser, that is 8 runs of unusable position data**, and it also
means the 9 `n_peaks` "false alarms" were never false — a broad plateau really
does have many local maxima. This needs a human looking at frames, not a
threshold.

Why the position numbers cannot be trusted for these runs even though every
frame is `fit_ok=True` — two measured effects:

**Truncation.** While the true centre is on the detector, even on the last
pixel, the fit is fine (mu recovered to 0.33 px, sigma to 0.04%). Once it
crosses, `mu` pins to `size` and `sigma` collapses:

```
  true mu   frac off     fit mu   fit sigma    fit amp     resid
     1216       100%    1215.97      249.90     140028     392.3
     1338       110%    1216.00      189.13     111371    2365.3
     1946       160%    1216.00       70.02       1549     409.4
     2432       200%     826.43        1.00        855     389.7   <- interior mu, railed sigma
```

Note the last row: far off-frame, it returns a *plausible interior position*.
Only `on_bound` (via sigma) catches it. And every seed from 50 to 600 returns
that identical answer — this is non-identifiability, not a local minimum, so no
re-seeding strategy can help.

**Skew.** A symmetric Gaussian fit to a skewed blob lands near the distribution's
**mean**, not its peak:

```
  skew a   true peak   true mean    fit mu   fit sigma     resid
       0       600.0       600.8    600.00      249.96     396.8
       4       704.0       786.5    755.93      141.89   10347.2
      16       642.0       792.1    749.87      128.04   17112.1
```

As skew increases the true peak *moves* (704 → 642) while the fitted mu barely
budges (756 → 750). **The reported centroid decouples from the physical spot** —
you would record a stable position while the thing itself moved 60 px.

springgenie today: 14,150 frames, **100% `fit_ok`**, `mu_y` drifting 992 → 1096
in a 1216-tall frame with `sigma_y ≈ 245`, so the +1σ point sits ~90 px past the
edge. Nothing in the old schema flagged any of it.

---

## 5. Test the lower-bound hypothesis

`20251014_minutelyovernight`, `20251029_longweekend`, `20251106_laser` rail with
`sigma → 1.0`. Section 4's synthetic experiment reproduced exactly that signature
two ways: a blank frame, and a source so far off-frame the remaining ramp is
nearly flat.

This is now a testable hypothesis rather than a mystery. For a railed frame from
each run, pull the 2D image and check:

- is the frame effectively blank (no source above background)?
- is there a bright source hard against one edge, or none at all?

`tests/make_fixtures.py` shows the read-only pattern for pulling frames off E:.
Record the answer in `KNOWN_UNUSUAL` in `tests/test_survey.py`.

---

## 6. Clear the two remaining test failures

Both are acceptance tests that could not pass without a reprocess.

### `test_no_ambiguous_bare_named_mirrors`

8 stale bare-named CSVs in `reprocess_output/`:

```
fanoff5237  fanoff  laserdaytest  minutely
morning5237  noon5237  overnight  warming
```

Reprocessing rewrites them but **nothing deletes the stale ones** — that is a
manual step. The underlying collision is worse than previously recorded: 10 bare
names map to 26 distinct datasets.

```
minutely     -> 7 datasets      morning5237  -> 6 datasets
fanoff5237   -> 3 datasets      fanoff, laserdaytest, noon5237,
                                overnight, warming, postspie,
                                postspiegenie -> 2 each
```

`20260814/postspie` and `20260814_data/postspie` are **different files with
different content** — do not assume a `_data` suffix is cosmetic.

### `test_summary_agreement_with_truth_does_not_regress`

6 runs read `DIVERGENT_WITH_FAILURES`. Re-check after the reprocess. Expect some
to clear; the genie ones probably will not, and if section 4 confirms the laser
diagnosis they *should not* — they should be recorded in `EXPECTED_VERDICT` with
the reason, so the red is documented rather than ambient.

**Do not lower `FWHM_MAX_PX` to 500 to make this pass.** That buys a green test
by discarding ~26,000 frames to agree with a threshold tuned on a camera whose
spot is 12 px wide. See `CSV_SCHEMA.md` on why FWHM is the wrong instrument.

---

## 7. Then, and only then — the deferred work

In rough priority order.

1. **`output_dir=` on `process_fits_run`**, which unblocks F1 (worker-count
   invariance: same data at 1, 2, 8 workers must give byte-identical CSVs).
   Monkeypatching cannot substitute — `from fits_reprocess import OUTPUT_DIR`
   creates a second binding in the parallel module, and Windows *spawns* workers
   that re-import the original value.
2. **Make `fit_ok` account for `on_bound`.** This is the principled fix that
   lets the arbitrary `FWHM_MAX_PX` go. Changes `fit_ok` semantics, so it wants
   its own reprocess — or a decision that old and new `fit_ok` need not be
   comparable.
3. **Rename the misleading columns** (see `CSV_SCHEMA.md`). Same constraint:
   schema change, wants a reprocess.
4. **Retune or retire `n_peaks`.** After this run you can measure directly how
   often it went blind during a crossing by comparing it against
   `two_comp_sep` — currently unknowable. The root defect is that
   `PEAK_MIN_DISTANCE_PX = 50` is a fixed pixel count against widths that vary
   30× across the corpus; scaling it with the fitted sigma is the obvious fix
   but is untested.
5. **Per-frame two-dot sweep.** The survey samples one frame per run and
   detection is frame-dependent — postwinterbreak's sources are 105 px apart in
   its median frame but 357–411 px apart in frames 1/501/5001 — so the run-level
   census undercounts. `two_component` is now per-frame, so this is just an
   aggregation over the new CSVs.

---

## Traps that have already cost time

- **`reprocess_output/*_frames.csv` is not a per-run index.** 20 runs collapse
  into 8 bare-named files, each holding whichever run finished last. Any glob
  over that directory silently reads one arbitrary run as if it were all of
  them. Read per-run CSVs from E:, or use `compare_pipelines._frames_files()`,
  which handles this correctly. Ad-hoc globs do not.
- **`E:` is read-only** by standing instruction, except during an explicitly
  authorised reprocess.
- **`fit_ok=False` in pre-2026-09-04 CSVs is stale** — 23 of 26 previously
  failing fixture frames now fit cleanly.
- **The summary CSV's "good frames" is post-FWHM-gate**, not fit-level `fit_ok`.
  That is why it reports 50 runs with failures when only 13 have `fit_ok=False`
  rows.
- **Bash heredocs mangle `\n` inside Python string literals** in this
  environment. Use the Write/Edit tools for Python containing escapes.
- **Don't compare `resid` across runs.** Normalise by `noise`. springgenie's
  raw residual is 100× allmetal's while fitting a 200× brighter source.
