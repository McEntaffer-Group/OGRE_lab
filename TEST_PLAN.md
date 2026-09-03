# Test plan — fit correctness, parallel equivalence, truth agreement

Companion to `FIT_DIAGNOSIS_HANDOFF.md`. Nothing here is written yet; this is the
proposal. Tests marked **DISCUSS** need a decision before I build them.

Revision 4. Changes from r3 are listed in §13. **§1 corrects a contaminated table
in the handoff** — read that first if you read nothing else.

---

## 0. Ground rules

- **Runner:** `pytest` 9.1.1, installed into `ReverseTelescopeDot/.venv` this
  session. From the repo root:
  `D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -v`
- **`-X utf8` is required** — default cp1252 crashes on non-ASCII output.
- **Offline by default.** Everything except two tests runs with `E:` unmounted.
  Those are `@pytest.mark.needs_data` and auto-skip.
- **Nothing is resolution-locked.** Every threshold derives from `profile.size`;
  synthetic tests sweep 64 → 2592 px.
- **Written red first.** No change to `fits_reprocess.py` until you've seen the
  red run.

---

## 1. Correction to the handoff: §1.5's table is contaminated

Found while grounding the collision tests you asked for. This changes the scope
estimate, so it goes first.

`reprocess_output/` holds **76 bare-named** `{runname}_frames.csv` files and only
**2 date-prefixed** ones. Eight of those bare files belong to runnames used by
more than one run:

```
minutely      x7   20250922, 20250923, 20250925, 20250926, 20251016, 20251017, 20251020
morning5237   x6   20250922, 20250926, 20250929, 20251017, 20251020, 20251022
fanoff5237    x3   20250925, 20250926, 20250929
fanoff        x2   noon5237 x2   overnight x2   laserdaytest x2   warming x2
```

**20 runs collapse into 8 CSV files.** Each holds whichever run finished last.

Handoff §4.1's railing scan globbed `reprocess_output/*_frames.csv`, so its table
in §1.5 has three affected rows — `minutely` (1563 rows, 14.7% railed),
`overnight` (1090 rows, 16.1%), `warming` (79 rows, 8.9%) — each describing *one
arbitrary run of several*. The other 12 colliding runs were never scanned at all.

So **"21 of ~78 runs are affected" is an undercount**, and the three rows above
are not aggregates. Not a reason to redo the diagnosis — the root cause in §1.1–1.3
was established on allmetal frames directly and stands — but the *scope* number
should be treated as a floor, and the fixture extractor below must read per-run
CSVs from their run directories on `E:`, never from `reprocess_output/`.

`compare_pipelines.py` is **not** affected: `_frames_files()` (line 122) already
detects bare names, flags them legacy, and drops any superseded by a dated mirror.
The contamination was in the handoff's ad-hoc glob, not in the tooling.

### Is the clobbering still live?

Mostly no, and the fix is already in the code. `run_key()` (`fits_reprocess.py:371`)
and `_mirror_for_run()` (line 394) date-prefix every mirrored artifact, and the
docstring at line 374 names this exact problem. Primary outputs are path-scoped
(`{date}_data/{runname}/`) so they never collided.

The residue is that **stale bare-named files are never cleaned up**. After
reprocessing, `reprocess_output/` will hold both `minutely_frames.csv` (stale,
one arbitrary run) and seven `2025xxxx_minutely_frames.csv` files. Anything that
globs the directory naively double-counts. That's what §8 tests.

---

## 2. The golden set — one frame from every run

Your call, and it's the right one: a six-frame fixture set proves the fix on the
frames I already knew about, which is circular. The broad set makes the suite a
survey.

### What gets extracted

Per run, from the run's own directory on `E:` (path-scoped, so collisions are
impossible):

1. **One healthy frame** — the `fit_ok==True` row at the *median* index, not
   frame 1. First frames are often atypical (settling, focus, lamp warm-up) and
   frame 1 is exactly where the position baseline is anchored, so it is the worst
   choice for "typical".
2. **One failure frame** — the first `fit_ok==False` row, for the 50 runs that
   have any. 44 runs have zero failures and contribute only a healthy frame.

From `all_runs_summary_reprocess.csv`: 94 runs, 50 with ≥1 failure, 503,521 total
frames. So roughly **144 frames → 288 profiles**.

### Size

~1800 samples × 8 bytes ≈ 14 KB per profile, so ~4 MB uncompressed, ~2.5–3.5 MB
in a compressed `.npz`. Chunky for a git commit but not unreasonable, and it is
written once. **I'd keep float64** — float32 would halve it but introduces ~1e-7
relative error on values of magnitude 3e4, and D1 asserts μ agreement to 0.01 px.
Not worth risking the tightest assertion in the suite to save 1.5 MB.

If you'd rather not commit 3 MB, the alternative is to keep the six-frame `.npz`
committed and generate the broad set on demand into the scratchpad, marking the
survey tests `needs_data`. That trades the offline guarantee for repo size. My
recommendation is to commit it — the whole point of the survey is that it runs
every time, not only when `E:` happens to be mounted.

### `tests/make_fixtures.py`

Reuses the pipeline's own loaders, so fixtures are provably what production sees
rather than a reimplementation of the `np.flip`.

```python
"""Extract 1-D profiles from every run into committed .npz files so the suite
runs with E: unmounted. Run once, with E: mounted; re-run when runs are added.

Two frames per run: the median healthy frame, and the first failure frame for
runs that have one. Per-run CSVs are read from the run directory, never from
reprocess_output/ -- 20 runs share 8 bare-named files there (see TEST_PLAN section 1).
"""
import numpy as np, pandas as pd
from pathlib import Path
import fits_reprocess as fr

E = Path("E:/Reverse Telescope Test Data")
DEST = Path(__file__).parent / "fixtures"

def _profiles(run_dir: Path, is_image: bool, filename: str):
    frames = run_dir if is_image else run_dir / f"{run_dir.name}_fits"
    load = fr._load_image_frame if is_image else fr._load_fits_frame
    img = load(frames / filename).astype(np.float64)
    return np.sum(img, axis=0), np.sum(img, axis=1)

healthy, failing, index = {}, {}, []
for run_dir, is_image in fr.discover_runs(E):          # exact name TBD, see below
    key = fr.run_key(run_dir)
    csv = fr.output_dir_for(run_dir, is_image) / f"{run_dir.name}_frames.csv"
    if not csv.exists():
        print(f"  skip {key}: no per-run frames CSV"); continue
    d = pd.read_csv(csv)
    d = d[~d.filename.str.contains(r"_(?:FFT|FWHM|position)", na=False)]   # plot PNGs

    ok = d[d.fit_ok == True]
    if len(ok):
        row = ok.iloc[len(ok) // 2]
        px, py = _profiles(run_dir, is_image, row.filename)
        healthy[f"{key}_px"], healthy[f"{key}_py"] = px, py

    bad = d[d.fit_ok == False]
    if len(bad):
        row = bad.iloc[0]
        px, py = _profiles(run_dir, is_image, row.filename)
        failing[f"{key}_px"], failing[f"{key}_py"] = px, py

    index.append({"run_key": key, "is_image": is_image, "n": len(d),
                  "n_ok": len(ok), "n_bad": len(bad)})

np.savez_compressed(DEST / "healthy.npz", **healthy)
np.savez_compressed(DEST / "failing.npz", **failing)
pd.DataFrame(index).to_csv(DEST / "index.csv", index=False)
```

Split into two `.npz` files so a test that only wants healthy frames doesn't
load 3 MB of failures. `index.csv` is committed too — it records how many frames
each run had when the fixture was cut, which is what lets a later drift be
noticed.

**One thing I need to check before writing this:** `fits_reprocess.py` has
`_discover_image_runs`; I saw it referenced at line 604 but have not confirmed
there is a matching public FITS-run discovery function or a combined one. If
there isn't, the extractor walks `E:` with the documented layout from the module
docstring (lines 5–6). Minor, but I don't want to write `fr.discover_runs` in the
plan as if I'd verified it — I haven't.

### Named anchors, in a third `anchors.npz`

The survey set can only assert invariants — it has no known-good answers. These
frames do, so they stay separate and get asserted against exact values:

| anchor | role | known value |
|---|---|---|
| `allmetal` f011, f018 | railing anchors | μ_y = 535.2666 / 530.6110, σ_y = 4.4065 / 5.3530 |
| `allmetal` f001, f101, f501 | healthy | fit centroid = 2D dot to **1.3 px** — the single-dot control for G4 |
| `postwinterbreak` f001, f501, f5001 | **two-dot positive** | 2D dot (1439, 626); fit lands at (1403, 998), 375 px away |
| `frosty` f001 | **two-dot**, was mislabelled | 2D dot (1191, 572), HWHM 3 px; fit at (1384, 1042), 508 px away |
| `genieshots` mid-run | **saturated**, third category | peak = 255, radial profile flat past 36 px |

Three `postwinterbreak` frames rather than one because the dot drifts 288 px in x
across the run — a single frame would let a detector pass by accident on a lucky
position. `allmetal`'s healthy frames double as G4's negative control, which is
what makes the separation threshold defensible rather than guessed.

---

## 3. Group G — the survey

This is where "I'd rather have the test fail and know which data sets are
unusual" gets implemented. Everything here is **parametrized by run**, so pytest
names the offender: `test_...[20260624_bridgetstatic]`.

### G1 — every run's healthy frame fits without railing

```python
RUN_KEYS = sorted(pd.read_csv(FIXTURES / "index.csv").run_key)

@pytest.mark.parametrize("run_key", RUN_KEYS)
def test_healthy_frame_does_not_rail(healthy, run_key):
    for axis in ("px", "py"):
        prof = healthy.get(f"{run_key}_{axis}")
        if prof is None:
            continue
        amp, mu, sigma, offset = fr._fit_one_profile(prof)
        assert np.isfinite(mu), f"{run_key} {axis}: fit declined on a frame marked fit_ok"
        assert 1e-3 < mu < prof.size - 1e-3, f"{run_key} {axis}: mu={mu:.3f} on a bound"
        assert sigma < prof.size / 2 - 1e-6, f"{run_key} {axis}: sigma={sigma:.3f} on a bound"
```

A frame the pipeline already marked `fit_ok=True` must not be railed. Given §1.5,
this should be red for a good number of runs today — and the parametrization
turns that into a **list of exactly which**, which is the survey result you want.

### G2 — every failure is classifiable

The one that handles "some of these failures are real."

`compare_pipelines.classify_failures()` (line 147) already labels rows with a
failure mode and is documented as knowing the `profile.size/2` sigma bound. Rather
than assert failures must fit — they mustn't, some are genuine — this asserts
every failure lands in a **known category**:

```python
@pytest.mark.parametrize("run_key", RUN_KEYS_WITH_FAILURES)
def test_failure_frame_has_a_known_cause(failing, run_key):
    reasons = []
    for axis in ("px", "py"):
        prof = failing.get(f"{run_key}_{axis}")
        if prof is None:
            continue
        reasons.append(_classify(prof, fr._fit_one_profile(prof)))
    assert reasons, f"{run_key}: marked as failing but no fixture profile"
    assert not any(r == "unclassified" for r in reasons), \
        f"{run_key}: failure with no known cause -- {reasons}"
```

with a small local taxonomy, reusing `classify_failures`' logic:

| label | meaning | verdict |
|---|---|---|
| `dot_off_frame` | no peak above background; μ pinned at an edge with amp ≈ 0 | **real** — the dot left the sensor |
| `railed_sigma` | σ on the `size/2` bound with a lower-residual alternative available | **bug** — this is what we're fixing |
| `railed_mu` | μ on 0 or `size` | **bug** |
| `saturated` | profile clipped at the ADC ceiling | **real** |
| `no_convergence` | every seed raised | needs a look |
| `unclassified` | none of the above | **fails the test** |

Note `two_component` is deliberately **absent** from this table — a two-dot frame
does not fail, it converges cleanly onto the wrong dot. It is a G1 problem, not a
G2 one. See §4.

The point is exactly your framing: the test doesn't decide whether a failure is
acceptable, it decides whether we *understand* it. An unclassified failure is a
dataset nobody has looked at, and it fails loudly with the run name attached.

### G3 — the failure census doesn't regress

One aggregate test rather than 94, so the overall picture is a single number:

```python
def test_failure_census_does_not_regress():
    """Count failure frames by category across every run. Categories that are
    bugs must reach zero; real ones are recorded, not asserted away."""
    census = collections.Counter(...)
    assert census["railed_sigma"] == 0, f"still railing: {census}"
    assert census["railed_mu"]    == 0, f"still railing: {census}"
    assert census["unclassified"] == 0, f"unexamined failures: {census}"
    # dot_off_frame / saturated deliberately unasserted -- printed for the record
```

### Handling the ones that are genuinely odd

When G1 or G2 fails for a run that turns out to be legitimately weird, it gets an
entry in a single visible table with a **reason**, not a silent skip:

```python
KNOWN_UNUSUAL = {
    # run_key: (test, reason, date characterised)
    # e.g. "20260624_bridgetstatic": ("G1", "dot parked at frame edge all run", "2026-09-03"),
}
```

Starts empty. Entries are added only after actually looking at the data, and each
one is a documented finding rather than a suppression. Using `pytest.xfail(strict=True)`
means an entry that later starts *passing* also fails the suite — so the table
can't rot.

---

## 4. Two-dot runs — a silent wrong answer, not a failure

Some runs legitimately contain **two** dots: a laser-pointer spot at ~500 FWHM
(common on the genies) and the real dot at ~5 FWHM. Not expected to be easily
fixable; the goal here is that it stops being invisible.

### The big dot wins, and it isn't close

The profile is a sum along an axis, so a 2D dot of peak amplitude `A` and widths
`σ` contributes a 1D Gaussian of amplitude ∝ `A·σ_perp·√(2π)` and width `σ_along`.

| | FWHM | σ | 1D amplitude | 1D width | least-squares leverage |
|---|---|---|---|---|---|
| laser | ~500 | ~212 | ∝ `A_L · 212` | 212 | ~10⁴ × |
| real dot | ~5 | ~2.1 | ∝ `A_D · 2.1` | 2.1 | 1 × |

At equal 2D peak brightness the laser is ~100× taller *and* ~100× wider in the
summed profile — about **10⁴× the weight** in the sum of squares. Even a laser
100× dimmer per pixel still carries ~100× the leverage. A single-Gaussian
least-squares fit has no mechanism to prefer the small dot. Your assumption
holds, with a large margin.

### Why that's worse than railing

A railed fit is loud. This one converges cleanly:

```
mu     = the laser's centroid        (not on a bound)
sigma  ~ 212                         (not on a bound, if size/2 > 212)
resid  = low
fit_ok = True                        (FWHM 500 passes the 1000 gate)
```

**Every test in groups B, C, G2 and R passes on it.** The run's reported position
tracks the laser rather than the telescope dot, and nothing in the pipeline says
so. It lands in G1's healthy set and passes.

### Verified: `n_peaks` cannot detect it as configured

`_count_peaks` (`fits_reprocess.py:209`) uses `PEAK_HEIGHT_FRAC=0.50` and
`PEAK_MIN_DISTANCE_PX=50`. It reports `n_peaks > 1` for **100% of frames in 22
runs**, which is not two dots — it is noise ripple. From a `lasersimultaneous`
x-profile, frame 1:

```
peak at 1301.0  height=20920 (100.0% of max)  width=394.9   <- the real component
peak at 1281.0  height=20548 ( 98.2% of max)  width=  1.6   <- noise on the plateau
peak at 1261.0  height=19714 ( 94.2% of max)  width=  1.2   <- noise
peak at 1241.0  height=18784 ( 89.8% of max)  width=  1.3   <- noise
FIT -> mu=1309.31 sigma=171.87 fwhm=404.7
```

A broad plateau's noise crosses the 50%-of-range threshold repeatedly, and 50 px
spacing is small next to a 390 px component. So the existing column is not a
usable two-dot flag, and a test built on it would be built on sand.

### CONFIRMED on real data — `20260105/postwinterbreak`

You named it, and it settles the question. Frame 1, 2D blob analysis:

```
thr=60% of peak -> 1 blob    area=    43 px  equiv-diam=  7.4  peak=123  at (x=1439, y= 625)
thr=20% of peak -> 2 sources area=136205 px  equiv-diam=416.4  peak= 71  at (x=1399, y= 997)
```

Two sources, 372 px apart in y. The small one is **brighter per pixel** (123 vs
71) and ~3000× smaller in area. The fit lands on the big one every time:

```
frame     fit centroid            2D small dot        separation
    1     x=1402.9 y= 997.9       x=1439 y= 626       375 px
  501     x=1405.3 y= 989.1       x=1418 y= 578       412 px
 5001     x=1418.0 y= 933.0       x=1234 y= 627       357 px
 9001     x=1425.7 y= 891.4       x=1151 y= 545       424 px
```

σ≈165 throughout, i.e. the reported FWHM ~386 is the *blob*. **The run's reported
position is not the dot's position.** Reported y drifts 998→891 (−107 px) while
the real dot's y drifts 626→545 (−81 px) — correlated but not equal, so the
published drift for this run is a blend of two sources.

### My proposed detector was wrong — corrected

The r3 detector (residual bump, global MAD, `n_sigma=5`) **fails on this run**:
3.8σ at frame 1 and 0.6–0.7σ at frames 501/5001. Two reasons:

1. The broad blob is **not Gaussian**, so single-Gaussian model mismatch leaves
   structured residuals with MAD ≈ 250 counts — far above the dot's ~600-count
   bump. Global MAD is the wrong noise scale.
2. I searched a **fixed** window around frame 1's dot position. This run measures
   *drift*; the dot moves 288 px in x and 82 px in y over the run. My "misses"
   were my error, not the detector's.

**What works: high-pass the residual, then take its global argmax.** Subtracting a
41 px boxcar of the residual from itself suppresses the broad model mismatch and
keeps the narrow dot:

```
frame  | 2D small dot (argmax) | 1D high-pass argmax | fit centroid
     1 | x=1439 y= 626         | x=1439 y= 627       | x=1402.9 y= 997.9
   501 | x=1418 y= 578         | x=1419 y= 578       | x=1405.3 y= 989.1
  2001 | x=1402 y= 564         | x=1403 y= 565       | x=1406.6 y= 993.0
  5001 | x=1234 y= 627         | x=1235 y= 627       | x=1418.0 y= 933.0
  9001 | x=1151 y= 545         | x=1152 y= 544       | x=1425.7 y= 891.4
 11001 | x=1370 y= 595         | x=1369 y= 595       | x=1407.8 y= 929.1
```

**The 1D high-pass recovers the 2D dot position to ±1 px in every frame**, across
the whole run. That's better than I expected and it matters: the detector needs
no access to the 2D image, so `_fill_fit_results(out, px, py)` keeps its signature.

### The discriminator: centroid vs high-pass separation

Significance alone doesn't separate cleanly (frosty 5.5σ vs allmetal 4.3σ). The
**separation between the fitted centroid and the high-pass peak** does, and
`allmetal` — a known clean single-dot run — is the control that proves it:

| run | fit centroid | 2D brightest compact source | separation | σ | verdict |
|---|---|---|---|---|---|
| `allmetal` | (662.7, 610.8) | (664, 611), HWHM 6 px | **1.3 px** | 5.6 | single dot ✓ |
| `postwinterbreak` | (1398.7, 948.9) | (1439, 626), HWHM ~4 px | **375 px** | 159 | **two dots** |
| `frosty` | (1383.6, 1041.5) | (1191, 572), HWHM 3 px | **508 px** | 167 | **two dots** |
| `genieshots` | (1180.8, 1037.7) | (1074, 727), HWHM >36 px, **peak=255** | 365 px | 212 | saturated, different |

For a single-dot run the fit centroid *is* the dot, to 1.3 px. That's the
invariant worth testing, and it's resolution- and brightness-independent.

### Two corrections this forces

**1. `frosty` is a two-dot run, and my C5 anchor was invalid.** Its 2D brightest
source is compact — HWHM 3 px, radial mean falling 94→45→11→3 counts over 9 px —
sitting 508 px from where the fit puts it. So handoff §1.5's classification of
`frosty`, `postwinterbreak`, `realwinterbreak` and `newprimary` as "genuinely
broad spots — those are real, not failures" is **wrong for at least the two I
checked in 2D**. Those σ≈165 fits are not broad dots; they are a fit locked onto a
diffuse companion while a compact dot sits elsewhere in the frame.

C5 (`test_broad_but_real_spot_is_kept`) therefore loses its anchor. Over-correction
is still worth guarding, but R3's *synthetic* broad spot does it without depending
on a run whose nature I'd have to keep re-verifying. **C5 is dropped; R3 covers it,
and `frosty` is re-tagged as a two-dot fixture.**

**2. `genieshots` is a third category, not a two-dot run.** Its brightest pixel is
255 — saturated — and its radial profile stays flat out past 36 px. Its 46.9σ
high-pass signal comes from saturation structure, not a compact companion. Any
taxonomy needs `saturated` separate from `two_component`.

### The detector, as corrected

A single Gaussian fit to the laser leaves the narrow dot as a **compact,
localized, significant positive residual** — a few px wide, many σ above the
residual noise. That is cheap to detect and doesn't require fitting two Gaussians.

```python
BOXCAR = 41          # px; wide vs a ~5px dot, narrow vs a ~390px blob

def compact_peak(prof, popt):
    """Locate the narrowest strong feature the Gaussian fit did not explain.

    High-passing the residual (subtract a 41px boxcar of itself) suppresses the
    broad model mismatch that swamps a plain-residual test -- on postwinterbreak
    that mismatch has MAD ~250 counts against a ~600-count dot. Validated against
    2D: reproduces the true dot position to +/-1 px on every frame checked.
    Returns (index, significance)."""
    x = np.arange(prof.size, dtype=np.float64)
    resid = prof - fr.gaussian(x, *popt)
    hp = resid - uniform_filter1d(resid, BOXCAR)
    mad = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    i = int(hp.argmax())
    return i, float(hp[i] / mad) if mad > 0 else np.inf

def second_component(px, py, popt_x, popt_y, min_sigma=5.0, min_sep_frac=1.0):
    """True when the fit sits far from the frame's strongest compact source.

    Separation, not significance, is the discriminator: allmetal (single dot)
    gives 4.3 sigma but only 1.3 px of separation, while frosty gives 5.5 sigma
    and 508 px. Threshold is in units of the fitted sigma, so it is
    resolution-independent."""
    ix, sx = compact_peak(px, popt_x)
    iy, sy = compact_peak(py, popt_y)
    if max(sx, sy) < min_sigma:
        return None
    sep = np.hypot(popt_x[1] - ix, popt_y[1] - iy)
    scale = max(popt_x[2], popt_y[2])
    return (ix, iy, sep, max(sx, sy)) if sep > min_sep_frac * scale else None
```

MAD rather than plain RMS specifically so the bump doesn't inflate the noise
estimate it's being measured against. Both thresholds are now **calibrated
against real data** — `postwinterbreak` as the positive case, `allmetal` as the
negative — rather than guessed.

### G4 — two-component frames are flagged, not silently fit

```python
@pytest.mark.parametrize("run_key", RUN_KEYS)
def test_two_component_frames_are_flagged(healthy, run_key):
    """A frame with a second compact component must be detected. This does NOT
    assert the fit is wrong -- with a laser present the broad fit is arguably
    correct -- only that we know the second dot is there."""
    for axis in ("px", "py"):
        prof = healthy.get(f"{run_key}_{axis}")
        if prof is None:
            continue
        popt = fr._fit_one_profile(prof)
        hit = second_component(prof, popt)
        if hit is not None:
            assert run_key in KNOWN_TWO_DOT, (
                f"{run_key} {axis}: undeclared second component at index {hit[0]}, "
                f"width {hit[1]:.1f}px, {hit[2]:.1f} sigma")
```

`KNOWN_TWO_DOT` starts empty and gets filled from the first run of this test.
Same discipline as `KNOWN_UNUSUAL`: the survey tells us which runs they are, then
each entry is a recorded finding rather than a suppression.

### Still unverified

Calibrated on `postwinterbreak` (positive) and `allmetal` (negative), and checked
in 2D on `frosty` and `genieshots`. **Not** verified in 2D: `realwinterbreak`
(26.3σ high-pass, 333 px separation) and `lowhumidity` (6.6σ, 1421 px) look like
two-dot runs on the 1D signature alone, and `newprimary` is on handoff §1.5's
"broad but real" list but I haven't touched it. G4 across the full fixture set is
what settles these — that's the survey doing its job rather than me guessing.

`lasersimultaneous` and `bridgetstatic`, checked earlier, are *not* clean two-dot
examples: the first shows one broad component plus speckle (227 blobs at 50% of
peak), the second has one compact ~10 px blob and rails `mu_x` on the bound in
45.7% of frames, which is the ordinary railing bug.

### Not proposed: actually fitting two Gaussians

It would work — an 7-parameter two-component model would separate them — but it
changes the model for every run to serve a handful, adds parameters that can
themselves rail, and needs a rule for which component is "the" dot. Given
"I don't expect this to be an easily fixable problem", flagging is the right
scope. Worth revisiting only if the two-dot runs turn out to be numerous.

---

## 5. Group H — run-name collision

Your second ask. Four tests, three of them instant and offline.

### H1 — `run_key` distinguishes same-named runs

```python
def test_run_key_distinguishes_same_name_on_different_dates():
    a = Path("E:/data/20250922_data/minutely")
    b = Path("E:/data/20251016_data/minutely")
    assert fr.run_key(a) != fr.run_key(b)
    assert fr.run_key(a) == "20250922_minutely"
```

### H2 — mirrored artifacts can't overwrite each other

The actual clobbering mechanism, exercised end to end against the filesystem:

```python
def test_mirrored_artifacts_do_not_clobber(tmp_path, monkeypatch):
    out = tmp_path / "out"; out.mkdir()
    monkeypatch.setattr(fr, "OUTPUT_DIR", out)
    for date in ("20250922", "20251016"):
        run_dir = tmp_path / f"{date}_data" / "minutely"
        run_dir.mkdir(parents=True)
        src = run_dir / "minutely_frames.csv"
        src.write_text(f"marker\n{date}\n")
        fr._mirror_for_run(src, run_dir)
    written = sorted(p.name for p in out.glob("*_frames.csv"))
    assert written == ["20250922_minutely_frames.csv", "20251016_minutely_frames.csv"]
    for date in ("20250922", "20251016"):
        assert date in (out / f"{date}_minutely_frames.csv").read_text()
```

The content assertion matters as much as the name one: two distinct files with
the *same* content would mean the second write clobbered the first upstream.

### H3 — no artifact name is claimed by two runs

The general invariant, and the one that catches a *future* collision rather than
the eight we know about:

```python
@pytest.mark.needs_data
def test_no_two_runs_claim_the_same_mirrored_name():
    seen = {}
    clashes = []
    for run_dir, is_image in _all_runs(E_DRIVE):
        for name in fr._run_output_names(run_dir.name):
            mirrored = f"{fr.date_prefix(run_dir)}_{name}"
            if mirrored in seen:
                clashes.append(f"{mirrored}: {seen[mirrored]} vs {run_dir}")
            seen[mirrored] = run_dir
    assert not clashes, "\n".join(clashes)
```

Should pass today — the date prefix makes it safe. It's a guard against someone
reintroducing a bare name, which is precisely how this happened the first time.

### H4 — no stale bare-named mirrors survive

The live residue from §1, as an acceptance test:

```python
COLLIDING = {"minutely", "morning5237", "fanoff5237", "fanoff",
             "noon5237", "overnight", "laserdaytest", "warming"}

def test_no_ambiguous_bare_named_mirrors():
    """A bare {runname}_frames.csv for a colliding runname holds whichever run
    finished last. Anything globbing reprocess_output/ then reads one arbitrary
    run as if it were all of them -- which is how the handoff's section 1.5 table
    ended up describing 1 of 7 'minutely' runs."""
    stale = [p.name for p in OUT_DIR.glob("*_frames.csv")
             if not _KEYED_RE.match(p.stem) and p.stem[:-7] in COLLIDING]
    assert not stale, (
        "ambiguous bare-named mirrors still present:\n  " + "\n  ".join(sorted(stale)))
```

**Red today** — all eight are present. Goes green when reprocessing writes dated
mirrors *and* the stale bare files are removed. That cleanup doesn't exist yet;
it's a small addition to the reprocess entry point, and worth doing because
nothing currently deletes them.

---

## 6. Group R — resolution independence

From your point that nothing may assume 1024 px. The bug is size-scaled:
`_estimate_sigma` clips at `size/4` and the σ bound is `size/2`, so a 64 px frame
fails differently from a 2592 px one.

`tests/synth.py`:

```python
def synth(size, mu_frac=0.523, sigma=4.0, amp=926.0, offset=21000.0,
          noise=60.0, seed=0):
    """A realistic profile at arbitrary resolution. Defaults match measured
    allmetal values: 4% dot contrast on a 21000 pedestal, RMS noise 60 (handoff
    section 2 -- NOT the invented 174 that drove the over-built rewrite)."""
    x = np.arange(size, dtype=np.float64)
    mu = mu_frac * size
    prof = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset
    return (prof + np.random.default_rng(seed).normal(0, noise, size) if noise
            else prof), mu
```

```python
SIZES = [64, 128, 256, 512, 1024, 2592]

@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sigma", [2.0, 5.0, 20.0])
def test_estimator_does_not_saturate_at_any_resolution(size, sigma):
    if sigma * 8 > size:
        pytest.skip(f"sigma={sigma} does not fit in a {size}px frame")
    prof, _ = synth(size, sigma=sigma)
    est = fr._estimate_sigma(prof, np.arange(size, dtype=np.float64), float(prof.argmax()))
    assert not np.isclose(est, size / 4.0), f"pinned at ceiling {size/4:.1f}"
    assert sigma / 3.0 < est < sigma * 3.0, f"est={est:.2f} vs true {sigma}"
```

The factor-of-3 band is loose on purpose: the estimator seeds the ladder, it
isn't the answer. §5.1's known 2× overshoot is fine and the residual-scored
ladder cleans it up. Tighter assertions would re-invite the version you rejected.

**R2** fits a known spot at every size (μ to 0.5 px, σ to 15%). **R3** fits a spot
occupying a fifth of the frame — at 2592 px that's FWHM ≈ 1221, past
`FWHM_MAX_PX` — and asserts the *fit*, saying nothing about `fit_ok`. See §9.

---

## 7. Master table

| # | Test | Why | Today | Cost |
|---|---|---|---|---|
| **A1** | `test_fixtures_present_and_sane` | Stale/corrupt `.npz` fails clearly, not confusingly | passes | instant |
| **G1** | `test_healthy_frame_does_not_rail` | **Survey.** Every run's typical frame, parametrized by run | **RED (many)** | ~30 s |
| **G2** | `test_failure_frame_has_a_known_cause` | Every failure classifiable; real ones allowed, unexamined ones not | **RED (some)** | ~20 s |
| **G3** | `test_failure_census_does_not_regress` | The whole picture as one number | **RED** | ~5 s |
| **G4** | `test_two_component_frames_are_flagged` | Laser + real dot. A *silent wrong answer* no other test catches. Calibrated on postwinterbreak vs allmetal — §4 | **RED (≥2 runs)** | ~20 s |
| **H1** | `test_run_key_distinguishes_same_name_on_different_dates` | Collision, unit level | passes | instant |
| **H2** | `test_mirrored_artifacts_do_not_clobber` | Collision, filesystem level | passes | instant |
| **H3** | `test_no_two_runs_claim_the_same_mirrored_name` | Catches a *future* collision. `needs_data` | passes | ~5 s |
| **H4** | `test_no_ambiguous_bare_named_mirrors` | The 8 stale files from §1 | **RED** | instant |
| **R1** | `test_estimator_does_not_saturate_at_any_resolution` | Saturation is size-scaled | **RED** | ~1 s |
| **R2** | `test_fit_recovers_known_spot_at_any_resolution` | No hardcoded resolution | **RED** | ~2 s |
| **R3** | `test_very_broad_spot_is_fit_not_railed` | A dot may legitimately exceed FWHM 1000 | **RED** | ~1 s |
| **B1** | `test_estimator_is_not_saturated` | R1 on real anchor profiles | **RED** | instant |
| **B2** | `test_estimator_survives_degenerate_input` | flat/zeros/hot pixel/edge/two peaks, swept over size | passes | instant |
| **C1** | `test_railed_frames_recover_truth` | **Headline.** Frames 11 & 18 → μ≈535.267 / 530.611 | **RED** | ~1 s |
| **C2** | `test_no_parameter_lands_on_a_bound` | §1.5's detector on the anchors | **RED** | ~1 s |
| **C3** | `test_ladder_returns_lowest_residual_fit` | **Causal test** — would have caught the `break` bug | **RED** | ~5 s |
| **C4** | `test_fitted_peak_matches_profile_peak` | §1.2 as physics: `amp + offset ≈ max` | **RED** | ~1 s |
| ~~C5~~ | ~~`test_broad_but_real_spot_is_kept`~~ | **Dropped** — frosty turned out to be two-dot, so the anchor was invalid. R3 covers over-correction synthetically. See §4 | — | — |
| **C6** | `test_fit_is_deterministic` | Same profile twice → identical bits | passes | instant |
| **D1** | `test_matches_truth_lm_on_golden_profiles` | Per-profile truth agreement, offline | **RED** | ~2 s |
| **D2** | `test_summary_agreement_with_truth_does_not_regress` | Summary line vs summary line | **RED** | ~2 s |
| **E1** | `test_parallel_imports_identical_fit_symbols` | No silent fork of the fit math | passes | instant |
| **E2** | `test_serial_and_parallel_workers_agree` | The 4 duplicated profile sites. `needs_data` | passes | ~5 s |
| **E3** | `test_worker_failure_returns_empty_row` | Corrupt file → `_empty_row`, not a dead pool | passes | instant |
| **E4** | `test_frame_ordering_is_stable` | Ties reachable in 31 runs — §7 | **RED** | instant |
| **F1** | `test_worker_count_does_not_change_output` | *The* parallelism test | **DISCUSS** | ~10 s |

25 tests, ~60 s for the suite (G1/G2 dominate). 14 expected red.

---

## 8. The frame-order sort is unstable

`csv_to_dotplots.py:83,85` sorts with an explicit `kind="stable"`. The four
pipeline sorts (`fits_reprocess.py:713,776`, `fits_reprocess_parallel.py:529,632`)
do not, and pandas defaults to quicksort. Ties are reachable in **31 of ~78 runs**:
22 from the `frame_num == -1` plot PNGs that commit `ccc3796` denylisted (stale
CSVs, all `fit_ok=False`, benign), and ~9 from genuine duplicate frame numbers
among real frames, which do sort non-deterministically.

You endorsed `kind="stable"`, so this becomes one helper replacing four
open-coded sorts:

```python
def _order_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by frame number, preserving discovery order among ties.
    Stable is required: 31 of ~78 runs have duplicate frame_num values."""
    return df.sort_values("frame_num", kind="stable").reset_index(drop=True)
```

E4 tests it directly with 50 rows sharing a frame number.

---

## 9. Truth agreement

**D1** compares each golden profile against truth's recipe from
`dot_movie-Copy3.ipynb` cell 16 — unbounded `curve_fit` (LM), `p0` sigma 5 — and
asserts μ to 0.01 px, σ to 1%.

Caveat I'll report rather than paper over: truth's `p0 σ=5` is a guess that suits
these spots. `frosty` and `postwinterbreak` are now known to be two-dot frames, so
truth and reprocess may agree with each other while **both sit on the wrong
component**. D1 asserts agreement with truth, not correctness — those are different
claims, and on a two-dot run only the first is available. Where they disagree, R3
and G4 carry the meaning.

**D2** is the summary-line comparison you meant, and you were right that it's
instant. `compare_pipelines.py` already implements it (`TRUTH_PAIRS`, `_reldiff`,
`report_truth()`, thresholds at lines 60–80). Current state, ~2 s:

```
28 overlapping runs:   MATCH 17 | NEAR 2 | DIVERGENT_WITH_FAILURES 6 | DIVERGENT_TRUTH_GATE 3
```

The six failures are the acceptance criterion for this whole effort:

```
 20260130_newsecondary   3999 total   1124 good   worst reldiff 1.636  y position
     20260213_allmetal   3922 total   3273 good   worst reldiff 0.977  FWHM y std
   20260219_hotandcold  15806 total  15789 good   worst reldiff 0.712  FWHM y std
  20260306_springbreak  14174 total  14166 good   worst reldiff 0.652  FWHM y std
   20260320_statictest   5660 total   5593 good   worst reldiff 0.935  FWHM y std
20260420_zoeysecondary  10156 total   9350 good   worst reldiff 0.954  FWHM x std
```

The three `DIVERGENT_TRUTH_GATE` runs are recorded as expected, not bugs — see §9.

Two caveats: D2 stays red until the pipeline is fixed **and** the affected runs
are reprocessed (it's the end-to-end acceptance test, not a unit test); and
`report_truth()` prints and writes a diagnostics CSV as a side effect, so it may
want a `quiet=True` or a compute/report split.

---

## 10. The FWHM gate — dropped as a test target

r1 proposed asserting no correct fit lands between FWHM 500 and 1000. **Dropped.**
Your reasoning changes the framing: a dot can legitimately be FWHM > 500 and
there's no principled reason it can't exceed 1000. The goal is that `curve_fit`
looks at the real image and picks an intelligent guess — the gate isn't the
mechanism, so testing its exact value encodes an arbitrary constant as physics.

R3 replaces it by asserting a huge spot is *fit* correctly while saying nothing
about `fit_ok`. The real rejection criterion should be `on_bound` plus residual,
not a magic ceiling (§1.5, open item #3).

The data makes one question concrete: `20260306_springgenie` has **12,920 of
14,150 frames** cut by truth's `fwhm_max=500` and kept by reprocess. If those are
real broad dots, truth is discarding good data and the divergence is truth being
wrong. Worth looking at one springgenie frame before touching either constant.

---

## 11. Still to discuss

### F1 — worker-count invariance

```python
@pytest.mark.parametrize("workers", [1, 4])
def test_worker_count_does_not_change_output(tmp_path, monkeypatch, workers):
    run_dir = _synthesize_fits_run(tmp_path, n=8, size=128)
    monkeypatch.setattr(frp, "WORKERS", workers)
    monkeypatch.setattr(fr, "OUTPUT_DIR", tmp_path / "out")
    frp.process_fits_run(run_dir, make_plots=False, make_movie=False)
    ...compare the two CSVs bit for bit...
```

1. `OUTPUT_DIR` is a module constant and `fits_reprocess_parallel.py:449` calls
   `OUTPUT_DIR.mkdir()` directly. Monkeypatching works but is fragile; an optional
   `output_dir=` parameter on `process_fits_run` is cleaner. **Signature change to
   working code — want it?**
2. `_choose_pipeline` (line 362) selects shmem vs stream by image size and free
   RAM, so a tiny run always takes one path. Covering both means forcing the mode.
3. The shmem worker needs `_SHMEM_ARR` from the pool initialiser, so it can't be
   called directly the way E2 calls `_fit_worker_fits`.

My lean: do F1 with whichever pipeline the tiny run picks, don't force both, skip
a separate shmem test. E1 already guarantees both paths call the same fit
function, which is where the risk is.

### Fixture size

3 MB committed, or generate-on-demand and mark the survey `needs_data`? I
recommend committing — a survey that only runs when `E:` is mounted isn't a
regression gate.

---

## 12. Deliberately not tested

- **The over-built estimator (§5.2).** Rejected; not resurrecting it. Its useful
  parts survive as R1 and B2.
- **Plot and movie output.** Brittle image comparison, and the bug isn't there.
- **`_count_peaks`.** Informational; doesn't gate `fit_ok` (`fits_reprocess.py:210`).
- **Timing / speedup.** The 3.2× from §4.11 is a result, not an invariant.

---

## 13. Changes from revision 3

| r3 | r4 | Why |
|---|---|---|
| — | §4 **verified on postwinterbreak** | Two sources 372 px apart; fit lands on the blob every frame. Confirmed in 2D |
| r3 detector: residual bump, fixed window, 5σ | high-pass + global argmax | r3 version scored 0.6σ on real data. High-pass tracks the dot to ±1 px |
| significance as discriminator | centroid-vs-peak **separation** | allmetal 4.3σ/1.3px vs frosty 5.5σ/508px — significance alone does not separate |
| C5 anchored on frosty | **C5 dropped** | frosty is two-dot; handoff §1.5 mislabelled it "broad but real" |
| failures taxonomy would cover it | `two_component` deliberately **not** a failure label | It converges cleanly; it's a G1 silent-wrong-answer, not a G2 failure |
| — | verified `n_peaks` is unusable | Reports >1 on 100% of frames in 22 runs — noise ripple, not dots |

Earlier, r2 → r3:

| r2 | r3 | Why |
|---|---|---|
| 6 golden frames | ~144 frames, one healthy + one failure per run | Six frames I already knew about proves the fix circularly |
| — | Group G, parametrized by run | "I'd rather have the test fail and know which data sets are unusual" |
| — | Group H, 4 collision tests | Your ask; the mechanism is real (20 runs → 8 files) |
| — | §1, handoff correction | §1.5's table describes 1 of 7 `minutely` runs; scope is a floor |
| fixtures from `reprocess_output/` | fixtures from per-run dirs on `E:` | That directory is where the collision damage is |

## 14. Build order

1. Confirm the run-discovery API, then `make_fixtures.py`; run it, commit —
   **while `E:` is mounted**
2. `conftest.py`, `synth.py`, A1
3. Groups H, R, B, C, E1, E3 — offline, written red
4. Group G — the survey. **This is the interesting output:** a named list of every
   run that rails, fails inexplicably, or carries a second dot
   - G4's thresholds want tuning against a frame you know has both dots. Until
     then it runs with the physics-derived defaults and I report what it finds
     rather than claiming the numbers are calibrated
5. **Show you the red run + the survey list.** No pipeline changes yet
6. D1, reporting honestly what the two-dot anchors do under truth's `p0 σ=5`
7. `_order_frames()` + E4
8. Decide F1
9. Apply the §5.1 fit rewrite; suite goes green except D2/H4
10. Reprocess + clean stale bare mirrors; D2 and H4 go green

Steps 1–6 touch nothing in the pipeline.
