# `{run}_frames.csv` — column reference

26 columns as of 2026-09-04. The last 10 are new and were **never written by any
run before this date**, so any CSV lacking them predates the fix and holds
old-fit output.

Written by `fits_reprocess._fill_fit_results`. Column order is fixed by
`_CSV_COLS`; `_empty_row` must define every one of them or
`tests/test_parallel.py::test_empty_row_covers_every_csv_column` fails.

---

## A warning about names

Several names in this pipeline describe the *library parameter that was passed*
or the *author's intent*, rather than the effect on the data. Two of these bit
us in 2026-09:

| name | sounds like | actually does |
|---|---|---|
| `fit_ok` | the fit converged | FWHM landed inside a range gate |
| `PEAK_MIN_DISTANCE_PX` | peaks must be ≥50 px apart | **deletes** the smaller of any pair closer than 50 px |
| `_estimate_sigma` (pre-fix) | measures the width | returned `size/4` on nearly every frame |
| `n_peaks_x` | how many peaks are there | how many survived two lossy filters on a summed profile |

Read the definition, not the name. A rename pass is worth doing but changes the
schema for every consumer, so it wants to ride a reprocess.

---

## Identity

| column | meaning |
|---|---|
| `frame_num` | frame index parsed from the filename |
| `filename` | basename of the source frame |
| `timestamp` | parsed from the filename; falls back to mtime for images that carry none |

## Fit parameters

All in **pixels**. No arcsec conversion happens here — downstream code looks up
the per-run scale in `pixel_scales.csv`.

| column | meaning |
|---|---|
| `mu_x`, `mu_y` | fitted centroid on each summed profile |
| `sigma_x`, `sigma_y` | fitted Gaussian width |
| `fwhm_x`, `fwhm_y` | `sigma * 2.3548` |
| `amp_x`, `amp_y` | height **above** `offset`, not absolute peak |
| `offset_x`, `offset_y` | fitted background pedestal |

`amp` being above-offset is load-bearing. Seeding it as the absolute peak was
one of the two original bugs — the starting model predicted roughly double the
data.

## Gates and counts

### `fit_ok` — a RANGE GATE, not a convergence flag

```python
FWHM_MIN_PX < fwhm_x < FWHM_MAX_PX and (same for y) and mu_x, mu_y finite
```

Consequences, all verified:

- A fit that converged cleanly onto a **genuinely broad spot** is `fit_ok=False`.
- A fit **railed onto a bound** inside the gate is `fit_ok=True`.
- A **completely blank frame** is `fit_ok=True`. `curve_fit` converges on the
  degenerate `amp≈0` solution, leaving `mu=1e-10` and `sigma=1.0` — both on
  their lower bounds — and FWHM 2.355 sits inside the gate. `mu=1e-10` then
  enters the position series as a real centroid at pixel 0. Pinned by
  `test_blank_frame_is_fit_ok_but_flagged_on_bound`.

Use these instead, depending on what you actually mean:

| you mean | read |
|---|---|
| the fit converged | `mu_x` / `mu_y` are non-NaN |
| the fit hit a wall | `on_bound_x` / `on_bound_y` |
| the fit is good | `resid_x / noise_x` (see below) |
| there are two sources | `two_component`, plus `resid/noise` for overlaps |

### `FWHM_MAX_PX = 1000` is not resolution-independent

`sigma` is already bounded at `size/2` by the fit, so the largest FWHM that can
ever be returned is `1.177 × size`:

```
frame size    max possible FWHM    is the 1000 cap reachable?
       480                565.2    NO  - filters nothing
       640                753.5    NO  - filters nothing
      1216               1431.7    yes
      2592               3051.8    yes
```

On runs at 640 px or smaller the cap is dead. On a 2592 px frame it cuts at 39%
of frame width. Same number, wildly different strictness.

Measured on the pre-fix CSVs, it removed 18,216 of 515,020 frames (3.54%) —
concentrated in `bridgetstatic` (95.7%), `newsecondary` (71.9%), `overnight`
(67.6%), which are the same runs at the top of the bound-railing list. **The cap
has mostly been a proxy for "this fit railed."** `on_bound` now measures that
directly.

It is nevertheless still doing real work, because `on_bound` is *recorded* but
does not *gate*. Today the cap is the only thing keeping a sigma-railed fit out
of the position series. Do not remove it until `fit_ok` accounts for `on_bound`.

### `n_peaks_x` / `n_peaks_y` — informational, gates nothing

Local maxima found by `scipy.signal.find_peaks` on the **summed 1D profile**,
independent of the fit, with two filters: a height floor at 50% of the profile's
full range, and `distance=50`.

Four measured blind spots:

```
two equal dots, walked together        two dots 300px apart, one fading
  separation  n_peaks                     amp_b/amp_a  n_peaks
         100        2                            0.45        2
          49        2                            0.30        1  <- deleted
          40        1  <- deleted               0.10        1  <- deleted
           0        1  <- deleted

ONE source, widened                     two dots stacked on an axis
       sigma  n_peaks                     diagonal pair  n_x=2 n_y=2
         4.3        1                     stacked in x   n_x=1 n_y=2
        60.0        3  <- spurious
       250.0       11  <- spurious
```

So it goes blind exactly during a crossing, deletes companions fainter than half
the range, invents peaks on any broad spot, and cannot see a pair aligned on the
other axis. Against the independent detector it scores 15 agreements, 9 false
alarms (7 of them `*genie`, i.e. the broad-plateau case), 3 misses.

Kept unchanged for continuity with historical runs. `two_component` and
`resid/noise` are the columns to actually trust.

---

## Evidence columns (added 2026-09-04)

These are byproducts of fitting — the bounds size the fit, the residual scores
the seeds — and were previously computed and discarded. **They cannot be
backfilled onto old CSVs**; producing them requires re-fitting, which is a
reprocess.

### `nx`, `ny` — profile lengths, i.e. image width and height

Every bound derives from these. Without them a railed fit is only detectable by
*inferring* frame size from the largest value observed, which is fragile.

### `resid_x`, `resid_y` — RMS residual of the returned fit

Same counts as the data. This is the number that chooses between seeds; it was
previously thrown away, which is why diagnosing a bad fit meant re-running the
fit by hand.

**Do not compare `resid` between runs.** springgenie's source is ~200× brighter
than a real dot, so its residual is large in counts while being small relative
to its own amplitude. Normalise it.

### `on_bound_x`, `on_bound_y` — the fit came to rest against its own box

True when `mu < 1e-3`, `mu > size - 1e-3`, `sigma < 1 + 1e-6`, or
`sigma > size/2 - 1e-6`. NaN (declined) reports False — a declined fit is not a
railed fit.

**This, not a large sigma, is the reliable failure signature.** Several runs have
genuinely broad spots (sigma ~160, amp ≈ offset, centroid nowhere near an edge)
and those are real data. A parameter pinned to the edge of the box it was allowed
to search is unambiguous.

Two distinct signatures, and they mean different things:

- **upper bound** (`sigma → size/2`, or `mu → size`) — the fit exploded, or the
  source's centre has left the detector. Reproduced synthetically: once the true
  centre crosses the frame edge, `mu` pins to `size` and `sigma` collapses
  (250 → 189 → 118 → 70) as the optimiser bends a flank into the visible ramp.
- **lower bound** (`sigma → 1.0`) — the fit collapsed onto something narrower
  than a pixel. Produced by blank frames, and also by a source so far off-frame
  that the remaining ramp is nearly flat (at 200% off-frame the fit returned
  `mu=826`, a plausible **interior** position, with `sigma=1.00`).

`20251014_minutelyovernight`, `20251029_longweekend` and `20251106_laser` rail
on the lower bound. Hypothesis, not yet confirmed in 2D: the source left the
frame.

### `noise_x`, `noise_y` — noise where the fit says there is no source

MAD of the residual outside 3σ of the fitted centre, scaled to compare with a
standard deviation. NaN when fewer than 30 samples survive.

**NaN is a diagnosis, not a gap**: it means the source fills the frame with no
background left to measure — "the source is larger than the detector."

### `resid / noise` — the scale-free goodness of fit

This is the most useful derived quantity in the file. Measured calibration:

```
allmetal, springbreak    single dot, model correct        1.0 - 1.2
postwinterbreak          two dots, fit on the wrong one   5.5 - 6.4
genieshots, springgenie  truncated skewed laser          19   - 57
```

~1.0 means the residual **is** the noise and there is nothing left to explain.
It reads correctly regardless of source brightness, which raw `resid` does not:
a 160× brighter source has a far larger residual while fitting just as well.

Suggested reading, to be re-derived from the full corpus after the next
reprocess rather than trusted as gospel:

| ratio | reading |
|---|---|
| < 2 | model is right, fit is noise-limited |
| 2 – 10 | something unmodelled — second source, mild asymmetry |
| > 10 | a single Gaussian does not describe this data |
| NaN | source fills the frame |

### `two_component`, `two_comp_sep`

`two_component` is True when the high-passed residual contains a compact feature
that is both ≥5σ significant and separated from the fitted centroid by more than
`max(2 × sigma, 0.05 × frame)`. `two_comp_sep` records the separation in px even
when the verdict is False-adjacent, **so the floors can be retuned from the CSVs
without another reprocess**.

Two floors are needed and neither alone works: a sigma multiple alone
false-positives on narrow dots with slightly non-Gaussian cores, while a
threshold loose enough to suppress those rejects `postwinterbreak`, whose real
dot sits only 2.4σ from the blob it is confused with.

**Known blind spot, by construction:** it requires separation, so it cannot see
two sources that overlap or are mid-crossing — the same case that defeats
`n_peaks`. `resid/noise` does see that case, because overlapping sources still
inflate the residual. The two columns are complementary; neither replaces the
other. Pinned by
`test_two_component_is_blind_to_overlapping_pairs_but_resid_is_not`.

---

## What a healthy row looks like

```
fit_ok        True
on_bound_*    False, both axes
resid/noise   ~1.0, both axes
two_component False
n_peaks_*     1   (but see the blind spots above)
mu            comfortably interior, far from 0 and from nx/ny
sigma         anything — width alone proves nothing
```
