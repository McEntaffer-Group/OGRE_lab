# Test plan — fit correctness, parallel equivalence, truth agreement

Companion to `FIT_DIAGNOSIS_HANDOFF.md`. Nothing here is written yet; this is the
proposal. Tests marked **DISCUSS** need a decision before I build them.

Revision 2 — reworked after review. Changes from r1 are listed in §9.

---

## 0. Ground rules

- **Runner:** `pytest` (installed into `ReverseTelescopeDot/.venv` this session,
  9.1.1). From the repo root:
  `D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -v`
- **`-X utf8` is required** — default cp1252 crashes on non-ASCII output.
- **Offline by default.** Everything except two tests runs with `E:` unmounted.
  Those two are `@pytest.mark.needs_data` and auto-skip.
- **Nothing is resolution-locked.** Every threshold derives from `profile.size`.
  Synthetic tests sweep 64 → 2592 px. See §2.
- **Speed target:** whole suite under ~15 s. There is no slow tier any more.
- **Written red first.** I will not touch `fits_reprocess.py` until you've seen
  the red run.

### `tests/conftest.py`

```python
import numpy as np, pytest
from pathlib import Path

E_DRIVE = Path("E:/Reverse Telescope Test Data")

def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: requires the E: data drive")

def pytest_collection_modifyitems(config, items):
    if E_DRIVE.exists():
        return
    skip = pytest.mark.skip(reason="E: data drive not mounted")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)

@pytest.fixture(scope="session")
def profiles():
    """Golden 1-D profiles from real frames. See tests/make_fixtures.py."""
    return dict(np.load(Path(__file__).parent / "fixtures" / "profiles.npz"))
```

### `tests/synth.py` — one synthetic-profile helper, used everywhere

```python
import numpy as np

def synth(size, mu_frac=0.523, sigma=4.0, amp=926.0, offset=21000.0,
          noise=60.0, seed=0):
    """A realistic profile at arbitrary resolution.

    Defaults match measured allmetal values: 4% dot contrast on a 21000 pedestal,
    RMS noise 60 (handoff section 2 -- NOT the invented 174 that drove the
    over-built rewrite). mu is a *fraction* of size so the spot sits off-centre
    at every resolution.
    """
    x = np.arange(size, dtype=np.float64)
    mu = mu_frac * size
    prof = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset
    if noise:
        prof = prof + np.random.default_rng(seed).normal(0, noise, size)
    return prof, mu
```

---

## 1. The fixtures

`tests/make_fixtures.py` — run once with `E:` mounted, output committed. A
generator, not a test. Stores **only profiles**, never expected fit values, so
the truth comparison re-derives truth live instead of trusting a baked number.

```python
"""Extract 1-D profiles from real frames into a committed .npz so the suite runs
with E: unmounted. Re-run only if the anchor frames change."""
import glob
from pathlib import Path
import numpy as np
from astropy.io import fits

E = Path("E:/Reverse Telescope Test Data")
ANCHORS = [
    # key,             fits dir,                                  0-based index
    ("allmetal_f011", E / "20260213_data/allmetal/allmetal_fits",  10),  # rails in Y
    ("allmetal_f018", E / "20260213_data/allmetal/allmetal_fits",  17),  # rails in Y
    ("allmetal_f001", E / "20260213_data/allmetal/allmetal_fits",   0),  # healthy
    ("allmetal_f101", E / "20260213_data/allmetal/allmetal_fits", 100),  # healthy
    ("allmetal_f501", E / "20260213_data/allmetal/allmetal_fits", 500),  # healthy
    ("frosty_f001",   E / "20251210_data/frosty/frosty_fits",       0),  # broad but REAL
]

out = {}
for key, d, i in ANCHORS:
    path = sorted(glob.glob(str(d / "*.fits")))[i]
    with fits.open(path) as h:
        img = np.flip(h[0].data, axis=(0, 1)).astype(np.float64)
    out[f"{key}_px"] = np.sum(img, axis=0)
    out[f"{key}_py"] = np.sum(img, axis=1)

dest = Path(__file__).parent / "fixtures" / "profiles.npz"
dest.parent.mkdir(exist_ok=True)
np.savez_compressed(dest, **out)
print(f"wrote {dest}  ({dest.stat().st_size/1024:.0f} KB, {len(out)} profiles)")
```

**Why these six.** Frames 11 and 18 are the handoff's verified railing anchors
with known-good answers (μ_y = 535.267 / 530.611, σ_y = 4.406 / 5.353). Frames 1,
101, 501 were characterised as healthy in §1.1. `frosty` frame 1 is the
false-positive guard: from `reprocess_output/frosty_frames.csv` it fits at
μ=(1385.6, 1033.3), σ=(173.3, 171.9), amp 20228 vs offset 34123 — a genuinely
broad spot nowhere near a bound. A "fix" that rejects it is over-corrected.

These two runs also happen to be different resolutions (allmetal 1280×1024,
frosty ≥2048 rows), which helps — but real data covers no *small* frames at all,
so resolution independence is covered synthetically in group R rather than here.

---

## 2. Group R — resolution independence

Your point, and it's a real gap in r1. The bug itself is resolution-scaled:
`_estimate_sigma` clips at `profile.size / 4` and the σ bound is
`profile.size / 2`, so a 64 px frame and a 2592 px frame fail *differently*. No
test may hardcode 1024, and no test may assume a minimum size.

```python
SIZES = [64, 128, 256, 512, 1024, 2592]
```

### R1 — estimator never saturates, at any size

```python
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("sigma", [2.0, 5.0, 20.0])
def test_estimator_does_not_saturate_at_any_resolution(size, sigma):
    if sigma * 8 > size:
        pytest.skip(f"sigma={sigma} spot does not fit in a {size}px frame")
    prof, _ = synth(size, sigma=sigma)
    est = fr._estimate_sigma(prof, np.arange(size, dtype=np.float64),
                             float(prof.argmax()))
    ceiling = size / 4.0
    assert not np.isclose(est, ceiling), (
        f"estimator pinned at its ceiling {ceiling:.1f} (size={size}, true sigma={sigma})")
    assert sigma / 3.0 < est < sigma * 3.0, (
        f"est={est:.2f} not within 3x of true sigma={sigma} (size={size})")
```

The factor-of-3 band is deliberately loose. The estimator's job is to *seed the
ladder*, not to be the answer — §5.1's known 2× overshoot is fine and the
residual-scored ladder cleans it up. Asserting tighter would re-invite the
over-built version you rejected.

### R2 — the fit recovers a known spot, at any size

```python
@pytest.mark.parametrize("size", SIZES)
def test_fit_recovers_known_spot_at_any_resolution(size):
    true_sigma = max(2.0, size / 200.0)          # scales with the frame
    prof, true_mu = synth(size, sigma=true_sigma)
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    assert mu    == pytest.approx(true_mu,    abs=0.5)
    assert sigma == pytest.approx(true_sigma, rel=0.15)
```

### R3 — a legitimately huge dot is not mangled

Directly from your "a dot can be FWHM greater than 500, and I don't see why it
can't be greater than 1000". A spot occupying a fifth of the frame is physically
fine and the fit must return it, not rail.

```python
@pytest.mark.parametrize("size", [512, 1024, 2592])
def test_very_broad_spot_is_fit_not_railed(size):
    true_sigma = size / 5.0                      # FWHM ~ 0.47 * size
    prof, true_mu = synth(size, sigma=true_sigma, amp=20000.0, offset=34000.0)
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    assert mu    == pytest.approx(true_mu,    rel=0.02)
    assert sigma == pytest.approx(true_sigma, rel=0.15)
    assert sigma < size / 2.0 - 1e-6, "sigma railed on its upper bound"
```

At size=2592 this spot has FWHM ≈ 1221 px — above `FWHM_MAX_PX`. The test
asserts the **fit**, deliberately saying nothing about `fit_ok`. See §6.

---

## 3. Master table

| # | Test | File | Why it exists | Status today | Cost |
|---|---|---|---|---|---|
| **A1** | `test_fixtures_present_and_sane` | `test_fixtures.py` | Catches a stale/corrupt `.npz` before it confuses everything downstream | passes | instant |
| **R1** | `test_estimator_does_not_saturate_at_any_resolution` | `test_resolution.py` | Saturation is size-scaled; 64 px fails differently from 2592 px | **RED** | ~1 s |
| **R2** | `test_fit_recovers_known_spot_at_any_resolution` | `test_resolution.py` | End-to-end fit, no hardcoded resolution anywhere | **RED** | ~2 s |
| **R3** | `test_very_broad_spot_is_fit_not_railed` | `test_resolution.py` | A huge-but-real dot must fit, even past `FWHM_MAX_PX` | **RED** | ~1 s |
| **B1** | `test_estimator_is_not_saturated` | `test_estimator.py` | Same as R1 but on the real golden profiles | **RED** | instant |
| **B2** | `test_estimator_survives_degenerate_input` | `test_estimator.py` | flat / zeros / hot pixel / edge peak / two peaks, swept over sizes | passes | instant |
| **C1** | `test_railed_frames_recover_truth` | `test_fit.py` | **The headline.** Frames 11 & 18 → μ≈535.267 / 530.611 | **RED** | ~1 s |
| **C2** | `test_no_parameter_lands_on_a_bound` | `test_fit.py` | §1.5's reliable detector, on all golden profiles | **RED** | ~1 s |
| **C3** | `test_ladder_returns_lowest_residual_fit` | `test_fit.py` | **The causal test** — would have caught the original `break` bug | **RED** | ~5 s |
| **C4** | `test_fitted_peak_matches_profile_peak` | `test_fit.py` | §1.2 as physics: `amp + offset ≈ profile.max()` | **RED** | ~1 s |
| **C5** | `test_broad_but_real_spot_is_kept` | `test_fit.py` | Over-correction guard on the real frosty frame | passes | ~1 s |
| **C6** | `test_fit_is_deterministic` | `test_fit.py` | Same profile twice → identical bits | passes | instant |
| **D1** | `test_matches_truth_lm_on_golden_profiles` | `test_truth.py` | Per-profile truth agreement, offline, vs unbounded LM `p0σ=5` | **RED** | ~2 s |
| **D2** | `test_summary_agreement_with_truth_does_not_regress` | `test_truth.py` | **The comparison you meant** — summary line vs summary line | **RED** | ~2 s |
| **E1** | `test_parallel_imports_identical_fit_symbols` | `test_parallel.py` | Prevents a silent fork of the fit math (§1.7) | passes | instant |
| **E2** | `test_serial_and_parallel_workers_agree` | `test_parallel.py` | The 4 duplicated profile sites. `needs_data` | passes | ~5 s |
| **E3** | `test_worker_failure_returns_empty_row` | `test_parallel.py` | Corrupt file → `_empty_row`, not a dead pool | passes | instant |
| **E4** | `test_frame_ordering_is_stable` | `test_parallel.py` | Ties in `frame_num` are reachable in 31 runs — §5 | **RED** | instant |
| **F1** | `test_worker_count_does_not_change_output` | `test_pipeline.py` | *The* parallelism-correctness test | **DISCUSS** | ~10 s |

RED = expected to fail against today's `fits_reprocess.py`, by design. 12 of 18.

---

## 4. Full code — the small ones

### A1 — fixtures sane

No minimum-size assumption (r1 had `size >= 512`; dropped).

```python
def test_fixtures_present_and_sane(profiles):
    assert len(profiles) == 12, "expected 6 frames x 2 axes"
    for key, prof in profiles.items():
        assert prof.ndim == 1 and prof.size > 0, key
        assert np.all(np.isfinite(prof)), key
        assert prof.max() > prof.min(), f"{key} is flat"
```

### B1 — estimator not saturated, on real profiles

```python
@pytest.mark.parametrize("key", ["allmetal_f001_py", "allmetal_f011_py",
                                 "allmetal_f018_py", "allmetal_f101_py"])
def test_estimator_is_not_saturated(profiles, key):
    prof = profiles[key]
    est = fr._estimate_sigma(prof, np.arange(prof.size, dtype=np.float64),
                             float(prof.argmax()))
    ceiling = prof.size / 4.0
    assert est < 0.5 * ceiling, f"pinned at ceiling ({est:.1f} vs {ceiling:.0f})"
```

Today: returns exactly 256.0 for every 1024 px y-profile.

### B2 — degenerate inputs, swept over size

```python
def _degenerate(size):
    hot = np.full(size, 21000.0); hot[size // 2] = 5e4
    return {
        "flat":      np.full(size, 21000.0),
        "zeros":     np.zeros(size),
        "hot_pixel": hot,
        "edge_peak": synth(size, mu_frac=0.002, sigma=4.0, noise=0.0)[0],
        "two_peaks": synth(size, mu_frac=0.3, sigma=4.0, noise=0.0)[0]
                   + synth(size, mu_frac=0.7, sigma=4.0, noise=0.0)[0],
    }

@pytest.mark.parametrize("size", [64, 512, 2592])
@pytest.mark.parametrize("name", ["flat", "zeros", "hot_pixel", "edge_peak", "two_peaks"])
def test_estimator_survives_degenerate_input(size, name):
    prof = _degenerate(size)[name]
    est = fr._estimate_sigma(prof, np.arange(size, dtype=np.float64),
                             float(prof.argmax()))
    assert np.isfinite(est) and 0 < est <= size, f"{name}@{size}: est={est}"
```

Deliberately weak. These inputs have no right answer; the test only pins down
"doesn't raise, doesn't return something that poisons the ladder."

### C1 — the headline

```python
# (key, expected mu, expected sigma) — handoff section 5.1, cross-checked there
# against an unbounded LM fit to 4 decimals.
RAILED_ANCHORS = [
    ("allmetal_f011_py", 535.2666, 4.4065),
    ("allmetal_f018_py", 530.6110, 5.3530),
]

@pytest.mark.parametrize("key,mu_want,sigma_want", RAILED_ANCHORS)
def test_railed_frames_recover_truth(profiles, key, mu_want, sigma_want):
    amp, mu, sigma, offset = fr._fit_one_profile(profiles[key])
    assert mu    == pytest.approx(mu_want,    abs=0.05)
    assert sigma == pytest.approx(sigma_want, rel=0.02)
```

Today: μ=0.000/σ=355.26 for frame 11, μ=410.4/σ=512.0 for frame 18.

### C2 — nothing on a bound

```python
def test_no_parameter_lands_on_a_bound(profiles):
    failures = []
    for key, prof in profiles.items():
        amp, mu, sigma, offset = fr._fit_one_profile(prof)
        if not np.isfinite(mu):
            continue                      # a declined fit is not a railed fit
        if mu < 1e-3 or mu > prof.size - 1e-3:
            failures.append(f"{key}: mu={mu:.4f} on bound [0, {prof.size}]")
        if sigma > prof.size / 2.0 - 1e-6:
            failures.append(f"{key}: sigma={sigma:.4f} on bound {prof.size/2:.0f}")
    assert not failures, "\n".join(failures)
```

Collects every failure instead of dying on the first. Bounds derive from
`prof.size`, so this is resolution-agnostic by construction.

### C4 — amp is above-offset

```python
@pytest.mark.parametrize("key", ["allmetal_f001_py", "allmetal_f101_py",
                                 "allmetal_f501_py", "frosty_f001_py"])
def test_fitted_peak_matches_profile_peak(profiles, key):
    prof = profiles[key]
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    assert amp + offset == pytest.approx(prof.max(), rel=0.05)
```

§1.2 stated as physics rather than an implementation check: the converged model's
peak height must match the data's.

### C6 — determinism

```python
def test_fit_is_deterministic(profiles):
    prof = profiles["allmetal_f011_py"]
    assert fr._fit_one_profile(prof) == fr._fit_one_profile(prof)
```

### E1 — no silent fork

```python
def test_parallel_imports_identical_fit_symbols():
    import fits_reprocess as fr, fits_reprocess_parallel as frp
    assert frp._fill_fit_results is fr._fill_fit_results
    assert frp._empty_row        is fr._empty_row
    assert frp._load_fits_frame  is fr._load_fits_frame
    assert frp._load_image_frame is fr._load_image_frame
```

Verified: `fits_reprocess_parallel.py:97-108` imports all four.

### E3 — worker failure is contained

```python
def test_worker_failure_returns_empty_row(tmp_path):
    import fits_reprocess_parallel as frp
    bogus = tmp_path / "nope0001 26-01-01 00-00-00.fits"
    bogus.write_bytes(b"not a fits file")
    row, img, px, py = frp._fit_worker_fits(str(bogus), need_image=False)
    assert row["fit_ok"] is False
    assert np.isnan(row["mu_x"]) and np.isnan(row["mu_y"])
    assert row["frame_num"] == 1          # from the name, not the payload
    assert (img, px, py) == (None, None, None)
```

### E4 — stable frame ordering

You endorsed `kind="stable"`, so this is now a straight code change plus its
regression test rather than an open question.

```python
def test_frame_ordering_is_stable():
    """Rows with equal frame_num keep discovery order through the pipeline sort."""
    rows = [fr._empty_row(f"f{i:03d}.fits", frame_num=7, timestamp=f"t{i}")
            for i in range(50)]
    df = fr._order_frames(pd.DataFrame(rows))
    assert list(df["filename"]) == [f"f{i:03d}.fits" for i in range(50)]
```

This needs one three-line helper in `fits_reprocess.py`, replacing the four
open-coded sorts:

```python
def _order_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by frame number, preserving discovery order among ties.
    Stable is required: 31 of ~78 runs have duplicate frame_num values."""
    return df.sort_values("frame_num", kind="stable").reset_index(drop=True)
```

Applied at `fits_reprocess.py:713,776` and `fits_reprocess_parallel.py:529,632`.
That's a smaller change than r1's source-regex hack, and gives a real thing to
test. See §5.

---

## 5. The frame-order sort is unstable

Not in the handoff; found while grounding this plan. `csv_to_dotplots.py:83,85`
sorts with an explicit `kind="stable"`. The four pipeline sorts do not:

```
fits_reprocess.py:713           df.sort_values("frame_num")
fits_reprocess.py:776           df.sort_values("frame_num")
fits_reprocess_parallel.py:529  df.sort_values("frame_num")
fits_reprocess_parallel.py:632  df.sort_values("frame_num")
```

pandas defaults to quicksort, which is not stable, so ties land in arbitrary
order. Ties are reachable in **31 of ~78 runs**, from two causes:

1. **`frame_num == -1`, 22 runs, always exactly 3 rows.** I checked these: they
   are the self-ingested plot PNGs (`warming_FFT_reprocess.png` etc.) that commit
   `ccc3796` added the denylist for. Those CSVs are **stale**, written before the
   fix. All three carry `fit_ok=False` so they never reach the baseline. Benign,
   and reprocessing clears them.
2. **Genuine duplicate `frame_num >= 0`, 2–3 rows in ~9 runs.** Real frames with
   colliding numbers. These do sort non-deterministically.

Whether cause 2 can move a result depends on the position baseline being taken
from the first surviving frame (`csv_to_dotplots.py:114`) — a tie at the *front*
of the ordering is the dangerous case, a tie in the middle is cosmetic. I have
not checked whether any affected run has its tie at frame 1, so I'd call this
latent rather than live. The fix costs one keyword either way.

---

## 6. The FWHM gate — dropped as a test target

r1 proposed `test_truth_gate_choice_is_irrelevant`, asserting no correct fit
lands between FWHM 500 and 1000. **Dropped.** Your reasoning is right and it
changes the framing: a dot can legitimately be FWHM > 500, and there's no
principled reason it can't exceed 1000 either. The goal is that `curve_fit` looks
at the real image and picks an intelligent guess — the gate is not the mechanism,
so building tests around its exact value encodes an arbitrary constant as if it
were physics.

What replaces it:

- **R3** asserts a genuinely huge spot is *fit* correctly, and says nothing about
  `fit_ok`. At size=2592 that spot has FWHM ≈ 1221, past `FWHM_MAX_PX` — the fit
  must still be right whether or not the gate later keeps the frame.
- The real rejection criterion should be **`on_bound` plus residual**, not a magic
  FWHM ceiling (handoff §1.5, open item #3). Recording `on_bound_x/y` and
  `resid_x/y` in `{run}_frames.csv` makes `FWHM_MAX_PX` mostly vestigial.

This does raise a live question, which the data now makes concrete — see D2
below: `20260306_springgenie` has **12920 of 14150 frames** that truth's
`fwhm_max=500` cuts and reprocess keeps. If those are real broad dots, truth is
the one throwing away good data, and the "divergence" there is truth being wrong.
I'd want to look at a springgenie frame before touching either constant.

---

## 7. Snippets — the bigger ones

### C3 — ladder returns the lowest-residual fit

The test that would have caught the original bug, written
implementation-agnostically: it doesn't care how many rungs exist or how they're
seeded, only that no alternative seed beats the returned answer.

```python
SEEDS = [2., 5., 10., 25., 50., 100., 200.]

def _rms(resid):
    return float(np.sqrt(np.mean(resid ** 2)))

@pytest.mark.parametrize("key", sorted(GOLDEN_KEYS))
def test_ladder_returns_lowest_residual_fit(profiles, key):
    prof = profiles[key]
    x = np.arange(prof.size, dtype=np.float64)
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    got = _rms(fr.gaussian(x, amp, mu, sigma, offset) - prof)

    bounds = ([0., 0., 1., -np.inf], [np.inf, float(prof.size), prof.size / 2., np.inf])
    for s in SEEDS:
        if s >= prof.size / 2.0:
            continue                       # seed outside the bound, not a fair rival
        try:
            p, _ = curve_fit(fr.gaussian, x, prof, bounds=bounds, maxfev=5000,
                             p0=[prof.max() - np.median(prof), float(prof.argmax()),
                                 s, float(np.median(prof))])
        except Exception:
            continue
        alt = _rms(fr.gaussian(x, *p) - prof)
        assert alt >= got * 0.999, (
            f"{key}: seed sigma={s} reaches RMS {alt:.1f}, "
            f"beating the returned fit's {got:.1f}")
```

On frame 11 today the returned fit scores RMS 93.6 while seed σ=5 reaches 61.2 —
§1.3 restated as an assertion.

### D1 — per-profile truth agreement, offline

Truth's recipe from `dot_movie-Copy3.ipynb` (§1.6): **unbounded** `curve_fit` →
Levenberg–Marquardt, `p0` sigma fixed at 5.

```python
def _truth_fit(prof):
    """dot_movie-Copy3.ipynb cell 16: unbounded curve_fit, p0 sigma = 5."""
    x = np.arange(prof.size, dtype=np.float64)
    p, _ = curve_fit(fr.gaussian, x, prof, maxfev=10000,
                     p0=[prof.max(), float(prof.argmax()), 5.0, float(np.median(prof))])
    amp, mu, sigma, offset = p
    return amp, mu, abs(sigma), offset

@pytest.mark.parametrize("key", sorted(GOLDEN_KEYS))
def test_matches_truth_lm_on_golden_profiles(profiles, key):
    prof = profiles[key]
    _, mu_t, sigma_t, _ = _truth_fit(prof)
    _, mu_n, sigma_n, _ = fr._fit_one_profile(prof)
    assert mu_n    == pytest.approx(mu_t,    abs=0.01)
    assert sigma_n == pytest.approx(sigma_t, rel=0.01)
```

**Caveat, flagged not papered over:** truth's `p0 σ=5` is itself a guess that
suits these particular spots. On `frosty` (true σ≈172) truth seeded at 5 may land
somewhere else or not converge. If so the honest move is to drop `frosty` from D1
and let C5 and R3 cover it — *not* to loosen the tolerance until it passes. I'll
report what it actually does.

### D2 — summary-line agreement with truth

**This is the comparison you meant, and you were right that it's instant.** It
compares `all_runs_summary.csv` (rows written by the notebook) against
`all_runs_summary_reprocess.csv`, line for line. `compare_pipelines.py` already
implements it — `TRUTH_PAIRS`, `_reldiff`, `report_truth()`, thresholds
`REL_TOL=1e-3` / `REL_WARN=2e-2` at lines 60–80. The test is a thin assertion
over that machinery, not new logic.

Current state, from `compare_pipelines.py --truth` run this session (~2 s):

```
28 overlapping runs:
    MATCH                      17     agree to <0.1% on all 8 stats
    NEAR                        2     <2% but worse than 0.1%
    DIVERGENT_WITH_FAILURES     6     <- the bug
    DIVERGENT_TRUTH_GATE        3     <- truth's fwhm_max=500 cuts frames we keep
```

The six failures are the acceptance criterion for the whole effort:

```
 20260130_newsecondary   3999 total   1124 good   worst reldiff 1.636  y position
     20260213_allmetal   3922 total   3273 good   worst reldiff 0.977  FWHM y std
   20260219_hotandcold  15806 total  15789 good   worst reldiff 0.712  FWHM y std
  20260306_springbreak  14174 total  14166 good   worst reldiff 0.652  FWHM y std
   20260320_statictest   5660 total   5593 good   worst reldiff 0.935  FWHM y std
20260420_zoeysecondary  10156 total   9350 good   worst reldiff 0.954  FWHM x std
```

```python
EXPECTED = {
    # run_key -> the worst verdict it may hold. Anything worse fails.
    # The 3 TRUTH_GATE runs are NOT bugs: truth's fwhm_max=500 discards frames we
    # legitimately keep (springgenie loses 12920 of 14150). See section 6.
    "20260306_springgenie":     "DIVERGENT_TRUTH_GATE",
    "20260320_statictestgenie": "DIVERGENT_TRUTH_GATE",
    "20260814_postspiegenie":   "DIVERGENT_TRUTH_GATE",
}
ACCEPTABLE = {"MATCH", "NEAR"}

def test_summary_agreement_with_truth_does_not_regress():
    import compare_pipelines as cp
    df = cp.report_truth()
    bad = []
    for _, r in df.iterrows():
        allowed = {EXPECTED.get(r.run_key, None)} | ACCEPTABLE
        if r.verdict not in allowed:
            bad.append(f"{r.run_key}: {r.verdict} "
                       f"(worst {r.worst_reldiff:.3f} on {r.worst_column})")
    assert not bad, "\n".join(bad)
```

Two honest caveats:

1. **This stays red until the pipeline is fixed *and* the affected runs are
   reprocessed.** It's the end-to-end acceptance test, not a unit test — it can't
   go green from a code change alone. That's the right shape for it, but it means
   it is red for most of this effort.
2. `report_truth()` currently prints and writes a diagnostics CSV as a side
   effect. Calling it from a test is fine but noisy; it may want a `quiet=True`
   or a split between compute and report. Small change, flag it now.

### E2 — serial and parallel workers agree on real frames

Covers the four duplicated `astype(np.float64)` → `sum` sites
(`fits_reprocess_parallel.py:134-136, 158-160, 309-311, 336-338`) plus the two in
`fits_reprocess.py`. `np.flip` is already centralised in the loaders, so this
should pass — a guard, not a bug hunt.

```python
@pytest.mark.needs_data
def test_serial_and_parallel_workers_agree():
    files = sorted(glob.glob(str(E_DRIVE / "20260213_data/allmetal/allmetal_fits/*.fits")))
    for path in files[:5] + [files[10], files[17]]:
        serial = fr.fit_frame(path)
        par, _, _, _ = frp._fit_worker_fits(path, need_image=False)
        for k in serial:
            a, b = serial[k], par[k]
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), f"{path} {k}"
            else:
                assert a == b, f"{path} {k}: {a!r} != {b!r}"
```

---

## 8. Still to discuss — F1 only

### F1 — worker-count invariance

The single best "the parallelisation is correct" test, and the only one left with
real plumbing cost.

```python
@pytest.mark.parametrize("workers", [1, 4])
def test_worker_count_does_not_change_output(tmp_path, monkeypatch, workers):
    run_dir = _synthesize_fits_run(tmp_path, n=8, size=128)   # tiny Gaussian frames
    monkeypatch.setattr(frp, "WORKERS", workers)
    monkeypatch.setattr(fr, "OUTPUT_DIR", tmp_path / "out")
    frp.process_fits_run(run_dir, make_plots=False, make_movie=False)
    ...compare the two CSVs bit for bit...
```

Three things to settle:

1. `OUTPUT_DIR` is a module constant in `fits_reprocess`, and
   `fits_reprocess_parallel.py:449` calls `OUTPUT_DIR.mkdir()` directly.
   Monkeypatching works but is fragile. Cleaner: an optional `output_dir=`
   parameter on `process_fits_run`. **Signature change to working code — want it?**
2. Synthetic frames must be small to stay fast, but `_choose_pipeline`
   (line 362) selects shmem vs stream by image size and free RAM, so a tiny run
   always takes one path. Covering both means forcing the mode.
3. Is the *shmem* path worth its own test? It needs `_SHMEM_ARR` set up by the
   pool initialiser, so it can't be called directly the way E2 calls
   `_fit_worker_fits`. Testing it means going through `_shmem_run` — most of an
   integration test.

My lean: do F1 with whichever pipeline the tiny run naturally picks, skip forcing
both modes, and skip a separate shmem test. E1 already guarantees both paths call
the same fit function, which is where the risk actually is.

---

## 9. Changes from revision 1

| r1 | r2 | Why |
|---|---|---|
| `assert prof.size >= 512` in A1 | `prof.size > 0` | Hardcoded a resolution assumption |
| synthetic tests at 1024 only | new group R, sizes 64→2592 | The bug is size-scaled; 64 px fails differently |
| — | R3, huge-spot test | A dot may legitimately exceed FWHM 500 or 1000 |
| D2 `test_truth_gate_choice_is_irrelevant` | **deleted** | Encoded an arbitrary constant as physics; the gate isn't the mechanism |
| D3, 4-min per-frame refit sweep | D2, instant summary-line compare | I described §4.11's sweep; you meant summary vs summary. `compare_pipelines.py` already does it in ~2 s |
| E4 as a source-text regex hack | `_order_frames()` helper + real test | You endorsed `kind="stable"`; a helper is cleaner than four open-coded sorts |
| `@pytest.mark.slow` tier | removed | Nothing left is slow |

---

## 10. Deliberately not tested

- **The over-built estimator (§5.2).** Rejected; not resurrecting it. Its useful
  parts survive as R1 and B2.
- **Plot and movie output.** Image comparison is brittle and the bug isn't there.
- **`_count_peaks`.** Informational; doesn't gate `fit_ok` (`fits_reprocess.py:210`).
- **Timing / speedup.** The 3.2× from §4.11 is a result, not an invariant worth
  failing a build over.

---

## 11. Build order

1. `make_fixtures.py`, run it, commit the `.npz` — **while `E:` is still mounted**
2. `conftest.py`, `synth.py`, A1
3. Groups R, B, C, E1, E3 — the offline core, written red
4. **Show you the red run.** No pipeline changes yet
5. D1, reporting honestly what `frosty` does under truth's `p0 σ=5`
6. `_order_frames()` + E4 — the one code change that isn't the fit itself
7. Decide F1 from §8
8. Apply the §5.1 fit rewrite; suite goes green except D2
9. Reprocess the affected runs; D2 goes green

Steps 1–5 touch nothing in the pipeline.
