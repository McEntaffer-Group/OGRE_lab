# Test plan — fit correctness, parallel equivalence, truth agreement

Companion to `FIT_DIAGNOSIS_HANDOFF.md`. Nothing here is written yet; this is the
proposal. Tests marked **DISCUSS** need a decision from you before I build them.

Every claim below about current behaviour was checked against the repo or the
CSVs this session. Where a number is quoted from the handoff rather than re-run,
it says so.

---

## 0. Ground rules

- **Runner:** `pytest` (installed into `ReverseTelescopeDot/.venv` this session,
  9.1.1). Run from the repo root:
  `D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe -X utf8 -m pytest tests/ -v`
- **`-X utf8` is required** — the default cp1252 console encoding crashes on
  non-ASCII output (handoff §4).
- **Offline by default.** Every test in groups A–E runs with `E:` unmounted,
  against committed fixtures. The handful that need real files are marked
  `@pytest.mark.needs_data` and auto-skip when `E:` is absent.
- **Speed target:** the default suite finishes in under ~10 s. Anything slower
  goes behind `@pytest.mark.slow` and is deselected by default.
- **Tests get written red first.** The point of most of group C is that they fail
  against today's `fits_reprocess.py`. I will not touch the pipeline until you've
  seen the red run.

### Markers (`tests/conftest.py`)

```python
import os, numpy as np, pytest
from pathlib import Path

E_DRIVE = Path("E:/Reverse Telescope Test Data")

def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: requires the E: data drive")
    config.addinivalue_line("markers", "slow: minutes, not seconds")

def pytest_collection_modifyitems(config, items):
    if E_DRIVE.exists():
        return
    skip = pytest.mark.skip(reason="E: data drive not mounted")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)

@pytest.fixture(scope="session")
def profiles():
    """Golden 1-D profiles extracted from real frames. See tests/make_fixtures.py."""
    return dict(np.load(Path(__file__).parent / "fixtures" / "profiles.npz"))
```

---

## 1. The fixtures

`tests/make_fixtures.py` — run once, with `E:` mounted, output committed. Not a
test; a generator. Stores **only profiles**, never expected fit values, so the
truth comparison in group D re-derives truth live rather than trusting a baked
number.

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
101, 501 are the ones §1.1 already characterised as healthy. `frosty` frame 1 is
the false-positive guard: I confirmed from `reprocess_output/frosty_frames.csv`
that it fits at μ=(1385.6, 1033.3), σ=(173.3, 171.9) with amp 20228 against
offset 34123 — a genuinely broad spot, nowhere near a bound. A "fix" that
rejects it is over-corrected (handoff §2, §1.5).

Expected size ~100–150 KB compressed. Committed.

---

## 2. Master table

| # | Test | File | Why it exists | Status today | Cost |
|---|---|---|---|---|---|
| **A1** | `test_fixtures_present_and_sane` | `test_fixtures.py` | Catches a stale/corrupt `.npz` before it produces confusing failures elsewhere | passes | instant |
| **B1** | `test_estimator_is_not_saturated` | `test_estimator.py` | Direct regression for §1.1 — estimator returns the `size/4` ceiling on every real profile | **RED** | instant |
| **B2** | `test_estimator_recovers_known_width` | `test_estimator.py` | Synthetic σ∈{3,5,12,40} at noise 60. Asserts *good enough to seed*, not exact | **RED** | instant |
| **B3** | `test_estimator_survives_degenerate_input` | `test_estimator.py` | flat / zeros / hot pixel / edge peak / two peaks (§4.10) | passes | instant |
| **C1** | `test_railed_frames_recover_truth` | `test_fit.py` | **The headline.** Frames 11 & 18 must land on μ≈535.267 / 530.611 | **RED** | ~1 s |
| **C2** | `test_no_parameter_lands_on_a_bound` | `test_fit.py` | §1.5's reliable detector, applied to all golden profiles | **RED** | ~1 s |
| **C3** | `test_ladder_returns_lowest_residual_fit` | `test_fit.py` | **The causal test** — would have caught the original `break` bug | **RED** | ~5 s |
| **C4** | `test_fitted_peak_matches_profile_peak` | `test_fit.py` | §1.2 as a physical invariant: `amp + offset ≈ profile.max()` | **RED** | ~1 s |
| **C5** | `test_broad_but_real_spot_is_kept` | `test_fit.py` | Over-correction guard (frosty). σ≈172 must survive | passes | ~1 s |
| **C6** | `test_fit_is_deterministic` | `test_fit.py` | Same profile twice → identical bits. Guards thread/RNG dependence | passes | instant |
| **D1** | `test_matches_truth_lm_on_golden_profiles` | `test_truth.py` | **The truth comparison you asked for**, offline. vs unbounded LM `p0σ=5` | **RED** | ~2 s |
| **D2** | `test_truth_gate_choice_is_irrelevant` | `test_truth.py` | Open item #5: after the fix, 500 vs 1000 shouldn't change any verdict | **DISCUSS** | instant |
| **D3** | `test_run_level_agreement_with_truth` | `test_truth.py` | §4.11 sweep as a test. `needs_data`, `slow` | **DISCUSS** | ~4 min |
| **E1** | `test_parallel_imports_identical_fit_symbols` | `test_parallel.py` | Prevents a silent fork of the fit math (handoff §1.7) | passes | instant |
| **E2** | `test_serial_and_parallel_workers_agree` | `test_parallel.py` | Covers the 4 duplicated profile-derivation sites. `needs_data` | passes | ~5 s |
| **E3** | `test_worker_failure_returns_empty_row` | `test_parallel.py` | Corrupt/missing file must yield `_empty_row`, not crash the pool | passes | instant |
| **E4** | `test_frame_order_is_stable_under_ties` | `test_parallel.py` | **New finding — see §5.** Unstable sort + tied `frame_num` | **likely RED** | instant |
| **F1** | `test_worker_count_does_not_change_output` | `test_pipeline.py` | *The* parallelism-correctness test. Synthetic run, WORKERS 1 vs 4 | **DISCUSS** | ~10 s |

RED = expected to fail against today's `fits_reprocess.py`, by design.

---

## 3. Full code — the small ones

### A1 — fixtures sane

```python
def test_fixtures_present_and_sane(profiles):
    assert len(profiles) == 12, "expected 6 frames x 2 axes"
    for key, prof in profiles.items():
        assert prof.ndim == 1 and prof.size >= 512, key
        assert np.all(np.isfinite(prof)), key
        assert prof.max() > prof.min(), f"{key} is flat"
```

### B1 — estimator not saturated

The whole original bug in three lines. Today `_estimate_sigma` returns exactly
`profile.size / 4` (320 in x, 256 in y) on every real profile.

```python
@pytest.mark.parametrize("key", ["allmetal_f001_py", "allmetal_f011_py",
                                 "allmetal_f018_py", "allmetal_f101_py"])
def test_estimator_is_not_saturated(profiles, key):
    prof = profiles[key]
    x = np.arange(prof.size, dtype=np.float64)
    est = fr._estimate_sigma(prof, x, float(prof.argmax()))
    ceiling = prof.size / 4.0
    assert est < 0.5 * ceiling, f"estimator pinned at its ceiling ({est:.1f} vs {ceiling:.0f})"
```

### B3 — degenerate inputs

```python
@pytest.mark.parametrize("name,prof", [
    ("flat",      np.full(1024, 21000.0)),
    ("zeros",     np.zeros(1024)),
    ("hot_pixel", _one_hot(1024, 512, 21000.0, 5e4)),
    ("edge_peak", _gauss_on(1024, mu=2.0,    sigma=4.0)),
    ("two_peaks", _gauss_on(1024, mu=300.0,  sigma=4.0)
                + _gauss_on(1024, mu=700.0,  sigma=4.0)),
])
def test_estimator_survives_degenerate_input(name, prof):
    x = np.arange(prof.size, dtype=np.float64)
    est = fr._estimate_sigma(prof, x, float(prof.argmax()))
    assert np.isfinite(est) and est > 0, name
    assert est <= prof.size, name
```

Deliberately weak assertions. These inputs have no right answer; the test only
pins down "does not raise, does not return garbage that poisons the ladder."

### C1 — the headline

```python
# (frame key, expected mu_y, expected sigma_y) — from handoff §5.1, cross-checked
# there against an unbounded LM fit to 4 decimal places.
RAILED_ANCHORS = [
    ("allmetal_f011_py", 535.2666, 4.4065),
    ("allmetal_f018_py", 530.6110, 5.3530),
]

@pytest.mark.parametrize("key,mu_want,sigma_want", RAILED_ANCHORS)
def test_railed_frames_recover_truth(profiles, key, mu_want, sigma_want):
    amp, mu, sigma, offset = fr._fit_one_profile(profiles[key])
    assert mu   == pytest.approx(mu_want,    abs=0.05)
    assert sigma == pytest.approx(sigma_want, rel=0.02)
```

Today this returns μ=0.000, σ=355.26 for frame 11 and μ=410.4, σ=512.0 for frame 18.

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

Collects all failures rather than dying on the first, so one run shows you the
whole picture. This is the §1.5 detector: bound-hitting is unambiguous, large
sigma alone is not.

### C4 — amp is above-offset

```python
@pytest.mark.parametrize("key", ["allmetal_f001_py", "allmetal_f101_py",
                                 "allmetal_f501_py", "frosty_f001_py"])
def test_fitted_peak_matches_profile_peak(profiles, key):
    prof = profiles[key]
    amp, mu, sigma, offset = fr._fit_one_profile(prof)
    assert amp + offset == pytest.approx(prof.max(), rel=0.05)
```

§1.2 stated as physics, not as an implementation check: the model's predicted
peak height must match the data's. Today the *seed* predicts ~2× the data; this
asserts the *converged* fit doesn't.

### C6 — determinism

```python
def test_fit_is_deterministic(profiles):
    prof = profiles["allmetal_f011_py"]
    assert fr._fit_one_profile(prof) == fr._fit_one_profile(prof)
```

### E1 — no silent fork

```python
def test_parallel_imports_identical_fit_symbols():
    import fits_reprocess as fr
    import fits_reprocess_parallel as frp
    assert frp._fill_fit_results is fr._fill_fit_results
    assert frp._empty_row         is fr._empty_row
    assert frp._load_fits_frame   is fr._load_fits_frame
    assert frp._load_image_frame  is fr._load_image_frame
```

Verified accurate: `fits_reprocess_parallel.py:97-108` imports all four.

### E3 — worker failure is contained

```python
def test_worker_failure_returns_empty_row(tmp_path):
    import fits_reprocess_parallel as frp
    bogus = tmp_path / "nope0001 26-01-01 00-00-00.fits"
    bogus.write_bytes(b"not a fits file")
    row, img, px, py = frp._fit_worker_fits(str(bogus), need_image=False)
    assert row["fit_ok"] is False
    assert np.isnan(row["mu_x"]) and np.isnan(row["mu_y"])
    assert row["frame_num"] == 1          # parsed from the name, not the payload
    assert (img, px, py) == (None, None, None)
```

---

## 4. Snippets — the bigger ones

### C3 — ladder returns the lowest-residual fit

The test that would have caught the original bug. Written
**implementation-agnostically**: it doesn't care how many rungs there are or what
they're seeded with, only that no alternative seed beats the answer you returned.

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

On frame 11 today: returned RMS is 93.6, seed σ=5 reaches 61.2 → fails loudly,
which is exactly the §1.3 finding restated as an assertion.

### D1 — agreement with the truth notebook, offline

The truth recipe from `dot_movie-Copy3.ipynb` (handoff §1.6): **unbounded**
`curve_fit` → Levenberg–Marquardt, `p0` sigma fixed at 5.

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

**Caveat I want to flag rather than paper over:** truth's `p0 σ=5` is itself a
guess that happens to suit these spots. On `frosty` (true σ≈172) truth seeded at
5 may converge somewhere different, or not converge at all. If it doesn't, the
honest move is to exclude `frosty` from D1 and let C5 cover it — *not* to loosen
the tolerance until it passes. I'll report what it actually does before deciding.

### E2 — serial and parallel workers agree on real frames

Covers the four duplicated `astype(np.float64)` → `sum` sites
(`fits_reprocess_parallel.py:134-136, 158-160, 309-311, 336-338`) plus the two in
`fits_reprocess.py`. The `np.flip` is already centralised in the loaders, so this
should pass — it's a guard, not a bug hunt.

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

### E4 — stable ordering under tied frame numbers

See §5 for why this is here.

```python
def test_frame_order_is_stable_under_ties():
    """Rows with equal frame_num must keep discovery order through the sort."""
    rows = [fr._empty_row(f"f{i}.fits", frame_num=7, timestamp=f"t{i}") for i in range(50)]
    df = pd.DataFrame(rows).sort_values("frame_num", kind="stable").reset_index(drop=True)
    assert list(df["filename"]) == [f"f{i}.fits" for i in range(50)]
```

As written this tests pandas, not us — it becomes a real test once the assertion
targets the pipeline's own sort. The honest version asserts on the source:

```python
def test_pipeline_sorts_stably():
    """All four frame-ordering sorts must pass kind='stable'."""
    import re, pathlib
    for f in ("fits_reprocess.py", "fits_reprocess_parallel.py"):
        src = pathlib.Path(f).read_text()
        for m in re.finditer(r'\.sort_values\(\s*"frame_num"([^)]*)\)', src):
            assert 'kind="stable"' in m.group(1), f"{f}: unstable sort at {m.start()}"
```

A source-text assertion is ugly. **DISCUSS** — the alternative is to route all
four sites through one `_order_frames(df)` helper and test *that*, which is
cleaner but is a refactor of working code.

---

## 5. New finding: the frame-order sort is unstable

Not in the handoff. Found while grounding this plan.

`csv_to_dotplots.py:83,85` sorts with an explicit `kind="stable"`. All four
sorting sites in the pipeline do **not**:

```
fits_reprocess.py:713           df.sort_values("frame_num")
fits_reprocess.py:776           df.sort_values("frame_num")
fits_reprocess_parallel.py:529  df.sort_values("frame_num")
fits_reprocess_parallel.py:632  df.sort_values("frame_num")
```

pandas defaults to quicksort, which is not stable. Ties therefore land in
arbitrary order. Ties are reachable — **31 of ~78 runs have them**:

```
runs with duplicate or -1 frame_num: 31
  fan_restart5237       rows= 6235 unique= 6232 neg=   0
  fanoff5237            rows= 9132 unique= 9129 neg=   0
  morning25             rows=11523 unique=11520 neg=   0
  ...
  warming               rows=   79 unique=   77 neg=   3
```

Two distinct causes, and they matter differently:

1. **`frame_num == -1` (22 runs, always exactly 3 rows).** I checked these: they
   are the self-ingested plot PNGs (`warming_FFT_reprocess.png` etc.) that commit
   `ccc3796` added the denylist for. These CSVs are **stale**, written before that
   fix. All three carry `fit_ok=False`, so they never reach the baseline. Benign,
   and reprocessing clears them.

2. **Genuine duplicate `frame_num >= 0` (2–3 rows per run, ~9 runs).** These are
   real frames with colliding numbers, and they *do* sort non-deterministically.

Whether cause 2 can actually move a result depends on `csv_to_dotplots.py:114`
baselining positions on the first surviving frame — per the standing note, one
changed first-frame shifts every position by a constant. A tie at the *front* of
the ordering is the dangerous case; a tie in the middle is cosmetic. I have not
yet checked whether any of the ~9 affected runs has its tie at frame 1, and I'd
want to before calling this a live bug rather than a latent one.

**Cheap fix regardless:** add `kind="stable"` at all four sites. One word each,
no behaviour change when there are no ties.

---

## 6. Tests I want to talk about first

### D2 — is the FWHM gate supposed to be irrelevant?

```python
def test_truth_gate_choice_is_irrelevant(profiles):
    """After the fix, no golden frame should sit between the 500 and 1000 gates."""
    for key, prof in profiles.items():
        _, _, sigma, _ = fr._fit_one_profile(prof)
        fwhm = sigma * fr.FWHM_FACTOR
        assert not (500 <= fwhm < 1000), f"{key}: fwhm={fwhm:.1f} is gate-sensitive"
```

This encodes a *design intent* (open item #5), not a known bug. It's only a fair
test if you agree the intent is "a correct fit never lands in that band." If you'd
rather keep 1000 as a deliberate safety margin for genuinely broad spots, this
test is wrong and I should drop it in favour of just recording `on_bound` in the
CSV. **Which is it?**

### D3 — the slow run-level sweep

§4.11 as a test: sample every 25th frame of allmetal, fit old/new/truth, assert
zero railed and max |Δμ| < 1 px. Recorded cost was ~4 min for 20 frames × 2 axes,
dominated by FITS reads.

Questions: is a 4-minute `--slow` test worth having, or is a one-off script you
run before reprocessing better? And how many frames — 20 is thin, 393 was
abandoned at >40 min (§4.12). I lean toward **not** making this a test: it's a
pre-flight check, and `compare_pipelines.py` already does the run-level
comparison properly once the data is reprocessed.

### F1 — worker-count invariance

The single best "the parallelisation is correct" test, and the one with real
plumbing cost. Shape:

```python
@pytest.mark.parametrize("workers", [1, 4])
def test_worker_count_does_not_change_output(tmp_path, monkeypatch, workers):
    run_dir = _synthesize_fits_run(tmp_path, n=8)     # 8 tiny Gaussian frames
    monkeypatch.setattr(frp, "WORKERS", workers)
    monkeypatch.setattr(fr, "OUTPUT_DIR", tmp_path / "out")
    frp.process_fits_run(run_dir, make_plots=False, make_movie=False)
    ...compare the two CSVs bit for bit...
```

Three things to settle:

1. `OUTPUT_DIR` is a module-level constant in `fits_reprocess`, and
   `fits_reprocess_parallel.py:449` calls `OUTPUT_DIR.mkdir()` directly.
   Monkeypatching it works but is fragile. Cleaner would be an optional
   `output_dir=` parameter on `process_fits_run` — a small signature change to
   working code. **Do you want that change, or is monkeypatching fine?**
2. Synthetic frames need to be small (say 128×128) to keep this fast, but the
   shmem-vs-stream pipeline choice keys off image size and available RAM
   (`_choose_pipeline`, line 362). A tiny run will always pick one path. To cover
   both I'd need to force the mode — `process_fits_run` may already take a `mode`
   argument; I'd check before writing.
3. Same question one level up: is the *shmem* path worth a separate test? It needs
   `_SHMEM_ARR` initialised via the pool initialiser, so it can't be called
   directly the way E2 calls `_fit_worker_fits`. Testing it means going through
   `_shmem_run`, which is most of an integration test.

---

## 7. Deliberately not tested

- **The over-built estimator (§5.2).** You rejected it; I'm not resurrecting it.
  Its test cells (§4.9, §4.10) survive as B2 and B3, which is the useful part.
- **Plot and movie output.** Image comparison is brittle and these aren't where
  the bug lives.
- **`_count_peaks`.** Informational only; it doesn't gate `fit_ok`
  (`fits_reprocess.py:210`).
- **Timing / speedup.** The 3.2× from §4.11 is a nice result, not an invariant
  worth failing a build over.

---

## 8. Order I'd build in

1. `make_fixtures.py`, run it, commit the `.npz` — **while `E:` is still mounted**
2. `conftest.py` + A1
3. Group B, C, E1, E3 — the offline core, written red
4. **Show you the red run.** No pipeline changes yet
5. D1 — reporting honestly what `frosty` does under truth's `p0 σ=5`
6. Decide D2 / D3 / E4 / F1 from §6
7. Only then: apply the §5.1 fix, watch the suite go green

Steps 1–5 touch nothing in the pipeline.
