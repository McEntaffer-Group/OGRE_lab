# Centroid-railing diagnosis — handoff notes (2026-09-02)

> **STALE IN PARTS — read `NEXT_SESSION.md` first.**
>
> Superseded 2026-09-03. What changed:
>
> * **The bug is FIXED.** Section 6 item 1 ("Nothing is fixed yet") and the
>   table in section 3 ("`fits_reprocess.py` **unchanged**") are both false now.
>   The fix landed in `fits_reprocess.py` on branch `claude/build-test-suite`.
> * **Section 6 item 4 is done.** The test suite exists in `tests/` — 304 tests.
> * **Section 1.5's table is contaminated.** It globbed
>   `reprocess_output/*_frames.csv`, where 20 runs collapse into 8 bare-named
>   files. Its `minutely`, `overnight` and `warming` rows each describe one
>   arbitrary run of several, and 12 colliding runs were never scanned. "21 of
>   ~78 runs affected" is a floor, not a count. See `TEST_PLAN.md` section 1.
> * **Section 1.5's "broad but real" claim is wrong for at least two runs.**
>   `frosty` and `postwinterbreak` do not have genuinely broad spots — they are
>   two-dot frames where the fit locks onto a diffuse companion while a compact
>   dot sits elsewhere. Verified in 2D. `realwinterbreak` and `newprimary` are
>   unchecked. See `TEST_PLAN.md` section 4.
>
> Sections 1.1-1.4, 1.6 and 1.7 held up and were the basis for the fix.

Investigation into the ±400 px centroid spikes in `allmetal_position_reprocess.png`.
Root cause found and verified against real data. Two candidate rewrites exist but
**nothing has been changed in `fits_reprocess.py` yet.**

Everything below was actually run this session. Where a claim was not verified, it
says so.

---

## 1. Verified findings

### 1.1 `_estimate_sigma` returns a constant, not a measurement

`fits_reprocess.py:166` takes the second moment of the **whole** profile after
subtracting only the median. `np.clip(profile - median, 0, None)` *rectifies* the
background noise — half the samples are above the median, and clipping keeps all of
them — so noise spread over ±512 px carries most of the weight. Since the second
moment weights by distance², it dominates completely.

```
frame    1  sigma_est_x=  303.42 (cap 320)   sigma_est_y=  256.00 (cap 256)
frame   11  sigma_est_x=  278.00 (cap 320)   sigma_est_y=  256.00 (cap 256)
frame   18  sigma_est_x=  303.41 (cap 320)   sigma_est_y=  251.97 (cap 256)
frame  101  sigma_est_x=  287.92 (cap 320)   sigma_est_y=  231.08 (cap 256)
frame  501  sigma_est_x=  285.79 (cap 320)   sigma_est_y=  255.18 (cap 256)
```

True sigma is ~4.3 (x) and ~5.1 (y). The raw value always exceeds the
`profile.size/4` clip, so the function returns the ceiling constant.

Weight breakdown (allmetal y-profile, dot contrast ~4% on a 21000 pedestal):

```
frame 1:  shifted.sum()=32271, only 31.2% of that weight is within ±15px of the peak
          -> 68.8% of what the second moment measures is rectified background noise
          raw sqrt(var) = 256.7  (ceiling 256)
```

### 1.2 `p0_amp` is ~24× too large

`gaussian(x, amp, mu, sigma, offset) = amp*exp(...) + offset`, so `amp` is height
**above** offset. `p0_amp = profile.max()` is absolute.

```
frame 1:  p0_amp used = 21955, should be 926  (24x)
```

The starting model predicts 21955 + 21029 ≈ 42,984 counts at the peak where the data
is 21,955 — off by ~2× across the whole profile.

**This is not the primary driver.** Fixing it alone does not rescue the railed frames:

```
frame 11  p0_amp fixed only (sigma_est still 256): mu=0.000   sigma=355.189
          both fixed (p0_amp above-offset, sigma=5):  mu=535.267 sigma=4.406
frame 18  p0_amp fixed only (sigma_est still 252): mu=410.365 sigma=512.000
          both fixed:                                 mu=530.611 sigma=5.353
```

### 1.3 `break` accepts the first rung that doesn't raise, not the best fit

This is the actual correctness bug. Per-rung results on the two railed frames:

```
=== frame 11  sigma_est=256.0
   rung 0 p0sigma=  256.0 -> mu=    0.000 sigma= 355.261  RMSresid=    93.6
   rung 1 p0sigma=  128.0 -> mu=  535.267 sigma=   4.407  RMSresid=    61.2
   rung 2 p0sigma=  512.0 -> mu=    0.000 sigma= 355.207  RMSresid=    93.6
   rung 3 p0sigma=   10.0 -> mu=  535.267 sigma=   4.406  RMSresid=    61.2
   rung 4 p0sigma=   50.0 -> mu=  535.267 sigma=   4.407  RMSresid=    61.2
   rung 5 p0sigma=  100.0 -> mu=  535.267 sigma=   4.406  RMSresid=    61.2

=== frame 18  sigma_est=252.0
   rung 0 p0sigma=  252.0 -> mu=  410.365 sigma= 512.000  RMSresid=    91.7
   rung 1 p0sigma=  126.0 -> mu=  530.611 sigma=   5.353  RMSresid=    55.6
   rung 2 p0sigma=  503.9 -> mu=  410.365 sigma= 512.000  RMSresid=    91.7
   rung 3-5                -> mu=  530.611 sigma=   5.353  RMSresid=    55.6
```

Five of six rungs find the right answer with a clearly lower residual. Rung 0 loses
on residual and wins anyway because it returned without raising.

Rung statistics over ~120 sampled Y profiles:

- rung 0 (`sigma_est`) returns without raising **12.1%** of the time (14/116); it
  raises `RuntimeError` otherwise, exhausting `maxfev=5000`
- when rung 0 *does* return, it rails ~38% of the time in Y (8 of 21 wins)
- rung 1 (`sigma_est/2`) wins ~100 times and **never** rails
- rungs 2–5 (`sigma_est*2`, `10.0`, `50.0`, `100.0`) **never won once**

### 1.4 Failures are ~100% on the Y axis

allmetal, from `reprocess_output/allmetal_frames.csv`:

```
fit_ok=False reasons:  fwhm_y>=1000: 649    fwhm_x>=1000: 0    nan: 0
fit_ok=True but sigma_x > 50: 0
sigma_x (fit_ok rows): min 3.39  median 4.23  max 5.50   <- never misbehaves
sigma_y (fit_ok rows): min 1.00  median 6.46  max 424.39
mu_y    (fit_ok rows): min 2.16e-13 (lower bound)  max 1024.0 (upper bound)
```

### 1.5 The reliable detector is a parameter on a bound, NOT large sigma

Large sigma alone gives false positives — several runs have genuinely broad spots
(`frosty`, `postwinterbreak`, `realwinterbreak`, `newprimary` all show sigma ~160
with amp ≈ offset and mu nowhere near a bound). Those are real, not failures.

Bound-hitting is unambiguous. Scan across every `*_frames.csv` (see §4.1):

```
run                        rows railed    pct   detail
laser                      5063   3175  62.7%   y:sigma at 240 (=size/2) x3082
bridgetstatic              6743   3081  45.7%   x:mu at 2592 (=size) x2675
newsecondary               3999   1751  43.8%   y:sigma at 972 x1586
longweekend                6970   1653  23.7%
weekend                    3966    940  23.7%
minutelyovernight          2493    531  21.3%
fansoff                    3991    837  21.0%
overnight                  1090    176  16.1%
minutely                   1563    229  14.7%
allmetal                   3922    443  11.3%   y:mu0=19 sgB=424(@512)
warming                      79      7   8.9%
overnightvideo             1376    108   7.8%
nightvideo                 9009    684   7.6%
zoeysecondary             10156    581   5.7%
minutely2                  1176     17   1.4%
statictest                 5660     48   0.8%
laserweekend               3995     19   0.5%
secondarylaser             2572      2   0.1%
hotandcold                15806     10   0.1%
morningsecondly           10002      4   0.0%
springbreak               14174      5   0.0%
```

**21 of ~78 runs are affected.** This is far bigger than allmetal alone.

### 1.6 Truth pipeline differences (`dot_movie-Copy3.ipynb`)

| | truth notebook | fits_reprocess |
|---|---|---|
| solver | `curve_fit` unbounded → **LM** | `bounds=` → **TRF** |
| sigma p0 | `5` (cell 16) | saturated `sigma_est` |
| rung selection | n/a, single fit | `break` on first non-raise |
| reject gate | `fwhm_min=1, fwhm_max=500` (cell 18) | `FWHM_MIN_PX=1, FWHM_MAX_PX=1000` |
| baseline | `mus_x_f - mus_x_f[0]` after filtering (cell 20) | same, `csv_to_dotplots.py:114` |

Truth's stricter 500 gate would catch some but not all leaked garbage (41 of 126
leaked frames in allmetal have `fwhm_y` under 500).

Summary comparison, allmetal (`allmetal_summary.csv` vs `allmetal_summary_reprocess.csv`):

```
              truth      reprocess
FWHM x (as)   1.5032     1.5016      <- 0.1% agreement, x is fine
FWHM y (as)   2.2419     5.7260      <- 155% off, y is destroyed
y pos std     4.1383     10.1745
```

### 1.7 `fits_reprocess_parallel.py` shares the fit math

`fits_reprocess_parallel.py:92` imports `_fill_fit_results` from `fits_reprocess`.
Fixing the fit in `fits_reprocess.py` fixes both entry points; nothing in the
parallel file needs to change. Worth asserting this identity in a test so it cannot
silently fork.

---

## 2. Corrections made mid-session — do not repeat these

- **I first claimed the retry ladder was dead code and rung 0 always returned.**
  Wrong. Rung 0 raises 88% of the time. The bad inference was reading "a fit
  converged" off the CSV and assuming that meant rung 0 converged — the CSV does not
  record which rung won.
- **I then guessed the good fits land on rung 3 (`sigma=10`).** Also wrong; they land
  on rung 1. Rungs 2–5 never win.
- **I flagged runs with sigma > 50 as broken.** Over-broad — see §1.5.
- **First `_estimate_sigma` rewrite was tested against invented noise (std 174)** when
  the real noise is ~60 counts (the RMS residual on real frames). That drove me to add
  smoothing/SNR-gating machinery that the real data does not need. Use the measured
  residual (~55–64 counts on allmetal) as the realistic noise level.

---

## 3. Files created / changed today

| file | status |
|---|---|
| `parallel_troubleshooting.ipynb` | **overwritten** — user's original stub cells were lost. Now holds the over-built rewrite plus synthetic / pathological / sweep test cells. User called it "the monstrosity"; keep only if the test cells are useful. |
| `parallel_troubleshooting_simple.ipynb` | The version the user actually wants. 4 cells. |
| `FIT_DIAGNOSIS_HANDOFF.md` | This file. |
| `fits_reprocess.py` | **unchanged** |
| `fits_reprocess_parallel.py` | **unchanged** |
| `csv_to_dotplots.py` | **unchanged** |

The user's preference is explicit: **small, simple, visible code.** Do not build
scratchpad modules they cannot see, and do not add machinery (smoothing, SNR gates,
sub-pixel interpolation) that has not been shown to be necessary on real data.

---

## 4. Test code inventory

Scratchpad snippets are reproduced in full here because the scratchpad directory does
not survive the session. Interpreter for all of these:

```
D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe
```

Run from `D:/Users/jad507/PycharmProjects/OGRE_lab`. Use `-X utf8` — the default
cp1252 console encoding crashes on non-ASCII output.

### 4.1 Bound-railing scan across all runs — HIGHEST VALUE, no E: needed

Pure CSV, runs in seconds, produced the §1.5 table. **This should become a test.**

```python
import pandas as pd, numpy as np, glob, os
rows=[]
for f in sorted(glob.glob('reprocess_output/*_frames.csv')):
    if f.endswith('_prev.csv'): continue
    d=pd.read_csv(f)
    if 'sigma_y' not in d: continue
    n=len(d); hits=0; detail=[]
    for ax in ('x','y'):
        mu=d['mu_'+ax]; sg=d['sigma_'+ax]
        lo=(mu.abs()<1e-3).sum()
        top=mu.max()
        hi=((mu-top).abs()<1e-9).sum() if abs(top-round(top))<1e-6 and ((mu-top).abs()<1e-9).sum()>1 else 0
        smax=sg.max()
        sb=((sg-smax).abs()<1e-9).sum() if abs(smax-round(smax))<1e-6 and ((sg-smax).abs()<1e-9).sum()>1 else 0
        hits+=lo+hi+sb
        if lo+hi+sb: detail.append('%s:mu0=%d muHi=%d(@%g) sgB=%d(@%g)'%(ax,lo,hi,top,sb,smax))
    if hits: rows.append((os.path.basename(f)[:-11],n,hits,100*hits/n,'; '.join(detail)))
rows.sort(key=lambda r:-r[3])
print('%-24s %6s %6s %6s  %s'%('run','rows','railed','pct','detail'))
for r in rows: print('%-24s %6d %6d %5.1f%%  %s'%r)
```

Note it must *infer* the profile size from the max observed value, because
`{run}_frames.csv` does not record image dimensions. Adding `nx`/`ny` columns would
make this exact and much simpler.

### 4.2 Per-rung diagnostic on a single frame — the definitive one

Produced §1.3. Needs E:. ~30s for two frames.

```python
import numpy as np, glob
from astropy.io import fits
from scipy.optimize import curve_fit
import fits_reprocess as fr
files=sorted(glob.glob(r'E:/Reverse Telescope Test Data/20260213_data/allmetal/allmetal_fits/*.fits'))
def g(x,a,m,s,o): return a*np.exp(-0.5*((x-m)/s)**2)+o
for i in [10,17]:
    with fits.open(files[i]) as h: img=np.flip(h[0].data,axis=(0,1)).astype(float)
    py=np.sum(img,axis=1); y=np.arange(py.size); mu=float(py.argmax())
    se=fr._estimate_sigma(py,y,mu); b=([0.,0.,1.,-np.inf],[np.inf,float(py.size),py.size/2.,np.inf])
    print('=== frame',i+1,' sigma_est=%.1f'%se)
    for k,s in enumerate([se,se/2.,se*2.,10.,50.,100.]):
        try:
            p,_=curve_fit(g,y,py,p0=[py.max(),mu,s,np.median(py)],bounds=b,maxfev=5000)
            r=np.sqrt(np.mean((g(y,*p)-py)**2))
            print('   rung %d p0sigma=%7.1f -> mu=%9.3f sigma=%8.3f  RMSresid=%8.1f'%(k,s,p[1],p[2],r))
        except Exception as e:
            print('   rung %d p0sigma=%7.1f -> RAISED %s'%(k,s,type(e).__name__))
```

### 4.3 `_estimate_sigma` saturation check

Produced §1.1. Needs E:. ~60s.

```python
import numpy as np, glob
from astropy.io import fits
import fits_reprocess as fr
files=sorted(glob.glob(r'E:/Reverse Telescope Test Data/20260213_data/allmetal/allmetal_fits/*.fits'))
for i in [0,10,17,100,500]:
    with fits.open(files[i]) as h: img=np.flip(h[0].data,axis=(0,1)).astype(float)
    px=np.sum(img,axis=0); py=np.sum(img,axis=1)
    sx=fr._estimate_sigma(px,np.arange(px.size),float(px.argmax()))
    sy=fr._estimate_sigma(py,np.arange(py.size),float(py.argmax()))
    print('frame %4d  sigma_est_x=%8.2f (cap %.0f)   sigma_est_y=%8.2f (cap %.0f)'
          %(i+1,sx,px.size/4,sy,py.size/4))
```

### 4.4 Noise-weight fraction and `p0_amp` scaling

Produced §1.1 / §1.2. Needs E:.

```python
import numpy as np, glob
from astropy.io import fits
files=sorted(glob.glob(r'E:/Reverse Telescope Test Data/20260213_data/allmetal/allmetal_fits/*.fits'))
for i in [0,10,17]:
    with fits.open(files[i]) as h: img=np.flip(h[0].data,axis=(0,1)).astype(float)
    py=np.sum(img,axis=1); y=np.arange(py.size); mu=float(py.argmax())
    med=np.median(py); shifted=np.clip(py-med,0,None); tot=shifted.sum()
    near=(np.abs(y-mu)<=15)
    print('--- frame',i+1)
    print('  median=%.0f max=%.0f amp above bkg=%.0f'%(med,py.max(),py.max()-med))
    print('  weight within +/-15px of peak = %.3f'%(shifted[near].sum()/tot))
    print('  raw sqrt(var) = %.1f (ceiling %.0f)'
          %(np.sqrt(np.sum((shifted/tot)*(y-mu)**2)), py.size/4))
    print('  p0_amp %.0f vs should-be %.0f (%.0fx)'%(py.max(),py.max()-med,py.max()/(py.max()-med)))
```

### 4.5 `fit_ok` rejection-reason breakdown

Produced §1.4. Pure CSV, instant.

```python
import pandas as pd, numpy as np
d=pd.read_csv('reprocess_output/allmetal_frames.csv')
bad=d[d.fit_ok==False]; ok=d[d.fit_ok]
print('fwhm_y>=1000:',int((bad.fwhm_y>=1000).sum()),' fwhm_x>=1000:',int((bad.fwhm_x>=1000).sum()),
      ' nan:',int(bad.mu_x.isna().sum()))
print('fit_ok but mu_y at lower bound:',int((ok.mu_y<1e-6).sum()))
print('fit_ok but mu_y at upper bound:',int((ok.mu_y>1023.999).sum()))
print('fit_ok but sigma_x>50:',int((ok.sigma_x>50).sum()))
for c in ['sigma_x','sigma_y','mu_y']: print(c, ok[c].describe().to_dict())
```

### 4.6 Rung-0 success rate

Produced the 12.1% figure in §1.3. Needs E:. **Slow, ~3 min for 116 profiles** —
rung 0 burns the full `maxfev=5000` before raising.

```python
import numpy as np, glob
from astropy.io import fits
from scipy.optimize import curve_fit
import fits_reprocess as fr
files=sorted(glob.glob(r'E:/Reverse Telescope Test Data/20260213_data/allmetal/allmetal_fits/*.fits'))
def g(x,a,m,s,o): return a*np.exp(-0.5*((x-m)/s)**2)+o
ok=0; n=0
for i in range(0,400,7):
    with fits.open(files[i]) as h: img=np.flip(h[0].data,axis=(0,1)).astype(float)
    for prof in (np.sum(img,axis=0), np.sum(img,axis=1)):
        x=np.arange(prof.size); mu=float(prof.argmax())
        se=fr._estimate_sigma(prof,x,mu)
        b=([0.,0.,1.,-np.inf],[np.inf,float(prof.size),prof.size/2.,np.inf]); n+=1
        try:
            curve_fit(g,x,prof,p0=[prof.max(),mu,se,np.median(prof)],bounds=b,maxfev=5000); ok+=1
        except Exception: pass
print('rung 0 returned without raising: %d / %d (%.1f%%)'%(ok,n,100*ok/n))
```

### 4.7 Which-rung-wins census

Produced the rung statistics in §1.3. Needs E:. **Very slow, >7 min for 100 frames.**
Same code as 4.2 but looping `range(0,300,3)` and tallying the winning `k` per axis,
with a `railed` flag (`mu<1e-3 or mu>size-1e-3 or sigma>size/2-1e-6`). Key output:
rung 1 dominates and never rails; rung 0 wins 21 times in Y with 8 railed; rungs 2–5
never win.

### 4.8 Broad-but-real vs railed discrimination

Produced §1.5. Pure CSV, instant. Confirms `frosty`/`realwinterbreak`/`newprimary`
have legitimate wide spots.

```python
import pandas as pd
for r in ['frosty','postwinterbreak','realwinterbreak','genieshots','laser','newprimary']:
    d=pd.read_csv(f'reprocess_output/{r}_frames.csv'); ok=d[d.fit_ok==True]
    print('===',r,len(d))
    for c in ['sigma_x','sigma_y','mu_x','mu_y','amp_x','amp_y']:
        print('   %-8s min=%10.3f med=%10.3f max=%10.3f'%(c,ok[c].min(),ok[c].median(),ok[c].max()))
```

### 4.9 Synthetic profile battery — in `parallel_troubleshooting.ipynb` cell 7

No disk I/O, instant. Known sigma and mu on a realistic pedestal. Excerpt only —
needs `gaussian`, `rng` and the estimator from the notebook's earlier cells:

```python
prof = gaussian(np.arange(1024.), 926., 537., true_sig, 21000.) + rng.normal(0, 60, 1024)
```

Sweeps `true_sig` in `[3, 5, 12, 40]` × noise in `[60 (realistic), 174 (harsh)]`.
**Use 60, not 174** — see §2.

### 4.10 Pathological-input battery — `parallel_troubleshooting.ipynb` cell 9

flat / all-zeros / one hot pixel / peak at edge / two peaks. Instant.
Known behaviour of the over-built version: flat and all-zeros decline (NaN); the hot
pixel converges but sets `on_bound=True`; two peaks picks one (expected for a
single-Gaussian model).

### 4.11 Old-vs-new sweep with timing — `parallel_troubleshooting.ipynb` cell 13

Needs E:. `STEP=25, N_MAX=500` → 20 frames × 2 axes in ~4 min. Compares old fit, new
fit and truth (unbounded `p0=5`), counts railed fits and `|mu-truth|>1px`, and times
both. Result recorded:

```
                       OLD      NEW
  railed fits             1        0
  |mu-truth|>1px          3        0
  max |mu-truth|      155.8      0.0
  fit wall time      187.12s   57.75s
```

The 3.2× speedup is fit time only, excluding FITS reads.

### 4.12 Never completed

A 393-frame old-vs-truth sweep (`range(0, 3922, 10)`) was started and killed after
several minutes. If a full-run comparison is wanted, budget ~40+ min or parallelise
the FITS reads.

---

## 5. Candidate rewrites

### 5.1 The one the user wants — `parallel_troubleshooting_simple.ipynb`

`_estimate_sigma`: same median cut as the old code, but count **only the contiguous
above-zero run around the max**, then divide by 2.355.

```python
profile = np.asarray(profile, dtype=np.float64)
shifted = np.clip(profile - np.median(profile), 0.0, None)
i0 = int(shifted.argmax())
li = i0
while li > 0 and shifted[li - 1] > 0:
    li -= 1
ri = i0
while ri < shifted.size - 1 and shifted[ri + 1] > 0:
    ri += 1
return (ri - li + 1) / FWHM_FACTOR
```

`_fit_one_profile`: `amp_guess = max - median`; try `[sigma_est, sigma_est*2,
sigma_est*0.5]`; **keep the lowest RMS residual**.

Verified — both railed frames fixed, matches truth to 3–4 decimals:

```
=== frame 11
  OLD sigma_est   256.00   NEW sigma_est     8.92
  OLD fit  -> mu=    0.0000  sigma= 355.2611
  NEW fit  -> mu=  535.2671  sigma=   4.4060
  TRUTH    -> mu=  535.2666  sigma=   4.4065
```

Known wart: the estimate runs ~2× high (8.92 vs true 4.41) because the median cut sits
well below half-max, so the surviving run is wider than FWHM and 2.355 is the wrong
divisor. Harmless — the ×0.5 rung brackets it and residual scoring picks the winner.
Cutting at `median + 0.5*(max-median)` instead makes 2.355 exactly right, one line.

### 5.2 The over-built one — `parallel_troubleshooting.ipynb`

Half-max crossing with boxcar smoothing, MAD noise estimate, SNR gate, sub-pixel
interpolation, boxcar deconvolution, a fallback ladder and a diagnostics dict. It
works (recovers sigma 3→3.34, 5→5.23, 12→12.44, 40→36.25) but the user rejected it as
over-engineered. **Do not resurrect this without being asked.** Its test cells (§4.9,
§4.10, §4.11) are worth keeping regardless of which estimator wins.

---

## 6. Open items

1. **Nothing is fixed yet.** `fits_reprocess.py` still has the original code.
2. Decide 5.1 vs 5.2, apply to `fits_reprocess.py`, then re-run the affected runs.
   21 runs are affected (§1.5) — all would need reprocessing, not just allmetal.
3. **Instrumentation worth adding to `{run}_frames.csv`:** `nx`/`ny` (so a railed fit
   is detectable from the CSV alone — §4.1 currently has to infer it), `resid_x`/
   `resid_y`, and `on_bound_x`/`on_bound_y`.
4. **The test suite from the earlier plan was never written.** No `tests/` directory
   exists. Highest-value tests, in order: ladder-selects-lowest-residual (would have
   caught this), no-parameter-on-a-bound, estimator-recovers-known-width, and
   `assert fits_reprocess_parallel._fill_fit_results is fits_reprocess._fill_fit_results`.
   Golden fixtures should be committed **profiles** (~16 KB `.npz`), not FITS, so tests
   run without E: mounted. Frames 11 and 18 of allmetal are the natural anchors.
5. `FWHM_MAX_PX = 1000` vs the notebook's 500 is still unreconciled. With `on_bound`
   recorded the gate matters much less.
6. Whether the mean-position comparison to truth is meaningful at all — both pipelines
   baseline on the first *surviving* frame, so a different filter outcome shifts every
   position by a constant. Compare baseline-invariant statistics (std, peak-to-peak,
   slope) instead.
