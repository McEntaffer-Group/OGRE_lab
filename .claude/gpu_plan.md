# GPU Fitting Plan (PyTorch / CUDA)

## Goal
Replace per-frame `scipy.optimize.curve_fit` with a batched GPU fit that processes
all frames in a run in one shot. Expected speedup: 10–50x on large runs.

## Hardware
- GPU: NVIDIA RTX A2000 12GB, CUDA 13.0 (driver 581.42)
- CPU: Intel Xeon Silver 4210R, 40 logical processors (2-socket, 20-core, HT)

## Step 0: Install PyTorch
In the project venv (ReverseTelescopeDot or whichever is active for this script):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
CUDA 13.0 is forward-compatible with cu121 wheels. Verify with:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Algorithm: log-linear weighted least squares

For a 1D Gaussian with offset:
```
I(x) = A * exp(-(x - mu)^2 / (2*sigma^2)) + bg
```
After background subtracting (bg ≈ median), taking the log gives a parabola:
```
log(I - bg) = a + b*x + c*x^2
```
where: `mu = -b / (2c)`, `sigma = sqrt(-1 / (2c))`, `A = exp(a - b^2/(4c))`

Solving this via **weighted normal equations** (weights = signal amplitude) is a
batched matrix operation — one GPU call fits N profiles simultaneously.

No iteration, no per-frame Python overhead. Works well when SNR is decent.

## Code changes needed in `fits_reprocess.py`

### 1. Two new module-level loader functions
`_load_image_data(path_str)` and `_load_fits_data(path_str)` — same as the
existing fit_frame functions but **only** return the 1D projection profiles
(not fit results). These run in ProcessPoolExecutor for parallel I/O.

### 2. `_fit_batch_1d_gpu(profiles_np, device) -> (amp, mu, sigma, offset, valid)`
- Input: `(N, L)` float32 numpy array
- Builds weighted normal equation matrices on GPU (all (N,3,3), (N,3))
- Solves via `torch.linalg.solve` with small ridge (1e-6) for stability
- Returns 5 numpy arrays, all shape (N,)

### 3. `_assemble_rows_from_gpu(loaded, amp_x, mu_x, sigma_x, offset_x, ...)`
Converts GPU output arrays + loaded metadata into list of result dicts,
matching the format of `_empty_row` / `_fill_fit_results`.

### 4. `process_fits_run(run_dir, device=None)` and `process_image_run(run_dir, device=None)`
Add optional `device` parameter. When not None:
1. Load profiles in parallel with `ProcessPoolExecutor` + loader functions
2. Stack into `(N, L)` arrays
3. GPU batch fit
4. If GPU fails (numpy stack error, torch error), fall back to CPU path

### 5. `main()` detects CUDA
```python
device = None
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass
```
Pass `device=device` to both process functions.

## Expected behavior
- I/O (image loading) stays parallel on CPU via ProcessPoolExecutor
- Fitting moves to GPU — eliminates per-frame Python GIL bottleneck entirely
- `n_peaks_x/n_peaks_y` will be 0 in GPU path (informational only, acceptable)
- `fit_ok` logic unchanged: same FWHM bounds check applied to GPU fit results
- Fallback to CPU if torch not installed or CUDA unavailable

## Accuracy note
The log-linear fit is less robust than scipy's LM on very noisy or multi-peaked
profiles, but should give nearly identical results on clean single-dot images.
If accuracy is a concern, compare a subset of GPU vs CPU results using
`run_timings.csv` fps numbers and spot-check the `_frames.csv` sigma values.