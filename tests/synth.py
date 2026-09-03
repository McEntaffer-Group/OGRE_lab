"""Synthetic 1-D profiles for resolution-independent tests.

Defaults match measured allmetal values: a ~4% dot contrast on a 21000 pedestal
with RMS noise 60. The noise figure matters -- an earlier investigation tested
against invented noise of 174 and concluded the estimator needed smoothing and
SNR gating that the real data does not need. 60 is the measured residual.
"""

import numpy as np


def synth(size, mu_frac=0.523, sigma=4.0, amp=926.0, offset=21000.0,
          noise=60.0, seed=0):
    """A realistic profile at arbitrary resolution.

    mu is given as a *fraction* of size so the spot sits off-centre at every
    resolution rather than landing on a symmetry point. Returns (profile, mu).
    """
    x = np.arange(size, dtype=np.float64)
    mu = mu_frac * size
    prof = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset
    if noise:
        prof = prof + np.random.default_rng(seed).normal(0.0, noise, size)
    return prof, mu


def one_hot(size, index=None, base=21000.0, spike=5e4):
    """Flat profile with a single hot pixel."""
    prof = np.full(size, base, dtype=np.float64)
    prof[size // 2 if index is None else index] = spike
    return prof


def two_peaks(size, sigma=4.0, amp=926.0, offset=21000.0, noise=60.0, seed=0):
    """Two equal narrow peaks at 30% and 70% of the frame."""
    a, _ = synth(size, mu_frac=0.3, sigma=sigma, amp=amp, offset=offset,
                 noise=noise, seed=seed)
    b, _ = synth(size, mu_frac=0.7, sigma=sigma, amp=amp, offset=0.0, noise=0.0)
    return a + b


def two_components(size, narrow_sigma=2.1, broad_sigma=None, narrow_frac=0.33,
                   broad_frac=0.52, narrow_amp=600.0, broad_amp=20000.0,
                   offset=21000.0, noise=60.0, seed=0):
    """A laser-style broad blob plus a small real dot, well separated.

    Amplitudes are chosen to reproduce the leverage gap measured on
    20260105/postwinterbreak: the narrow component contributes a few hundred
    counts against a broad component of tens of thousands, so an unweighted
    least-squares fit locks onto the broad one.
    """
    if broad_sigma is None:
        broad_sigma = size / 6.0
    x = np.arange(size, dtype=np.float64)
    mu_n, mu_b = narrow_frac * size, broad_frac * size
    prof = (narrow_amp * np.exp(-0.5 * ((x - mu_n) / narrow_sigma) ** 2)
            + broad_amp * np.exp(-0.5 * ((x - mu_b) / broad_sigma) ** 2)
            + offset)
    if noise:
        prof = prof + np.random.default_rng(seed).normal(0.0, noise, size)
    return prof, mu_n, mu_b
