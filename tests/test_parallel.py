"""The parallel entry point must produce exactly what the serial one does.

fits_reprocess_parallel imports the fit math from fits_reprocess rather than
reimplementing it. These tests keep that true, and cover the places where the
parallel file DOES have its own copy of pipeline logic.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fits_reprocess as fr
import fits_reprocess_parallel as frp

E_DRIVE = Path("E:/Reverse Telescope Test Data")


def test_parallel_imports_identical_fit_symbols():
    """Identity, not equality: a copy-paste fork would still be 'equal' by
    behaviour today and diverge silently tomorrow."""
    assert frp._fill_fit_results is fr._fill_fit_results
    assert frp._empty_row is fr._empty_row
    assert frp._load_fits_frame is fr._load_fits_frame
    assert frp._load_image_frame is fr._load_image_frame
    assert frp._extract_frame_num is fr._extract_frame_num
    assert frp._extract_timestamp is fr._extract_timestamp


def test_parallel_does_not_shadow_fit_constants():
    """The parallel module may re-import a constant, but must never define a
    diverging copy.

    It currently imports none of these -- the fit runs entirely through
    fits_reprocess -- so this is a guard against someone adding a local
    FWHM_MAX_PX that silently disagrees with the real gate.
    """
    for name in ("FWHM_FACTOR", "FWHM_MIN_PX", "FWHM_MAX_PX",
                 "PEAK_HEIGHT_FRAC", "PEAK_MIN_DISTANCE_PX"):
        if name in vars(frp):
            assert vars(frp)[name] == getattr(fr, name), (
                f"{name} diverges: parallel={vars(frp)[name]!r} "
                f"serial={getattr(fr, name)!r}")


def test_worker_failure_returns_empty_row(tmp_path):
    """A corrupt frame must yield the empty-row shape, not kill the pool."""
    bogus = tmp_path / "nope0001 26-01-01 00-00-00.fits"
    bogus.write_bytes(b"not a fits file")
    row, img, px, py = frp._fit_worker_fits(str(bogus), need_image=False)
    assert row["fit_ok"] is False
    assert np.isnan(row["mu_x"]) and np.isnan(row["mu_y"])
    assert row["frame_num"] == 1          # parsed from the name, not the payload
    assert (img, px, py) == (None, None, None)


def test_empty_row_covers_every_csv_column():
    """_write_frames_csv selects _CSV_COLS; a missing key would raise at write
    time, i.e. after a full run has been computed."""
    row = fr._empty_row("x0001 26-01-01 00-00-00.fits", 1, "2026-01-01 00:00:00")
    missing = [c for c in fr._CSV_COLS if c not in row]
    assert not missing, f"_empty_row missing CSV columns: {missing}"


def test_frame_ordering_is_stable():
    """Rows sharing a frame_num must keep discovery order.

    Reachable in 31 of ~78 runs: 22 have three rows at frame_num=-1 (older CSVs
    that ingested their own plot PNGs) and about 9 have genuine duplicate frame
    numbers among real frames. pandas' default quicksort is not stable, so ties
    land in arbitrary order -- and the position baseline is taken from the first
    surviving frame, which makes a reordering at the front shift every position.
    """
    rows = [fr._empty_row(f"f{i:03d}.fits", frame_num=7, timestamp=f"t{i}")
            for i in range(50)]
    ordered = fr._order_frames(pd.DataFrame(rows))
    assert list(ordered["filename"]) == [f"f{i:03d}.fits" for i in range(50)]


def test_order_frames_sorts_by_frame_number():
    rows = [fr._empty_row(f"f{i}.fits", frame_num=n, timestamp="")
            for i, n in enumerate([5, 1, 3, 2, 4])]
    ordered = fr._order_frames(pd.DataFrame(rows))
    assert list(ordered["frame_num"]) == [1, 2, 3, 4, 5]


@pytest.mark.needs_data
def test_serial_and_parallel_workers_agree():
    """Covers the profile-derivation code the parallel file does NOT import --
    four separate astype/sum sites in the workers, plus two in fits_reprocess.
    """
    files = sorted(glob.glob(str(
        E_DRIVE / "20260213_data/allmetal/allmetal_fits/*.fits")))
    if len(files) < 20:
        pytest.skip("allmetal frames not available")
    for path in files[:3] + [files[10], files[17]]:
        serial = fr.fit_frame(path)
        par, _img, _px, _py = frp._fit_worker_fits(path, need_image=False)
        assert set(serial) == set(par), Path(path).name
        for k in serial:
            a, b = serial[k], par[k]
            if isinstance(a, float) and np.isnan(a):
                assert isinstance(b, float) and np.isnan(b), f"{Path(path).name} {k}"
            else:
                assert a == b, f"{Path(path).name} {k}: {a!r} != {b!r}"


@pytest.mark.needs_data
def test_parallel_worker_profiles_match_direct_computation():
    """The worker's px/py must equal a straightforward flip-and-sum."""
    files = sorted(glob.glob(str(
        E_DRIVE / "20260213_data/allmetal/allmetal_fits/*.fits")))
    if not files:
        pytest.skip("allmetal frames not available")
    path = files[10]
    _row, img, px, py = frp._fit_worker_fits(path, need_image=True)
    direct = fr._load_fits_frame(path).astype(np.float64)
    assert np.allclose(px, np.sum(direct, axis=0), rtol=1e-6)
    assert np.allclose(py, np.sum(direct, axis=1), rtol=1e-6)
    assert np.array_equal(img, fr._load_fits_frame(path))
