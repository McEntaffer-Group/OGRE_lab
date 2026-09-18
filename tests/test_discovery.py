"""Run discovery: which folders get processed, and which are skipped.

The bug these pin: a {date}_data/ sibling used to make _discover_image_runs skip
the *entire* {date}/ folder. But a _data folder only means some of that date was
converted to FITS -- a run that was never converted is still real, unprocessed
data, and it was invisible to both discovery paths. On the real corpus that hid
one run (20250923/collimationtests, 16 frames); the rule was one partially
converted date away from dropping a large one.

Trees are built under tmp_path rather than read off E:, which is read-only and
slow to walk.
"""

from pathlib import Path

import pytest

import fits_reprocess as fr


def make_image_run(root: Path, date: str, run: str, n: int = 3) -> Path:
    d = root / date / run
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"{run}{i:04d} 26-01-01 00-00-00.bmp").write_bytes(b"")
    return d


def make_fits_run(root: Path, date: str, run: str, n: int = 3,
                  fits_name: str | None = None) -> Path:
    d = root / f"{date}_data" / run
    fdir = d / (fits_name or f"{run}_fits")
    fdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (fdir / f"{run}{i:04d}.fits").write_bytes(b"")
    return d


def names(paths):
    return sorted(p.name for p in paths)


# --- the regression -------------------------------------------------------

def test_unconverted_run_is_found_even_when_the_date_has_a_data_sibling(tmp_path):
    """The exact shape of 20250923: seven runs converted, one not."""
    make_image_run(tmp_path, "20250923", "fanoff")
    make_image_run(tmp_path, "20250923", "collimationtests")
    make_fits_run(tmp_path, "20250923", "fanoff")

    img = fr._discover_image_runs(tmp_path)
    fits = fr._discover_fits_runs(tmp_path)

    assert names(img) == ["collimationtests"], (
        "a run with no FITS conversion must still be discovered as an image run"
    )
    assert names(fits) == ["fanoff"]


def test_whole_date_is_not_skipped_for_one_conversion(tmp_path):
    for run in ("a", "b", "c", "d"):
        make_image_run(tmp_path, "20250923", run)
    make_fits_run(tmp_path, "20250923", "a")

    assert names(fr._discover_image_runs(tmp_path)) == ["b", "c", "d"]


# --- the preference rule --------------------------------------------------

def test_fits_is_preferred_over_images_of_the_same_name(tmp_path):
    """A converted run is processed once, from FITS, never twice."""
    make_image_run(tmp_path, "20260814", "postspie", n=5)
    make_fits_run(tmp_path, "20260814", "postspie", n=5)

    img = fr._discover_image_runs(tmp_path)
    fits = fr._discover_fits_runs(tmp_path)

    assert img == []
    assert names(fits) == ["postspie"]


def test_fits_is_preferred_even_when_it_holds_fewer_frames(tmp_path):
    """20260320/statictestgenie: 7018 images, 5676 FITS, conversion truncated.

    Preferring FITS is deliberate -- FITS carry the headers the pipeline reads --
    so the shortfall must be reported, not resolved by silently switching to the
    image folder.
    """
    make_image_run(tmp_path, "20260320", "statictestgenie", n=10)
    make_fits_run(tmp_path, "20260320", "statictestgenie", n=4)

    assert fr._discover_image_runs(tmp_path) == []
    assert names(fr._discover_fits_runs(tmp_path)) == ["statictestgenie"]

    kinds = {k: d for k, _, d in fr.discovery_anomalies(tmp_path)}
    assert "partial_conversion" in kinds
    assert "6 frame(s) not processed" in kinds["partial_conversion"]


def test_case_differing_names_are_treated_as_the_same_run(tmp_path):
    """20250923/bigShakeB vs 20250923_data/bigshakeB.

    Without casefolding the image copy is discovered as a second run and the
    same data is processed twice under two keys.
    """
    make_image_run(tmp_path, "20250923", "bigShakeB")
    make_fits_run(tmp_path, "20250923", "bigshakeB")

    assert fr._discover_image_runs(tmp_path) == []
    assert names(fr._discover_fits_runs(tmp_path)) == ["bigshakeB"]

    kinds = [k for k, _, _ in fr.discovery_anomalies(tmp_path)]
    assert "case_mismatch" in kinds


def test_a_date_with_no_data_sibling_is_unaffected(tmp_path):
    make_image_run(tmp_path, "20251020", "minutely")
    make_image_run(tmp_path, "20251020", "morning5237")

    assert names(fr._discover_image_runs(tmp_path)) == ["minutely", "morning5237"]
    assert fr.discovery_anomalies(tmp_path) == []


# --- anomalies ------------------------------------------------------------

def test_mismatched_fits_folder_is_reported_not_silently_processed(tmp_path):
    """20260306_data/springgenie_big/ holds springgenie_fits/, not
    springgenie_big_fits/.

    process_fits_run builds {run}/{run}_fits, so discovering it would hand the
    processor a path it cannot read. Skipping is correct; skipping silently is
    not.
    """
    make_fits_run(tmp_path, "20260306", "springgenie_big", n=7,
                  fits_name="springgenie_fits")

    assert fr._discover_fits_runs(tmp_path) == []

    found = [(k, d) for k, _, d in fr.discovery_anomalies(tmp_path)]
    assert any(k == "mismatched_fits" for k, _ in found)
    detail = next(d for k, d in found if k == "mismatched_fits")
    assert "springgenie_fits" in detail and "7 .fits" in detail


def test_fits_dir_for_requires_an_exact_name_match(tmp_path):
    run = make_fits_run(tmp_path, "20260306", "big", fits_name="other_fits")
    assert fr.fits_dir_for(run) is None

    run2 = make_fits_run(tmp_path, "20260306", "good")
    assert fr.fits_dir_for(run2) == run2 / "good_fits"


def test_stale_mirror_csv_in_the_image_tree_is_reported(tmp_path):
    """20260814/postspie/postspie_frames.csv: the run is processed from FITS,
    so nothing rewrites this copy and it keeps whatever an older build wrote."""
    d = make_image_run(tmp_path, "20260814", "postspie")
    make_fits_run(tmp_path, "20260814", "postspie")
    (d / "postspie_frames.csv").write_text("frame_num,mu_x\n0,1.0\n")

    found = [(k, p) for k, p, _ in fr.discovery_anomalies(tmp_path)]
    assert any(k == "stale_mirror" and p.name == "postspie_frames.csv"
               for k, p in found)


def test_no_anomalies_on_a_clean_tree(tmp_path):
    make_image_run(tmp_path, "20251020", "minutely")
    make_fits_run(tmp_path, "20250923", "fanoff")
    make_image_run(tmp_path, "20250923", "fanoff")

    assert fr.discovery_anomalies(tmp_path) == []


# --- the real corpus ------------------------------------------------------

E_DRIVE = Path("E:/Reverse Telescope Test Data")
needs_e = pytest.mark.skipif(not E_DRIVE.is_dir(), reason="E: not mounted")


@needs_e
def test_no_run_is_discovered_by_both_paths():
    """A run processed twice writes its CSV twice, and the second write wins."""
    fits = fr._discover_fits_runs(E_DRIVE)
    img = fr._discover_image_runs(E_DRIVE)

    assert set(fits) & set(img) == set()

    both = [(p, q) for p in img for q in fits
            if p.name.casefold() == q.name.casefold()
            and fr.date_prefix(p) == fr.date_prefix(q)]
    assert both == [], f"same run reachable both ways: {both}"


@needs_e
def test_collimationtests_is_discovered():
    """The run the old rule hid. Pinned by name because it is the only
    surviving instance of the pattern on the real corpus."""
    img = fr._discover_image_runs(E_DRIVE)
    assert any(p.name == "collimationtests" and fr.date_prefix(p) == "20250923"
               for p in img)
