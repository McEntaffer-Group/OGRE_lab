"""Run-name collisions.

Run names repeat across dates: 'minutely' is seven separate runs, 'morning5237'
is six. Twenty runs share eight names in total. Artifacts keyed on the bare run
name let a later run silently overwrite an earlier one.
"""

from pathlib import Path

import pytest

import fits_reprocess as fr

E_DRIVE = Path("E:/Reverse Telescope Test Data")
OUT_DIR = Path(__file__).resolve().parent.parent / "reprocess_output"

# Verified against all_runs_summary_reprocess.csv.
COLLIDING = {"minutely", "morning5237", "fanoff5237", "fanoff",
             "noon5237", "overnight", "laserdaytest", "warming"}


def test_run_key_distinguishes_same_name_on_different_dates():
    a = Path("E:/data/20250922_data/minutely")
    b = Path("E:/data/20251016_data/minutely")
    assert fr.run_key(a) == "20250922_minutely"
    assert fr.run_key(b) == "20251016_minutely"
    assert fr.run_key(a) != fr.run_key(b)


def test_mirrored_artifacts_do_not_clobber(tmp_path, monkeypatch):
    """The actual clobbering mechanism, exercised against a real filesystem.

    Asserts content as well as names: two files with identical content would
    mean the second write had overwritten the first upstream.
    """
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(fr, "OUTPUT_DIR", out)

    for date in ("20250922", "20251016"):
        run_dir = tmp_path / f"{date}_data" / "minutely"
        run_dir.mkdir(parents=True)
        src = run_dir / "minutely_frames.csv"
        src.write_text(f"marker\n{date}\n", encoding="utf-8")
        fr._mirror_for_run(src, run_dir)

    written = sorted(p.name for p in out.glob("*_frames.csv"))
    assert written == ["20250922_minutely_frames.csv",
                       "20251016_minutely_frames.csv"]
    for date in ("20250922", "20251016"):
        text = (out / f"{date}_minutely_frames.csv").read_text(encoding="utf-8")
        assert date in text, f"{date} mirror holds the wrong run's content"


def test_mirror_suffix_variant_is_also_date_prefixed(tmp_path, monkeypatch):
    """The _frames_prev.csv path takes a different branch through
    _mirror_for_run and must be date-prefixed too."""
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(fr, "OUTPUT_DIR", out)
    run_dir = tmp_path / "20250922_data" / "minutely"
    run_dir.mkdir(parents=True)
    src = run_dir / "minutely_frames.csv"
    src.write_text("x\n", encoding="utf-8")
    fr._mirror_for_run(src, run_dir, suffix="_frames_prev.csv")
    assert [p.name for p in out.glob("*")] == ["20250922_minutely_frames_prev.csv"]


@pytest.mark.needs_data
def test_no_two_runs_claim_the_same_mirrored_name():
    """The general invariant: catches a future collision, not just the eight
    known ones. Read-only scan of E:."""
    seen, clashes = {}, []
    runs = ([(p, False) for p in fr._discover_fits_runs(E_DRIVE)]
            + [(p, True) for p in fr._discover_image_runs(E_DRIVE)])
    for run_dir, _is_image in runs:
        for name in fr._run_output_names(run_dir.name):
            mirrored = f"{fr.date_prefix(run_dir)}_{name}"
            if mirrored in seen:
                clashes.append(f"{mirrored}: {seen[mirrored]} vs {run_dir}")
            seen[mirrored] = run_dir
    assert not clashes, "mirrored artifact names claimed twice:\n  " + "\n  ".join(clashes)


@pytest.mark.needs_data
def test_colliding_runnames_are_actually_present():
    """Guards the COLLIDING list above from going stale as runs are added."""
    from collections import Counter
    runs = ([p for p in fr._discover_fits_runs(E_DRIVE)]
            + [p for p in fr._discover_image_runs(E_DRIVE)])
    counts = Counter(p.name for p in runs)
    found = {n for n, c in counts.items() if c > 1}
    missing = COLLIDING - found
    assert not missing, (
        f"COLLIDING lists names that no longer repeat on disk: {sorted(missing)}")


def test_no_ambiguous_bare_named_mirrors():
    """A bare {runname}_frames.csv for a colliding run name holds whichever run
    finished last, so anything globbing reprocess_output/ reads one arbitrary
    run as if it were all of them.

    Red until the affected runs are reprocessed and the stale bare files are
    removed. Nothing currently deletes them.
    """
    if not OUT_DIR.exists():
        pytest.skip("reprocess_output/ not present")
    stale = []
    for p in OUT_DIR.glob("*_frames.csv"):
        stem = p.name[: -len("_frames.csv")]
        if stem.endswith("_frames_prev"):
            continue
        if not stem[:8].isdigit() and stem in COLLIDING:
            stale.append(p.name)
    assert not stale, (
        "ambiguous bare-named mirrors still present:\n  " + "\n  ".join(sorted(stale)))
