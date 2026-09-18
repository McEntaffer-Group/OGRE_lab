"""Extract 1-D profiles from every run into committed .npz files.

Run once with E: mounted; re-run when runs are added. The suite then runs with
E: unmounted.

    D:/Users/jad507/PycharmProjects/ReverseTelescopeDot/.venv/Scripts/python.exe \
        -X utf8 tests/make_fixtures.py

E: IS READ-ONLY. This script opens files on E: for reading only. It never
writes, creates, moves or deletes anything there. All output goes to
tests/fixtures/ inside the repo.

Per-run frame CSVs are read from each run's own directory, never from
reprocess_output/ -- 20 runs share 8 bare-named files there, so a CSV picked up
from that directory may describe a different run of the same name. See
TEST_PLAN.md section 1.

Three outputs:
  healthy.npz  one fit_ok frame per run (the median one, not the first)
  failing.npz  one fit_ok=False frame per run that has any
  anchors.npz  frames whose correct answer is independently known
  index.csv    per-run frame counts as of extraction, for drift detection
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import fits_reprocess as fr  # noqa: E402

E = Path("E:/Reverse Telescope Test Data")
DEST = Path(__file__).resolve().parent / "fixtures"

# Frames whose correct answer is known independently of the pipeline, so a test
# can assert a value rather than only an invariant. Indices are 0-based.
#   allmetal 10/17  - the two railing anchors; mu_y/sigma_y cross-checked
#                     against an unbounded LM fit to 4 decimals
#   allmetal 0/100/500 - healthy; fit centroid matches the 2D dot to 1.3 px,
#                     which is the single-dot control for the two-dot detector
#   postwinterbreak - two dots, verified in 2D. Three frames at different drift
#                     positions so a detector cannot pass by lucky positioning
#   frosty 0        - two dots, verified in 2D (compact source HWHM 3 px, 508 px
#                     from where the fit lands)
#   genieshots      - saturated (peak 255, radial profile flat past 36 px);
#                     a third category, distinct from two-dot
ANCHORS = [
    ("allmetal_f001", "20260213_data/allmetal", 0),
    ("allmetal_f011", "20260213_data/allmetal", 10),
    ("allmetal_f018", "20260213_data/allmetal", 17),
    ("allmetal_f101", "20260213_data/allmetal", 100),
    ("allmetal_f501", "20260213_data/allmetal", 500),
    ("postwinterbreak_f001", "20260105_data/postwinterbreak", 0),
    ("postwinterbreak_f501", "20260105_data/postwinterbreak", 500),
    ("postwinterbreak_f5001", "20260105_data/postwinterbreak", 5000),
    ("frosty_f001", "20251210_data/frosty", 0),
    ("genieshots_mid", "20260302_data/genieshots", None),  # None -> middle frame
]

_PLOT_RE = r"_(?:FFT|FWHM|position)(?:_reprocess)?\.png$"


def profiles_from(path: Path, is_image: bool):
    """Read one frame and return (px, py). Uses the pipeline's own loaders so
    the fixtures are provably what production sees, flip included."""
    load = fr._load_image_frame if is_image else fr._load_fits_frame
    img = load(path).astype(np.float64)
    return np.sum(img, axis=0), np.sum(img, axis=1)


def frame_dir(run_dir: Path, is_image: bool) -> Path:
    return run_dir if is_image else run_dir / f"{run_dir.name}_fits"


def all_runs():
    """(run_dir, is_image) for every discoverable run. Read-only."""
    runs = [(p, False) for p in fr._discover_fits_runs(E)]
    runs += [(p, True) for p in fr._discover_image_runs(E)]
    return sorted(runs, key=lambda t: fr.run_key(t[0]))


def build_survey(limit=None):
    healthy, failing, index = {}, {}, []
    runs = all_runs()
    if limit:
        runs = runs[:limit]
    print(f"discovered {len(runs)} runs")

    for n, (run_dir, is_image) in enumerate(runs, 1):
        key = fr.run_key(run_dir)
        csv = fr.output_dir_for(run_dir, is_image) / f"{run_dir.name}_frames.csv"
        if not csv.exists():
            print(f"  [{n:3d}/{len(runs)}] {key}: no per-run CSV, skipped")
            continue
        try:
            d = pd.read_csv(csv)
        except Exception as exc:
            print(f"  [{n:3d}/{len(runs)}] {key}: unreadable CSV ({exc})")
            continue
        if "fit_ok" not in d.columns or "filename" not in d.columns:
            print(f"  [{n:3d}/{len(runs)}] {key}: CSV missing columns, skipped")
            continue

        # Drop the pipeline's own plot PNGs, which older runs ingested as frames.
        d = d[~d["filename"].astype(str).str.contains(_PLOT_RE, regex=True, na=False)]
        fdir = frame_dir(run_dir, is_image)

        ok = d[d.fit_ok == True]  # noqa: E712
        bad = d[d.fit_ok == False]  # noqa: E712
        got_ok = got_bad = False

        if len(ok):
            row = ok.iloc[len(ok) // 2]
            p = fdir / str(row.filename)
            if p.exists():
                px, py = profiles_from(p, is_image)
                healthy[f"{key}_px"], healthy[f"{key}_py"] = px, py
                got_ok = True

        if len(bad):
            for _, row in bad.iterrows():
                p = fdir / str(row.filename)
                if p.exists():
                    px, py = profiles_from(p, is_image)
                    failing[f"{key}_px"], failing[f"{key}_py"] = px, py
                    got_bad = True
                    break

        index.append({"run_key": key, "is_image": is_image, "n": len(d),
                      "n_ok": len(ok), "n_bad": len(bad),
                      "has_healthy": got_ok, "has_failing": got_bad})
        print(f"  [{n:3d}/{len(runs)}] {key}: {len(d)} frames, "
              f"{len(ok)} ok / {len(bad)} bad -> "
              f"{'H' if got_ok else '-'}{'F' if got_bad else '-'}")

    return healthy, failing, index


def build_anchors():
    out = {}
    for key, sub, idx in ANCHORS:
        run_dir = E / sub
        if not run_dir.exists():
            print(f"  anchor {key}: {run_dir} missing, skipped")
            continue
        fdir = frame_dir(run_dir, False)
        files = sorted(fdir.glob("*.fits"))
        if not files:
            print(f"  anchor {key}: no FITS in {fdir}, skipped")
            continue
        i = len(files) // 2 if idx is None else idx
        if i >= len(files):
            print(f"  anchor {key}: index {i} out of range ({len(files)}), skipped")
            continue
        px, py = profiles_from(files[i], False)
        out[f"{key}_px"], out[f"{key}_py"] = px, py
        print(f"  anchor {key}: frame {i + 1}/{len(files)}  "
              f"px={px.size} py={py.size}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N runs (for a quick check)")
    ap.add_argument("--anchors-only", action="store_true")
    args = ap.parse_args()

    if not E.exists():
        print(f"E: not mounted at {E} -- cannot extract fixtures")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)

    print("=== anchors ===")
    anchors = build_anchors()
    if anchors:
        np.savez_compressed(DEST / "anchors.npz", **anchors)

    if not args.anchors_only:
        print("\n=== survey ===")
        healthy, failing, index = build_survey(args.limit)
        if healthy:
            np.savez_compressed(DEST / "healthy.npz", **healthy)
        if failing:
            np.savez_compressed(DEST / "failing.npz", **failing)
        if index:
            pd.DataFrame(index).to_csv(DEST / "index.csv", index=False)
        print(f"\nhealthy: {len(healthy) // 2} runs, failing: {len(failing) // 2} runs")

    print("\nwrote:")
    for f in sorted(DEST.glob("*")):
        print(f"  {f.name:16s} {f.stat().st_size / 1024:9.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
