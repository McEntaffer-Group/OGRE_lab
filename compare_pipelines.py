#!/usr/bin/env python
"""
Diagnostics for fits_reprocess_parallel.py output.

Three independent reports, all runnable together (the default) or one at a time:

  --fits        Which runs had fit failures, and *why*. Classifies every
                fit_ok==False frame into a failure mode rather than just
                counting it.

  --truth       Compares reprocess_output/all_runs_summary_reprocess.csv
                (fits_reprocess_parallel) against all_runs_summary.csv
                (the bmp_to_fits.ipynb -> dot_movie-Copy3.ipynb pipeline we
                treat as truth). Categorises each overlapping run as MATCH /
                NEAR / DIVERGENT_* / AMBIGUOUS_NAME / MISSING, and says which
                cause is responsible for each disagreement.

  --timings     Parses the human-written timing log (timing_info.txt) and
                run_timings.csv to compare fits_reprocess_parallel wall time
                against the notebook pipeline on the same runs.

  --collisions  Run names used by more than one dated run. Output is now keyed
                on {date}_{runname} so these no longer overwrite each other;
                the report flags mirrors still named under the old scheme.

Usage:
    python compare_pipelines.py                    # everything
    python compare_pipelines.py --fits --truth
    python compare_pipelines.py --fits --run bridgetstatic --verbose

CSV copies of every table land in reprocess_output/diagnostics/.

Unit note: the truth notebook writes its summary columns already multiplied by
a hardcoded 0.15 arcsec/px, under *unsuffixed* names ('FWHM x'). csv_to_dotplots
writes both '(px)' and '(as)' variants. This script compares the truth columns
against the '(as)' variants, so the pixel scale cancels as long as the run's
pixel_scales.csv entry is blank (all 76 currently are -> both use 0.15).

Naming note: mirrored artifacts in reprocess_output/ are named
{YYYYMMDD}_{runname}_frames.csv. Files without the date prefix were written
before run_key existed and hold whichever same-named run finished last; they are
reported but flagged stale, and are dropped once a dated mirror for that name
exists.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT_DIR = HERE / "reprocess_output"
DIAG_DIR = OUT_DIR / "diagnostics"
TIMING_LOG = HERE / "timing_info.txt"
PIXEL_SCALES = HERE / "pixel_scales.csv"

# Must mirror fits_reprocess.py. If those constants move, these follow.
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
FWHM_MIN_PX = 1.0
FWHM_MAX_PX = 1000.0
# dot_movie-Copy3.ipynb cell 18 calls filter_fits(fwhm_min=1, fwhm_max=500).
# The tighter truth ceiling is a real source of good-frame-set divergence.
TRUTH_FWHM_MAX_PX = 500.0

# Plot PNGs written into the run directory, by csv_to_dotplots (_reprocess) or by
# dot_movie-Copy3 (bare). Kept in sync with fits_reprocess._GENERATED_PNG_RE.
PHANTOM_RE = re.compile(r"_(?:FFT|FWHM|position)(?:_reprocess)?\.png$", re.I)

# Agreement thresholds for the truth comparison (relative, on arcsec values).
REL_TOL = 1e-3      # 0.1% -> MATCH
REL_WARN = 2e-2     # 2%   -> NEAR; above this is DIVERGENT
ABS_FLOOR = 1e-6    # below this magnitude, compare absolutely not relatively


# ---------------------------------------------------------------- helpers

def _emit(df: pd.DataFrame, name: str) -> None:
    """Write a diagnostics table and say where it went."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAG_DIR / name
    df.to_csv(path, index=False)
    print(f"    -> {path.relative_to(HERE)}  ({len(df)} rows)")


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _load_image_shapes() -> dict[str, tuple[int, int]]:
    """{run_key and runname} -> (height, width) from pixel_scales.csv.

    Indexed under both so lookups work for date-prefixed mirrors and for legacy
    bare-name ones left over from before run_key existed."""
    shapes: dict[str, tuple[int, int]] = {}
    if not PIXEL_SCALES.exists():
        return shapes
    ps = pd.read_csv(PIXEL_SCALES)
    for _, r in ps.iterrows():
        m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", str(r.get("image_shape", "") or ""))
        if not m:
            continue
        shape = (int(m.group(1)), int(m.group(2)))
        for k in (r.get("run_key"), r.get("runname")):
            if isinstance(k, str) and k:
                shapes.setdefault(k, shape)
    return shapes


# Mirrored artifacts are named {YYYYMMDD}_{runname}_frames.csv since the
# collision fix. Files without the date prefix predate it and are stale.
_KEYED_RE = re.compile(r"^(\d{8})_(.+)$")


def _frames_files() -> list[tuple[Path, str, str, bool]]:
    """(path, run_key, runname, is_legacy) for each mirrored frames CSV.

    Legacy bare-name mirrors are still reported so nothing silently disappears
    from the diagnostics, but they are flagged: a run that has since been
    reprocessed appears under both names, and only the dated one is current."""
    out = []
    for p in sorted(glob.glob(str(OUT_DIR / "*_frames.csv"))):
        if p.endswith("_frames_prev.csv"):
            continue
        path = Path(p)
        stem = path.name[: -len("_frames.csv")]
        m = _KEYED_RE.match(stem)
        if m:
            out.append((path, stem, m.group(2), False))
        else:
            out.append((path, stem, stem, True))

    # A legacy file whose runname now has a dated mirror is superseded; drop it.
    dated = {rn for _, _, rn, legacy in out if not legacy}
    return [t for t in out if not (t[3] and t[2] in dated)]


# ---------------------------------------------------------------- fit report

def classify_failures(df: pd.DataFrame, shape: tuple[int, int] | None) -> pd.Series:
    """Label every row with a failure mode ('' for rows that fit fine).

    The sigma upper bound in _fit_one_profile is profile.size/2, so a fit that
    rails against it produces fwhm ~= FWHM_FACTOR * profile.size/2. That is the
    signature of a fit that locked onto the background instead of the dot.
    """
    n = len(df)
    mode = pd.Series([""] * n, index=df.index, dtype=object)

    ok = df["fit_ok"].astype(bool)
    fname = df["filename"].astype(str)

    # Rail ceilings, per axis. x profile runs along width, y along height.
    if shape is not None:
        h, w = shape
        rail_x = FWHM_FACTOR * (w / 2.0)
        rail_y = FWHM_FACTOR * (h / 2.0)
    else:
        rail_x = rail_y = np.inf

    fx, fy = df["fwhm_x"], df["fwhm_y"]
    sx, sy = df["sigma_x"], df["sigma_y"]

    railed = ((fx >= 0.98 * rail_x) | (fy >= 0.98 * rail_y)) & fx.notna() & fy.notna()
    nan_sigma = sx.isna() | sy.isna()
    too_big = (fx >= FWHM_MAX_PX) | (fy >= FWHM_MAX_PX)
    too_small = (fx <= FWHM_MIN_PX) | (fy <= FWHM_MIN_PX)
    bad_mu = ~np.isfinite(df["mu_x"]) | ~np.isfinite(df["mu_y"])

    # Order matters: most specific / most actionable first.
    is_phantom = fname.str.contains(PHANTOM_RE)
    mode[~ok & is_phantom] = "phantom_input_png"
    mode[~ok & ~is_phantom & nan_sigma] = "no_convergence"
    mode[~ok & ~is_phantom & ~nan_sigma & railed] = "sigma_railed_to_bound"
    mode[~ok & ~is_phantom & ~nan_sigma & ~railed & too_big] = "fwhm_over_max"
    mode[~ok & ~is_phantom & ~nan_sigma & ~railed & ~too_big & too_small] = "fwhm_under_min"
    mode[~ok & ~is_phantom & ~nan_sigma & ~railed & ~too_big & ~too_small & bad_mu] = "nonfinite_centroid"
    mode[~ok & (mode == "")] = "unclassified"
    return mode


def report_fits(only: str | None, verbose: bool) -> pd.DataFrame:
    _rule("FIT FAILURE REPORT")
    shapes = _load_image_shapes()

    per_run, per_frame = [], []
    files = _frames_files()
    n_legacy = sum(1 for _, _, _, legacy in files if legacy)
    for path, key, run, legacy in files:
        if only and only not in (key, run):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  !! {run}: unreadable ({exc})")
            continue
        need = {"fit_ok", "fwhm_x", "fwhm_y", "sigma_x", "sigma_y", "mu_x", "mu_y", "filename"}
        if not need.issubset(df.columns):
            print(f"  !! {run}: missing columns {sorted(need - set(df.columns))}")
            continue

        mode = classify_failures(df, shapes.get(key) or shapes.get(run))
        ok = df["fit_ok"].astype(bool)
        bad = ~ok

        # Frames the reprocess gate keeps but the truth gate (fwhm_max=500)
        # would drop -- the good-frame sets differ by exactly these.
        truth_only_drop = int(
            (ok & ((df["fwhm_x"] >= TRUTH_FWHM_MAX_PX) | (df["fwhm_y"] >= TRUTH_FWHM_MAX_PX))).sum()
        )

        counts = mode[bad].value_counts().to_dict()
        row = {
            "run_key": key,
            "runname": run,
            "stale_mirror": legacy,
            "frames": len(df),
            "good": int(ok.sum()),
            "bad": int(bad.sum()),
            "bad_pct": round(100.0 * bad.mean(), 2) if len(df) else 0.0,
            "phantom_input_png": counts.get("phantom_input_png", 0),
            "no_convergence": counts.get("no_convergence", 0),
            "sigma_railed_to_bound": counts.get("sigma_railed_to_bound", 0),
            "fwhm_over_max": counts.get("fwhm_over_max", 0),
            "fwhm_under_min": counts.get("fwhm_under_min", 0),
            "nonfinite_centroid": counts.get("nonfinite_centroid", 0),
            "unclassified": counts.get("unclassified", 0),
            "kept_but_truth_would_drop": truth_only_drop,
            "image_shape": "x".join(map(str, shapes[key])) if key in shapes
                           else ("x".join(map(str, shapes[run])) if run in shapes else ""),
        }
        # Multi-peak share among *good* frames: candidate laser-splotch + LED runs.
        if {"n_peaks_x", "n_peaks_y"}.issubset(df.columns) and ok.any():
            mp = (df["n_peaks_x"] > 1) | (df["n_peaks_y"] > 1)
            row["multipeak_pct_of_good"] = round(100.0 * mp[ok].mean(), 1)
            row["median_peaks_x_good"] = float(df.loc[ok, "n_peaks_x"].median())
        per_run.append(row)

        if bad.any():
            fr = df.loc[bad, ["frame_num", "filename", "mu_x", "mu_y", "fwhm_x", "fwhm_y"]].copy()
            fr.insert(0, "run_key", key)
            fr["failure_mode"] = mode[bad].values
            per_frame.append(fr)

    if not per_run:
        print("  no frames CSVs found.")
        return pd.DataFrame()

    runs = pd.DataFrame(per_run).sort_values("bad", ascending=False)
    frames = pd.concat(per_frame, ignore_index=True) if per_frame else pd.DataFrame()

    tot_f, tot_b = int(runs["frames"].sum()), int(runs["bad"].sum())
    print(f"  {len(runs)} runs, {tot_f:,} frames, {tot_b:,} failures "
          f"({100.0 * tot_b / max(tot_f, 1):.2f}%)")
    print(f"  runs with >=1 failure: {(runs['bad'] > 0).sum()}")
    if n_legacy:
        print(f"  note: {n_legacy} mirror(s) predate the run_key fix and have no date in")
        print("        their name. They are the last run of that name to finish under the")
        print("        old scheme; re-running those runs replaces them with dated mirrors.")

    print("\n  Failure modes across all runs:")
    for col in ["phantom_input_png", "no_convergence", "sigma_railed_to_bound",
                "fwhm_over_max", "fwhm_under_min", "nonfinite_centroid", "unclassified"]:
        tot = int(runs[col].sum())
        if tot:
            print(f"    {col:<24} {tot:>7,}   in {int((runs[col] > 0).sum())} run(s)")

    ph = runs[runs["phantom_input_png"] > 0]
    if len(ph):
        print(f"\n  [STALE DATA] {len(ph)} run(s) contain phantom frames: rows whose 'filename'")
        print("        is one of this pipeline's own plot PNGs. Image runs used to be discovered")
        print("        by globbing '*.png' in the run directory, which picked up the")
        print("        {run}_{FFT,FWHM,position}_reprocess.png csv_to_dotplots had written there")
        print("        on the previous pass -- 3 extra 'frames' per re-run.")
        print("        The glob now excludes them (fits_reprocess.list_run_images), so these are")
        print("        leftovers in CSVs written before the fix; re-running the run clears them.")
        print(f"        Phantom rows still present: {int(ph['phantom_input_png'].sum())}")

    real = runs[runs["bad"] - runs["phantom_input_png"] > 0].copy()
    real["real_bad"] = real["bad"] - real["phantom_input_png"]
    if len(real):
        print(f"\n  Runs with genuine fit failures (phantom PNGs excluded), worst first:")
        cols = ["run_key", "frames", "good", "real_bad", "bad_pct",
                "sigma_railed_to_bound", "fwhm_over_max", "image_shape"]
        print(real.sort_values("real_bad", ascending=False)[cols].to_string(index=False))

    tw = runs[runs["kept_but_truth_would_drop"] > 0]
    if len(tw):
        print(f"\n  [TRUTH GATE] {len(tw)} run(s) keep frames the notebook would drop")
        print(f"               (reprocess fwhm_max={FWHM_MAX_PX:g} vs notebook {TRUTH_FWHM_MAX_PX:g}):")
        print(tw[["run_key", "good", "kept_but_truth_would_drop"]].to_string(index=False))

    if verbose and len(frames):
        print("\n  Per-frame failures:")
        print(frames.to_string(index=False, max_colwidth=48))

    _emit(runs, "fit_failures_by_run.csv")
    if len(frames):
        _emit(frames, "fit_failures_by_frame.csv")
    return runs


# ---------------------------------------------------------------- truth compare

# (truth column, reprocess column). Truth values are already arcsec.
TRUTH_PAIRS = [
    ("x position", "x position (as)"),
    ("x position std", "x pos std (as)"),
    ("y position", "y position (as)"),
    ("y position std", "y pos std (as)"),
    ("FWHM x", "FWHM x (as)"),
    ("FWHM x std", "FWHM x std (as)"),
    ("FWHM y", "FWHM y (as)"),
    ("FWHM y std", "FWHM y std (as)"),
]


def _reldiff(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    denom = max(abs(a), abs(b))
    if denom < ABS_FLOOR:
        return abs(a - b)
    return abs(a - b) / denom


def _colliding_runnames() -> set[str]:
    """Runnames used by more than one run in the latest batch of run_timings.csv.

    A collision means reprocess_output/{runname}_* holds whichever run finished
    last, so a truth comparison keyed on runname may be comparing two different
    datasets. Cross-referenced by report_truth to explain frame-count mismatches.
    """
    tpath = OUT_DIR / "run_timings.csv"
    if not tpath.exists():
        return set()
    t = pd.read_csv(tpath)
    t = t[t["runname"] != "_TOTAL_"]
    if t.empty:
        return set()
    b = t[t["batch_timestamp"] == t["batch_timestamp"].max()]
    return set(b.loc[b.duplicated("runname", keep=False), "runname"])


def _truth_gate_drops(key: str, run: str | None = None) -> tuple[int, float]:
    """(frames the notebook's fwhm_max=500 cuts, resulting baseline shift in px).

    A run can show zero fit failures and still disagree with truth, because the
    notebook averages over a strictly smaller set of frames.

    The baseline shift matters more than the count. Both pipelines report
    position as mu - mu[0] over the *surviving* frames, so if the gate happens
    to cut frame 0, every position in the run moves by a constant and the mean
    position disagrees no matter how well the fits themselves agree. One dropped
    frame is enough to do this.
    """
    # Prefer the dated mirror; fall back to a legacy bare-name one.
    p = OUT_DIR / f"{key}_frames.csv"
    if not p.exists() and run:
        p = OUT_DIR / f"{run}_frames.csv"
    if not p.exists():
        return 0, 0.0
    try:
        df = pd.read_csv(p, usecols=["fit_ok", "fwhm_x", "fwhm_y", "mu_y"])
    except Exception:
        return 0, 0.0
    g = df[df["fit_ok"].astype(bool)].reset_index(drop=True)
    if g.empty:
        return 0, 0.0
    over = (g["fwhm_x"] >= TRUTH_FWHM_MAX_PX) | (g["fwhm_y"] >= TRUTH_FWHM_MAX_PX)
    kept = g[~over]
    shift = 0.0 if kept.empty else float(kept["mu_y"].iloc[0] - g["mu_y"].iloc[0])
    return int(over.sum()), shift


def report_truth() -> pd.DataFrame:
    _rule("REPROCESS vs TRUTH (bmp_to_fits.ipynb -> dot_movie-Copy3.ipynb)")
    collisions = _colliding_runnames()
    old_p, new_p = OUT_DIR / "all_runs_summary.csv", OUT_DIR / "all_runs_summary_reprocess.csv"
    if not old_p.exists() or not new_p.exists():
        print(f"  missing {old_p.name} or {new_p.name}")
        return pd.DataFrame()

    old, new = pd.read_csv(old_p), pd.read_csv(new_p)

    # all_runs_summary.csv is a mix: rows written by the truth notebook fill the
    # unsuffixed columns; rows written by an older csv_to_dotplots fill the
    # (px)/(as) ones. Only the former are truth.
    truth = old[old["x position"].notna()].copy()
    print(f"  all_runs_summary.csv: {len(old)} rows, {len(truth)} of them written by the notebook")
    if len(old) - len(truth):
        print(f"    ({len(old) - len(truth)} rows are csv_to_dotplots output, not truth -- skipped)")

    dup = new["runname"].duplicated(keep=False)
    if dup.any():
        print(f"  all_runs_summary_reprocess.csv: {len(new)} rows, "
              f"{new.loc[dup, 'runname'].nunique()} runname(s) shared by several dates")

    rows = []
    for _, r in truth.iterrows():
        run = r["runname"]
        cand = new[new["runname"] == run]
        if cand.empty:
            rows.append({"runname": run, "verdict": "MISSING_FROM_REPROCESS"})
            continue
        if len(cand) > 1:
            # The truth summary carries no date, so several reprocess runs can
            # match by name. Frame count identifies which dataset the notebook
            # actually ran on; if that is still ambiguous, say so rather than
            # guessing and reporting a bogus disagreement.
            exact = cand[cand["total frames"] == r.get("number of frames")]
            if len(exact) == 1:
                cand = exact
            else:
                rows.append({
                    "runname": run,
                    "truth_frames": r.get("number of frames"),
                    "name_collision": True,
                    "candidates": " | ".join(
                        f"{k}({int(n)})" for k, n in
                        zip(cand.get("run_key", cand["runname"]), cand["total frames"])),
                    "verdict": "AMBIGUOUS_NAME",
                })
                continue
        nr = cand.iloc[0]
        rec = {
            "run_key": nr.get("run_key", run),
            "runname": run,
            "truth_frames": r.get("number of frames"),
            "reprocess_total": nr.get("total frames"),
            "reprocess_good": nr.get("good frames"),
        }
        worst, worst_col = 0.0, ""
        for tc, nc in TRUTH_PAIRS:
            if tc not in truth.columns or nc not in new.columns:
                continue
            d = _reldiff(float(r[tc]), float(nr[nc]))
            rec[f"reldiff:{tc}"] = d
            if np.isfinite(d) and d > worst:
                worst, worst_col = d, tc
        rec["worst_reldiff"] = worst
        rec["worst_column"] = worst_col
        rec["name_collision"] = run in collisions
        rec["truth_gate_drops"], rec["baseline_shift_px"] = _truth_gate_drops(
            nr.get("run_key", run), run)

        tf, rt = rec["truth_frames"], rec["reprocess_total"]
        frames_differ = (
            pd.notna(tf) and pd.notna(rt) and int(tf) != int(rt)
        )
        good_differ = (
            pd.notna(rt) and pd.notna(rec["reprocess_good"])
            and int(rt) != int(rec["reprocess_good"])
        )
        if frames_differ:
            rec["verdict"] = "FRAME_COUNT_MISMATCH"
        elif worst <= REL_TOL:
            rec["verdict"] = "MATCH"
        elif worst <= REL_WARN:
            rec["verdict"] = "NEAR"
        elif good_differ:
            rec["verdict"] = "DIVERGENT_WITH_FAILURES"
        elif rec["truth_gate_drops"] > 0:
            # No fit failures, but the two pipelines still averaged different
            # frame sets because of the fwhm_max=500 vs 1000 ceiling.
            rec["verdict"] = "DIVERGENT_TRUTH_GATE"
        else:
            rec["verdict"] = "DIVERGENT_UNEXPLAINED"
        rows.append(rec)

    cmp = pd.DataFrame(rows)
    if cmp.empty:
        print("  no overlapping runs.")
        return cmp

    print(f"\n  {len(cmp)} overlapping run(s):")
    for v, n in cmp["verdict"].value_counts().items():
        print(f"    {v:<26} {n}")

    show = ["run_key", "truth_frames", "reprocess_total", "reprocess_good",
            "truth_gate_drops", "baseline_shift_px", "worst_reldiff", "worst_column", "name_collision", "verdict"]
    for verdict, note in [
        ("FRAME_COUNT_MISMATCH", "different number of input frames -- not comparable until resolved"),
        ("DIVERGENT_UNEXPLAINED", "same frames, same good count, no gate difference: a real algorithm difference"),
        ("DIVERGENT_TRUTH_GATE", f"no fit failures, but the notebook's fwhm_max={TRUTH_FWHM_MAX_PX:g} cuts frames reprocess keeps"),
        ("DIVERGENT_WITH_FAILURES", "explained by reprocess dropping frames the notebook kept (or vice versa)"),
        ("NEAR", f"agree to better than {REL_WARN:.0%} but worse than {REL_TOL:.1%}"),
        ("AMBIGUOUS_NAME", "several reprocess runs share this name and none matches truth's frame count"),
        ("MISSING_FROM_REPROCESS", "truth has it, reprocess does not"),
    ]:
        sub = cmp[cmp["verdict"] == verdict]
        if len(sub):
            print(f"\n  --- {verdict} ({len(sub)}) --- {note}")
            print(sub[[c for c in show if c in sub.columns]].to_string(index=False))
            hit = sub[sub.get("name_collision", False) == True]  # noqa: E712
            if len(hit):
                print(f"      NOTE: {', '.join(hit['runname'])} -- runname collision. The row above")
                print("      compares truth against whichever same-named run finished last, which is")
                print("      a different dataset. Not a pipeline disagreement; see --collisions.")

    matched = cmp[cmp["verdict"] == "MATCH"]
    if len(matched):
        print(f"\n  --- MATCH ({len(matched)}) --- agree to within {REL_TOL:.1%} on all 8 summary stats")
        print("      " + ", ".join(matched["runname"].tolist()))
        print(f"      worst reldiff among these: {matched['worst_reldiff'].max():.2e}")

    _emit(cmp, "truth_vs_reprocess.csv")
    return cmp


# ---------------------------------------------------------------- timings

RE_RUN_DONE = re.compile(r"^\s*--\s+(\d+)/(\d+) ok in ([\d.]+)s")
RE_RUN_HDR = re.compile(r"^\[(\d+)/(\d+)\]\s+(FITS|IMAGE)\s+(\S+)")
RE_WALL = re.compile(r"^Done\. Total wall time: ([\d.]+) min\.")
RE_INVOKE = re.compile(r"python\s+\.\\(\S+\.py)(.*)$")


def report_timings() -> pd.DataFrame:
    _rule("WALL-TIME COMPARISON")
    if not TIMING_LOG.exists():
        print(f"  {TIMING_LOG.name} not found.")
        return pd.DataFrame()

    text = TIMING_LOG.read_text(encoding="utf-8", errors="replace").splitlines()

    invocations, cur = [], None
    for line in text:
        m = RE_INVOKE.search(line)
        if m:
            if cur:
                invocations.append(cur)
            cur = {"script": m.group(1), "args": m.group(2).strip(),
                   "runs": [], "wall_min": np.nan}
            continue
        if cur is None:
            continue
        m = RE_RUN_HDR.match(line)
        if m:
            cur["_pending"] = m.group(4)
            continue
        m = RE_RUN_DONE.match(line)
        if m:
            cur["runs"].append({
                "run": cur.get("_pending", "?"),
                "n_ok": int(m.group(1)), "n_files": int(m.group(2)),
                "elapsed_s": float(m.group(3)),
            })
            continue
        m = RE_WALL.match(line)
        if m:
            cur["wall_min"] = float(m.group(1))
    if cur:
        invocations.append(cur)

    print(f"  {len(invocations)} script invocation(s) in {TIMING_LOG.name}:\n")
    inv_rows = []
    for i, inv in enumerate(invocations, 1):
        n_runs = len(inv["runs"])
        secs = sum(r["elapsed_s"] for r in inv["runs"])
        frames = sum(r["n_files"] for r in inv["runs"])
        inv_rows.append({
            "n": i, "script": inv["script"], "args": inv["args"],
            "runs": n_runs, "frames": frames,
            "sum_run_s": round(secs, 1),
            "wall_min": inv["wall_min"],
            "frames_per_s": round(frames / secs, 2) if secs else np.nan,
        })
        print(f"    [{i}] {inv['script']} {inv['args']}".rstrip())
        print(f"        {n_runs} run(s), {frames:,} frames, "
              f"sum of per-run time {secs / 60:.1f} min, wall {inv['wall_min']:.1f} min")
    invdf = pd.DataFrame(inv_rows)

    # The one apples-to-apples comparison in the log: postspie / postspiegenie,
    # which were run through the notebook pipeline AND both scripts.
    print("\n  --- Head-to-head on postspie / postspiegenie ---")
    print("  The notebook timings below are the hand-written notes in the log.\n")

    # Hand-transcribed from timing_info.txt lines 26-38. Kept explicit because
    # they are prose, not machine output.
    notebook = {
        "postspie": {
            "bmp_to_fits_min": 11 + 5 / 60,
            "load_min": 17 + 5 / 60, "fit_min": 50 / 60,
            "sum_min": 17 / 60, "movie_min": 61.0,
        },
        "postspiegenie": {
            "bmp_to_fits_min": 14 + 49 / 60,
            "load_min": 9.0, "fit_min": 1 + 5 / 60,
            "sum_min": 35 / 60, "movie_min": 94.0,
        },
    }

    par = {}
    for inv in invocations:
        if "parallel" not in inv["script"]:
            continue
        for r in inv["runs"]:
            base = r["run"].split("/")[-1]
            if base in notebook:
                par[base] = r["elapsed_s"] / 60.0  # later invocation wins

    ser = {}
    for inv in invocations:
        if "parallel" in inv["script"]:
            continue
        for r in inv["runs"]:
            base = r["run"].split("/")[-1]
            if base in notebook:
                ser[base] = r["elapsed_s"] / 60.0

    rows = []
    for run, nb in notebook.items():
        nb_total = sum(nb.values())
        rows.append({
            "run": run,
            "notebook_total_min": round(nb_total, 1),
            "  bmp_to_fits": round(nb["bmp_to_fits_min"], 1),
            "  load": round(nb["load_min"], 1),
            "  fit": round(nb["fit_min"], 2),
            "  movie": round(nb["movie_min"], 1),
            "serial_min": round(ser[run], 1) if run in ser else np.nan,
            "parallel_min": round(par[run], 1) if run in par else np.nan,
            "speedup_vs_notebook": round(nb_total / par[run], 2) if run in par else np.nan,
        })
    head = pd.DataFrame(rows)
    print(head.to_string(index=False))

    nb_tot = sum(sum(v.values()) for v in notebook.values())
    par_tot = sum(par.values()) if par else np.nan
    ser_tot = sum(ser.values()) if ser else np.nan
    print(f"\n  Both runs together ({sum(1 for _ in notebook)} runs, ~11.5k frames):")
    print(f"    notebook pipeline : {nb_tot:6.1f} min")
    if np.isfinite(ser_tot):
        print(f"    fits_reprocess    : {ser_tot:6.1f} min  (fit only; movie time is extra)")
    if np.isfinite(par_tot):
        print(f"    ..._parallel      : {par_tot:6.1f} min  ({nb_tot / par_tot:.1f}x faster than notebook)")
        movie_tot = sum(v["movie_min"] for v in notebook.values())
        fit_tot = sum(v["fit_min"] for v in notebook.values())
        print(f"\n  Note: the parallel number INCLUDES movie encoding (frames are streamed to")
        print(f"  ffmpeg via grab_frame as they are fitted). In the notebook the movie is a")
        print(f"  separate serial pass costing {movie_tot:.0f} min of the {nb_tot:.0f} min total, against only")
        print(f"  {fit_tot:.1f} min of actual curve fitting. Overlapping the animation with the fit,")
        print(f"  not speeding up the fit, is where essentially all of the gain comes from.")

    _emit(invdf, "timing_invocations.csv")
    _emit(head, "timing_head_to_head.csv")
    return head


# ---------------------------------------------------------------- collisions

def report_collisions() -> pd.DataFrame:
    _rule("RUNNAME COLLISIONS (same name, different date -> overwritten output)")
    tpath = OUT_DIR / "run_timings.csv"
    if not tpath.exists():
        print(f"  {tpath.name} not found.")
        return pd.DataFrame()
    t = pd.read_csv(tpath)
    t = t[t["runname"] != "_TOTAL_"]

    # One physical run == one source path. Keying on the latest batch alone would
    # miss collisions whenever the last batch was a partial re-run of a few runs.
    b = t.sort_values("batch_timestamp").drop_duplicates("source", keep="last")
    print(f"  {len(b)} distinct run(s) across all batches in {tpath.name}")

    dup = b[b.duplicated("runname", keep=False)].sort_values(["runname", "source"])
    if dup.empty:
        print("  no shared run names.")
        return dup

    print(f"\n  {dup['runname'].nunique()} runname(s) shared by {len(dup)} distinct runs.")
    print("  Mirrored artifacts are now keyed on {date}_{runname}, so these no longer")
    print("  overwrite each other. Mirrors written before that fix are still bare-named and")
    print("  hold whichever run finished last; re-running a run replaces it with a dated one.")
    print("  The per-run copies under E:\\...\\{date}\\{run}\\ were never affected.\n")
    cols = [c for c in ["run_key", "runname", "camera", "n_files", "n_ok", "source"]
            if c in dup.columns]
    print(dup[cols].to_string(index=False))

    unkeyed = dup[dup.get("run_key", pd.Series(dtype=object)).fillna("") == ""]
    if len(unkeyed):
        print(f"\n  {len(unkeyed)} of these were last processed before the fix (no run_key),")
        print("  so their mirrors in reprocess_output/ are still colliding by name.")

    _emit(dup, "runname_collisions.csv")
    return dup


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", action="store_true", help="fit-failure report")
    ap.add_argument("--truth", action="store_true", help="reprocess vs notebook comparison")
    ap.add_argument("--timings", action="store_true", help="wall-time comparison")
    ap.add_argument("--collisions", action="store_true", help="runname collision report")
    ap.add_argument("--run", default=None, help="restrict --fits to one runname")
    ap.add_argument("--verbose", action="store_true", help="list every failing frame")
    a = ap.parse_args(argv)

    if not (a.fits or a.truth or a.timings or a.collisions):
        a.fits = a.truth = a.timings = a.collisions = True

    if a.fits:
        report_fits(a.run, a.verbose)
    if a.truth:
        report_truth()
    if a.timings:
        report_timings()
    if a.collisions:
        report_collisions()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
