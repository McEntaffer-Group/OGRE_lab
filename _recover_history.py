"""One-shot recovery: rebuild pixel_scales.csv and reprocess_output/run_timings.csv
from git history + current on-disk state. After running once, this file can be deleted.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent

PIXEL_SCALES_HISTORY = [
    # (commit, date) newest last so later versions win on tie
    ("ef7bbe9", "2026-06-12"),
    ("dc0408e", "2026-06-15"),
    ("baff4d9", "2026-06-25"),
    ("9568eaf", "2026-08-17"),
    ("f580345", "2026-08-18"),
]
RUN_TIMINGS_HISTORY = [
    ("9568eaf", "2026-08-17"),
]


def _git_show(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"],
        text=True, encoding="utf-8",
    )


def recover_pixel_scales() -> None:
    frames = []
    for commit, date in PIXEL_SCALES_HISTORY:
        text = _git_show(commit, "pixel_scales.csv")
        df = pd.read_csv(StringIO(text))
        df["_source_date"] = date
        frames.append(df)

    current = REPO / "pixel_scales.csv"
    if current.exists():
        df_cur = pd.read_csv(current)
        df_cur["_source_date"] = "current"
        frames.append(df_cur)

    all_rows = pd.concat(frames, ignore_index=True)

    all_rows["_has_scale"] = all_rows["pixel_scale_arcsec_per_pixel"].notna() & (
        all_rows["pixel_scale_arcsec_per_pixel"].astype(str).str.strip() != ""
    )
    all_rows["_priority"] = all_rows["_has_scale"].astype(int)

    all_rows = all_rows.sort_values(
        ["runname", "_priority", "_source_date"], ascending=[True, False, False]
    )
    merged = all_rows.drop_duplicates(subset=["runname"], keep="first").copy()
    merged = merged.drop(columns=["_has_scale", "_priority", "_source_date"])
    merged = merged.sort_values("runname").reset_index(drop=True)

    out_cols = ["runname", "camera", "image_shape", "pixel_scale_arcsec_per_pixel", "notes"]
    merged[out_cols].to_csv(current, index=False)
    print(f"pixel_scales.csv: {len(merged)} unique runnames written")


def recover_run_timings() -> None:
    frames = []
    for commit, date in RUN_TIMINGS_HISTORY:
        text = _git_show(commit, "reprocess_output/run_timings.csv")
        df = pd.read_csv(StringIO(text))
        df["batch_timestamp"] = date
        frames.append(df)

    current = REPO / "reprocess_output" / "run_timings.csv"
    if current.exists():
        df_cur = pd.read_csv(current)
        df_cur["batch_timestamp"] = "2026-08-18"  # timestamp of the overwriting run
        frames.append(df_cur)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["runname"] != "_TOTAL_"].reset_index(drop=True)

    out_cols = ["batch_timestamp", "runname", "camera", "n_files", "n_ok",
                "elapsed_s", "fps", "source"]
    all_rows[out_cols].to_csv(current, index=False)
    print(f"run_timings.csv: {len(all_rows)} rows written (excluding _TOTAL_ summary rows)")


if __name__ == "__main__":
    recover_pixel_scales()
    recover_run_timings()