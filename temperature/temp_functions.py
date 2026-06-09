"""
temp_functions.py — utilities for loading and querying OGRE lab temperature/humidity data.

Primary usage:
    import temp_functions as tf
    df = tf.builder("2025-12-10 15:00", "2025-12-15 11:24")
    df = tf.builder("2025-12-10", "2025-12-16", resample_freq="1min")

All functions return DataFrames with Schema B columns indexed by Timestamp:
    SHT_Temperature_C, MCP_Temperature_C, HDC_Temperature_C,
    SHT_Relative_Humidity, HDC_Relative_Humidity
"""

import os
import glob
import re
from datetime import date, timedelta

import pandas as pd

# Default directory — same folder as this file
_DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA_B_COLS = [
    "SHT_Temperature_C", "MCP_Temperature_C", "HDC_Temperature_C",
    "SHT_Relative_Humidity", "HDC_Relative_Humidity",
]

# Matches temperature_log_YYYY-MM-DD.csv produced by temperature_logger3
_DATE_FILENAME_RE = re.compile(r"temperature_log_(\d{4}-\d{2}-\d{2})\.csv$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(path):
    """Load one CSV and normalize to Schema B, indexed by Timestamp."""
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.set_index("Timestamp")

    if "Temperature_C" in df.columns and "SHT_Temperature_C" not in df.columns:
        # Schema A (legacy single-sensor): map to MCP column, rest NaN
        legacy_temp = df["Temperature_C"]
        df = df.drop(columns=["Temperature_C"])
        for col in SCHEMA_B_COLS:
            df[col] = float("nan")
        df["MCP_Temperature_C"] = legacy_temp

    # Ensure all Schema B columns exist (handles partially disabled sensors)
    for col in SCHEMA_B_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    return df[SCHEMA_B_COLS]


def _date_range_from_filename(path):
    """Return (date, date) if filename encodes a date, else None."""
    m = _DATE_FILENAME_RE.search(os.path.basename(path))
    if m:
        d = date.fromisoformat(m.group(1))
        return d, d
    return None


def _date_range_from_csv(path):
    """Read first and last timestamps from a CSV without loading it all."""
    try:
        first_row = pd.read_csv(path, nrows=1, parse_dates=["Timestamp"])
        if first_row.empty:
            return None
        # Read last row efficiently
        last_row = pd.read_csv(path, parse_dates=["Timestamp"]).tail(1)
        t_start = first_row["Timestamp"].iloc[0].date()
        t_end = last_row["Timestamp"].iloc[0].date()
        return t_start, t_end
    except Exception:
        return None


def _overlaps(file_start, file_end, query_start, query_end):
    return file_start <= query_end.date() and file_end >= query_start.date()


def _candidate_files(source_dir, start, end):
    """Return list of CSV paths whose date range overlaps [start, end]."""
    csvs = glob.glob(os.path.join(source_dir, "*.csv"))
    candidates = []
    for path in csvs:
        date_range = _date_range_from_filename(path)
        if date_range is None:
            date_range = _date_range_from_csv(path)
        if date_range is None:
            continue
        if _overlaps(date_range[0], date_range[1], start, end):
            candidates.append(path)
    return candidates


def _find_gaps(df, threshold):
    """Return list of (gap_start, gap_end, duration) for gaps larger than threshold."""
    if len(df) < 2:
        return []
    deltas = df.index.to_series().diff().iloc[1:]
    big = deltas[deltas > threshold]
    gaps = []
    for ts, dur in big.items():
        gap_start = df.index[df.index.get_loc(ts) - 1]
        gaps.append((gap_start, ts, dur))
    return gaps


def _warn_gaps(df, threshold=pd.Timedelta("5min")):
    gaps = _find_gaps(df, threshold)
    if gaps:
        print(f"WARNING: {len(gaps)} gap(s) found in requested time range:")
        for gap_start, gap_end, dur in gaps:
            print(f"  {gap_start}  ->  {gap_end}  ({_fmt_duration(dur)} missing)")


def _fmt_duration(td):
    total = int(td.total_seconds())
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s and not d:
        parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def builder(start, end, source_dir=None, resample_freq=None):
    """
    Return a DataFrame of temperature/humidity data between start and end.

    Parameters
    ----------
    start, end : str or datetime-like
        Anything pandas.Timestamp can parse, e.g. "2025-12-10 15:00" or a datetime object.
    source_dir : str, optional
        Directory to search for CSV files. Defaults to the temperature/ folder.
    resample_freq : str, optional
        If provided (e.g. '1min', '5min', '1h'), resample to that interval mean.
        If None, returns raw 1-second data.

    Returns
    -------
    pd.DataFrame indexed by Timestamp with Schema B columns.
    """
    if source_dir is None:
        source_dir = _DEFAULT_DIR

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    candidates = _candidate_files(source_dir, start, end)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {source_dir} covering {start} to {end}. "
            "Run split_to_daily() first, or check your source_dir."
        )

    frames = []
    for path in sorted(candidates):
        df = _load_csv(path)
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    combined = combined.loc[start:end]

    if combined.empty:
        raise ValueError(f"No data found between {start} and {end}.")

    _warn_gaps(combined, threshold=pd.Timedelta("5min"))

    if resample_freq is not None:
        combined = combined.resample(resample_freq).mean()

    return combined


def split_to_daily(source_dir=None, output_dir=None):
    """
    Read every CSV in source_dir, split all rows by calendar date, and write
    one file per day named temperature_log_YYYY-MM-DD.csv into output_dir.

    Does NOT delete originals. Safe to re-run (appends are avoided by skipping
    dates whose output file already exists).

    Parameters
    ----------
    source_dir : str, optional
        Directory containing existing CSVs. Defaults to temperature/ folder.
    output_dir : str, optional
        Where to write daily files. Defaults to source_dir/daily/.
    """
    if source_dir is None:
        source_dir = _DEFAULT_DIR
    if output_dir is None:
        output_dir = os.path.join(source_dir, "daily")
    os.makedirs(output_dir, exist_ok=True)

    csvs = glob.glob(os.path.join(source_dir, "*.csv"))
    if not csvs:
        print(f"No CSV files found in {source_dir}")
        return

    print(f"Loading {len(csvs)} CSV files...")
    frames = []
    for path in csvs:
        try:
            df = _load_csv(path)
            nat_mask = df.index.isna()
            nat_count = nat_mask.sum()
            if nat_count:
                name = os.path.basename(path)
                total = len(df)
                # Check if all NaT rows are at the tail of the file
                last_good = df[~nat_mask].index[-1] if nat_count < total else None
                nat_indices = df.index[nat_mask]
                trailing = nat_mask[::-1].cumsum()[::-1] <= nat_count  # True for last nat_count rows
                all_trailing = nat_mask[trailing].all() and not nat_mask[~trailing].any()
                if nat_count <= 5 and all_trailing:
                    print(f"  {name}: dropping {nat_count} trailing NaT row(s) (likely incomplete write at shutdown)")
                else:
                    print(f"  WARNING: {name} has {nat_count} NaT-timestamped rows out of {total:,} total.")
                    print(f"    Last valid timestamp: {last_good}")
                    print(f"    This may indicate real corruption — inspect this file manually.")
                df = df[~nat_mask]
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")

    if not frames:
        print("No data loaded.")
        return

    all_data = pd.concat(frames).sort_index()
    all_data = all_data[~all_data.index.duplicated(keep='first')]

    days = all_data.index.normalize().unique()
    print(f"Writing {len(days)} daily files to {output_dir}/")

    for day in days:
        out_path = os.path.join(output_dir, f"temperature_log_{day.date()}.csv")
        if os.path.exists(out_path):
            print(f"  {day.date()} already exists, skipping.")
            continue
        day_data = all_data.loc[str(day.date())]
        day_data.to_csv(out_path)
        print(f"  Wrote {day.date()} ({len(day_data):,} rows)")

    print("Done.")


def list_runs(source_dir=None):
    """
    Print and return a summary of all date ranges available across CSVs.

    Returns
    -------
    list of (filename, start_timestamp, end_timestamp)
    """
    if source_dir is None:
        source_dir = _DEFAULT_DIR

    csvs = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    runs = []
    for path in csvs:
        rng = _date_range_from_filename(path)
        if rng is None:
            rng = _date_range_from_csv(path)
        name = os.path.basename(path)
        if rng:
            print(f"  {name}: {rng[0]} -> {rng[1]}")
            runs.append((name, rng[0], rng[1]))
        else:
            print(f"  {name}: (unreadable)")
    return runs


def check_gaps(df, threshold="5min"):
    """
    Print and return all gaps in a builder() DataFrame larger than threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Output from builder(), indexed by Timestamp.
    threshold : str or pd.Timedelta
        Minimum gap size to report. Default "5min".

    Returns
    -------
    list of (gap_start, gap_end, duration) tuples.
    """
    threshold = pd.Timedelta(threshold)
    gaps = _find_gaps(df, threshold)
    if not gaps:
        print(f"No gaps larger than {threshold} found.")
    else:
        print(f"{len(gaps)} gap(s) larger than {threshold}:")
        for gap_start, gap_end, dur in gaps:
            print(f"  {gap_start}  ->  {gap_end}  ({_fmt_duration(dur)} missing)")
    return gaps


def resample(df, freq='1min'):
    """Resample a builder() DataFrame to the given frequency, returning the mean."""
    return df.resample(freq).mean()


def to_fahrenheit(df):
    """
    Return a copy of df with all *_Temperature_C columns converted to Fahrenheit.
    New columns are named *_Temperature_F.
    """
    df = df.copy()
    for col in df.columns:
        if col.endswith("_Temperature_C"):
            new_col = col.replace("_Temperature_C", "_Temperature_F")
            df[new_col] = df[col] * 9 / 5 + 32
            df = df.drop(columns=[col])
    return df
