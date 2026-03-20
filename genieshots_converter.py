# Functions to help convert pngs from genie into fits for the dot_movie-Copy3.ipynb toolchain
# Should be as easy as convert_folder(source_dir), but there are more complex options.

from pathlib import Path
from datetime import datetime
import os

import numpy as np
from PIL import Image
from astropy.io import fits


def get_png_files(source_dir):
    """
    Return a sorted list of PNG files in the source directory.

    Currently sorted by filename; use st_mtime to sort by modify time instead.
    """
    png_files = sorted(
        Path(source_dir).glob("*.png"),
        key=lambda p: p.name  # or: key=lambda p: p.stat().st_mtime
    )
    return png_files


def get_file_mtime(path_obj):
    """
    Return the file modification time as a datetime object.
    """
    stat = path_obj.stat()
    return datetime.fromtimestamp(stat.st_mtime)


def format_timestamp(dt):
    """
    Format datetime as 'yy-mm-dd HH-MM-SS'
    Example: 2026-03-02 17:04:30 -> '26-03-02 17-04-30'
    """
    return dt.strftime("%y-%m-%d %H-%M-%S")


def png_to_fits(png_path, fits_path):
    """
    Read a PNG image and write it as a FITS file.

    - Converts to grayscale if needed.
    - Uses float32 data in the FITS file.
    """
    with Image.open(png_path) as img:
        if img.mode not in ("I", "F", "L"):
            img = img.convert("L")

        data = np.array(img, dtype=np.float32)

    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])

    fits_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: overwrite behavior is controlled at the call site now
    hdul.writeto(fits_path, overwrite=True)


def build_output_name(index, timestamp_str):
    """
    Build output filename like: image0001 26-03-02 17-04-30.fits
    """
    return f"image{index:04d} {timestamp_str}.fits"

def infer_dest_dir_from_source(source_dir: Path) -> Path:
    """
    Given a source_dir like:
        .../20260302/genieshots
    return:
        .../20260302_data/genieshots/genieshots_fits
    """
    source_dir = Path(source_dir)
    base_date_dir = source_dir.parent          # .../20260302
    run_name = source_dir.name                 # "genieshots"

    # Create sibling of base_date_dir with "_data" suffix
    data_dir = base_date_dir.with_name(base_date_dir.name + "_data")

    # Full destination: 20260302_data/genieshots/genieshots_fits
    dest_dir = data_dir / run_name / f"{run_name}_fits"
    return dest_dir

def convert_folder(
    source_dir,
    dest_dir=None,
    log_path=None,
    dry_run=False,
    skip_existing=True,
):
    """
    Convert all PNGs in source_dir to FITS files in dest_dir,
    with names based on index and file modification time, and
    write a log of filename and creation date.

    - If dest_dir is None, it will be inferred:
        .../YYYYMMDD/runname  ->  .../YYYYMMDD_data/runname/runname_fits
    - If log_path is None, it will be:
        dest_dir / f"{run_name}.txt"

    Parameters
    ----------
    source_dir : str or Path
        Directory containing PNG files.
    dest_dir : str or Path
        Destination directory for FITS files.
    log_path : str or Path or None
        Path to the log file. If None, defaults to dest_dir / 'creation_dates.txt'.
    dry_run : bool
        If True, only print planned operations without writing files.
    skip_existing : bool
        If True, do not overwrite existing FITS files; just skip them.
    """
    source_dir = Path(source_dir)
    # --- Infer dest_dir if not provided ---
    if dest_dir is None:
        dest_dir = infer_dest_dir_from_source(source_dir)
    else:
        dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        # Use the parent directory name of dest_dir as the run name
        run_name = source_dir.name  # "genieshots" in your example
        log_path = dest_dir.parent / f"{run_name}.txt"
    else:
        log_path = Path(log_path)

    png_files = get_png_files(source_dir)

    if not png_files:
        print(f"No PNG files found in {source_dir}")
        return

    print(f"Found {len(png_files)} PNG files in {source_dir}")
    print(f"FITS will be written to: {dest_dir}")
    print(f"Log will be written to:  {log_path}")

    # Open log file and write header (overwrite each run like your BMP pipeline)
    with open(log_path, "w") as log_f:
        log_f.write("Filename\tCreation Date\n")
        log_f.write("-----------------------------------\n")

        for idx, png_path in enumerate(png_files, start=1):
            # Get modification time and format it
            mtime = get_file_mtime(png_path)
            ts_str = format_timestamp(mtime)

            # Build output filename and path
            out_name = build_output_name(idx, ts_str)
            fits_path = dest_dir / out_name

            # Logging line (matches your BMP log style)
            # Using creation_date as the datetime object, not just string.
            creation_date = mtime
            log_f.write(f"{fits_path.name}\t{creation_date}\n")

            if fits_path.exists() and skip_existing:
                print(f"Skipping existing file: {fits_path.name}")
                continue

            if dry_run:
                print(f"[DRY RUN] {png_path.name} -> {fits_path.name}")
            else:
                print(f"Converting {png_path.name} -> {fits_path.name}")
                # Will overwrite if exists (we already checked skip_existing above)
                png_to_fits(png_path, fits_path)

    print(f"Creation dates saved to {log_path}")

if __name__ == "__main__":
    # Adjust these paths as needed; raw strings avoid backslash-escape issues
    source = r"E:\Reverse Telescope Test Data\20260302\genieshots"
    # dest = r"E:\Reverse Telescope Test Data\20260302_data\genieshots\genieshots_fits"

    # First run as dry-run to check naming without writing files:
    # convert_folder(source, dest, dry_run=True)

    # When you're happy, run the actual conversion:
    convert_folder(source, dry_run=False)