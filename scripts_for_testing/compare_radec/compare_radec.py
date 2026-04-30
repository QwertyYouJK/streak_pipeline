# Python script to compare RA/Dec from FITS metadata with propagated RA/Dec from user-provided TLEs.
# Usage: python compare_radec.py

import csv
from pathlib import Path

from astropy.io import fits
from astropy.time import Time
from skyfield.api import EarthSatellite, load, wgs84

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_CSV = Path(__file__).parent / "propagated_radec.csv"
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
SATELLITE_NAME = "MANUAL-TLE"

ts = load.timescale()


def find_fits_files():
    return sorted(
        path for path in INPUT_DIR.rglob("*") if path.suffix.lower() in FITS_SUFFIXES
    )


def read_fits_metadata(fits_path):
    with fits.open(fits_path, ignore_missing_end=True) as hdul:
        header = hdul[0].header

    lat = float(header["GPS_LAT"])
    lon = float(header["GPS_LONG"])
    alt = float(header["GPS_ALT"])
    obstime = Time(str(header["DATE-AVG"]).rstrip("Z"), format="isot", scale="utc")
    return lat, lon, alt, obstime


def prompt_tle():
    while True:
        line1 = input("TLE line 1 (`skip` to skip, `quit` to stop): ").strip()
        if line1.lower() in {"skip", "quit"}:
            return line1.lower(), None
        if line1.startswith("1 "):
            break
        print("Line 1 should start with `1 `.")

    while True:
        line2 = input("TLE line 2 (`skip` to skip, `quit` to stop): ").strip()
        if line2.lower() in {"skip", "quit"}:
            return line2.lower(), None
        if line2.startswith("2 "):
            return line1, line2
        print("Line 2 should start with `2 `.")


fits_files = find_fits_files()
if not fits_files:
    raise SystemExit(f"No FITS files found in {INPUT_DIR}")

rows_written = 0

with OUTPUT_CSV.open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "file",
            "mid_exposure_utc",
            "tle_epoch_utc",
            "propagated_ra_deg",
            "propagated_dec_deg",
        ]
    )

    for index, fits_path in enumerate(fits_files, start=1):
        lat, lon, alt, obstime = read_fits_metadata(fits_path)

        print()
        print(f"[{index}/{len(fits_files)}] {fits_path.relative_to(INPUT_DIR.parent)}")
        print(f"Mid-exposure UTC: {obstime.iso} UTC")

        line1, line2 = prompt_tle()
        if line1 == "quit":
            print("Stopping.")
            break
        if line1 == "skip":
            print("Skipped.")
            continue

        satellite = EarthSatellite(line1, line2, SATELLITE_NAME, ts)
        observer = wgs84.latlon(lat, lon, elevation_m=alt)
        ra, dec, _ = (satellite - observer).at(ts.from_astropy(obstime)).radec()
        tle_epoch = satellite.epoch.utc_strftime("%Y-%m-%d %H:%M:%S UTC")

        writer.writerow(
            [
                fits_path.relative_to(INPUT_DIR.parent),
                obstime.iso,
                tle_epoch,
                f"{ra.degrees:.8f}",
                f"{dec.degrees:.8f}",
            ]
        )
        rows_written += 1

        print(f"TLE epoch: {tle_epoch}")
        print(f"Propagated RA: {ra.degrees:.8f} deg")
        print(f"Propagated Dec: {dec.degrees:.8f} deg")

print()
print(f"Wrote {rows_written} row(s) to {OUTPUT_CSV}")
