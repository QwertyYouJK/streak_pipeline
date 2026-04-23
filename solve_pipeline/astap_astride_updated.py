#!/usr/bin/env python3
"""
astap_astride_updated.py

For each FITS file in ./input:
- run ASTAP plate solve
- get WCS from ASTAP's clean .ini solution
- run the updated ASTRiDE streak detector on a blurred temporary FITS
- filter short detections using the tuned thresholds from
  astride_evaluation/evaluate_astride_combined.py
- merge linked detections using connectivity
- reject probable border artifacts caused by padded-image detection
- compute one center per kept merged streak group
- convert image center and streak center to RA/Dec
- write CSV report

Usage:
    python astap_astride_updated.py OUTPUT_CSV

Output:
    out/OUTPUT_CSV, if OUTPUT_CSV is a filename

Notes:
- The ASTAP/WCS flow is intentionally kept the same as astap_astride.py.
- The streak-detection flow mirrors
  astride_evaluation/evaluate_astride_combined.py.
- The CSV includes per-file timing columns for the major pipeline stages.
- If no ASTAP .ini WCS is available, RA/Dec fields are left blank.
- If no streak groups remain after filtering, one CSV row is still written
  with blank streak coordinates for that file.
- Pixel coordinates written to CSV are 0-based Python-style coordinates.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from astride import Streak


# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("input")
OUT_DIR = Path("out")

ASTAP_TIMEOUT = 1000
ASTAP_CANDIDATES = ["astap", "astap-cli", "astap.exe"]

ASTRIDE_PARAMS = dict(
    remove_bkg="map",
    bkg_box_size=50,
    contour_threshold=2,
    min_points=5,
    shape_cut=0.2,
    area_cut=10,
    radius_dev_cut=0.5,
    connectivity_angle=10.0,
    output_path=None,
)

BLUR_SIGMA = 0.8
MIN_LENGTH = 50.0
ZERO_ANGLE_TOL_DEG = 0.5
BORDER_CENTER_MARGIN_PX = 1.0

CSV_FIELDNAMES = [
    "file",
    "astap_status",
    "image_center_ra_deg",
    "image_center_dec_deg",
    "streak_center_x_px",
    "streak_center_y_px",
    "streak_center_ra_deg",
    "streak_center_dec_deg",
    "astap_solve_seconds",
    "center_radec_seconds",
    "astride_with_blur_seconds",
    "group_and_row_seconds",
]


# ----------------------------
# File / command helpers
# ----------------------------
def find_input_files(input_dir: Path):
    files = []
    for pattern in ["**/*.fits", "**/*.fit", "**/*.fts"]:
        files.extend(input_dir.glob(pattern))
    return sorted(files)


def find_astap():
    from shutil import which

    for name in ASTAP_CANDIDATES:
        p = which(name)
        if p:
            return p
    return None


def run_subprocess(args, timeout):
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + f"\nTIMEOUT after {timeout}s"


# ----------------------------
# ASTAP / WCS
# ----------------------------
ASTAP_REQUIRED_KEYS = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2")
ASTAP_CD_KEYS = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
ASTAP_SCALE_ROTATION_KEYS = ("CDELT1", "CDELT2", "CROTA1", "CROTA2")


def run_astap(src: Path, astap_cmd: str):
    args = [astap_cmd, "-f", str(src), "-r", "180"]
    return run_subprocess(args, ASTAP_TIMEOUT)


def read_astap_ini(src: Path):
    ini_path = src.with_suffix(".ini")
    values = {}

    if not ini_path.exists():
        return values

    for raw_line in ini_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()

    return values


def astap_ini_is_solved(values):
    return values.get("PLTSOLVD", "").upper() in {"T", "TRUE", "1", "Y", "YES"}


def has_keys(values, keys):
    return all(key in values for key in keys)


def astap_ini_has_wcs(values):
    return (
        astap_ini_is_solved(values)
        and has_keys(values, ASTAP_REQUIRED_KEYS)
        and (
            has_keys(values, ASTAP_CD_KEYS)
            or has_keys(values, ASTAP_SCALE_ROTATION_KEYS)
        )
    )


def astap_float(values, key):
    return float(values[key])


def astap_ini_to_wcs(src: Path):
    # Build an Astropy WCS from ASTAP's .ini values only.
    values = read_astap_ini(src)
    if not astap_ini_has_wcs(values):
        return None

    hdr = fits.Header()
    hdr["WCSAXES"] = 2
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["RADESYS"] = "ICRS"

    for key in ASTAP_REQUIRED_KEYS:
        hdr[key] = astap_float(values, key)

    if has_keys(values, ASTAP_CD_KEYS):
        for key in ASTAP_CD_KEYS:
            hdr[key] = astap_float(values, key)
    else:
        for key in ASTAP_SCALE_ROTATION_KEYS:
            hdr[key] = astap_float(values, key)

    return WCS(hdr)


def solve_with_astap(src: Path, astap_cmd: str):
    _rc, _stdout, _stderr = run_astap(src, astap_cmd)

    wcs = astap_ini_to_wcs(src)
    if wcs is not None:
        return wcs, "ok"

    return None, "failed"


def image_center_radec(src: Path, wcs: WCS):
    with fits.open(src) as hdul:
        hdr = hdul[0].header
        nx = hdr["NAXIS1"]
        ny = hdr["NAXIS2"]

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    ra_deg, dec_deg = wcs.pixel_to_world_values(cx, cy)
    return cx, cy, float(ra_deg), float(dec_deg)


def pixel_to_radec(wcs: WCS, x: float, y: float):
    ra_deg, dec_deg = wcs.pixel_to_world_values(x, y)
    return float(ra_deg), float(dec_deg)


# ----------------------------
# ASTRiDE
# ----------------------------
def write_blurred_temp_fits(src: Path, sigma: float):
    data, header = fits.getdata(src, header=True, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    fits.writeto(tmp_path, img, header=header, overwrite=True)
    return tmp_path


def image_shape(src: Path):
    data = fits.getdata(src, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))
    if img.ndim < 2:
        raise ValueError(f"Expected at least a 2D FITS image, got shape {img.shape}")
    return img.shape[-2:]


def run_astride(src_for_astride: Path):
    streak = Streak(str(src_for_astride), **ASTRIDE_PARAMS)
    streak.detect()

    streak.streaks = [s for s in streak.streaks if s["length"] > MIN_LENGTH]
    return streak


def build_groups(streak):
    # Build connected groups from ASTRiDE connectivity
    idx_to_streak = {s["index"]: s for s in streak.streaks}
    adj = {idx: set() for idx in idx_to_streak}

    for s in streak.streaks:
        i = s["index"]
        j = s["connectivity"]
        if j != -1 and j in idx_to_streak:
            adj[i].add(j)
            adj[j].add(i)

    groups = []
    seen = set()

    for start in adj:
        if start in seen:
            continue

        stack = [start]
        group = []
        seen.add(start)

        while stack:
            u = stack.pop()
            group.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        groups.append(sorted(group))

    return groups, idx_to_streak


def merged_geometry(group, idx_to_streak):
    # Merge a connected group into one center/endpoints
    pts = []

    for idx in group:
        s = idx_to_streak[idx]

        x = np.asarray(s.get("x", []), dtype=float)
        y = np.asarray(s.get("y", []), dtype=float)

        if len(x) > 0 and len(y) > 0:
            pts.append(np.column_stack([x, y]))
        else:
            ep1, ep2 = s["extreme_points"]
            pts.append(np.array([[ep1[0], ep1[1]], [ep2[0], ep2[1]]], dtype=float))

    P = np.vstack(pts)
    mean_pt = P.mean(axis=0)
    Q = P - mean_pt
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    direction = vh[0] / np.linalg.norm(vh[0])

    t = Q @ direction
    ep1 = mean_pt + t.min() * direction
    ep2 = mean_pt + t.max() * direction
    center = 0.5 * (ep1 + ep2)
    length = float(np.linalg.norm(ep2 - ep1))

    return center, ep1, ep2, length


def group_records(groups, idx_to_streak):
    records = []

    for group_number, group in enumerate(groups, start=1):
        center, ep1, ep2, merged_length = merged_geometry(group, idx_to_streak)
        raw_lengths = [idx_to_streak[idx]["length"] for idx in group]
        slope_angles = [idx_to_streak[idx]["slope_angle"] for idx in group]

        records.append(
            {
                "group_number": group_number,
                "component_indices": ";".join(str(idx) for idx in group),
                "num_components": len(group),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "endpoint1_x": float(ep1[0]),
                "endpoint1_y": float(ep1[1]),
                "endpoint2_x": float(ep2[0]),
                "endpoint2_y": float(ep2[1]),
                "merged_length": merged_length,
                "component_length_sum": float(np.sum(raw_lengths)),
                "mean_slope_angle": float(np.mean(slope_angles)),
            }
        )

    return records


def is_probable_border_artifact(rec, img_shape):
    height, width = img_shape
    max_x = width - 1
    max_y = height - 1

    near_zero_angle = abs(rec["mean_slope_angle"]) <= ZERO_ANGLE_TOL_DEG
    near_border = (
        rec["center_x"] <= BORDER_CENTER_MARGIN_PX
        or rec["center_x"] >= max_x - BORDER_CENTER_MARGIN_PX
        or rec["center_y"] <= BORDER_CENTER_MARGIN_PX
        or rec["center_y"] >= max_y - BORDER_CENTER_MARGIN_PX
    )

    return near_zero_angle and near_border


def filter_border_artifacts(groups, idx_to_streak, img_shape):
    grouped = group_records(groups, idx_to_streak)
    kept_groups = []
    rejected_records = []

    for group, rec in zip(groups, grouped):
        if is_probable_border_artifact(rec, img_shape):
            rejected_records.append(rec)
        else:
            kept_groups.append(group)

    return kept_groups, group_records(kept_groups, idx_to_streak), rejected_records


def show_image_only(fpath: Path):
    data = fits.getdata(fpath, ignore_missing_end=True)
    img = np.squeeze(np.array(data, dtype=float))

    vmin, vmax = np.percentile(img, [5, 99.7])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{fpath.name} - no kept detections")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASTAP and the updated ASTRiDE pipeline on FITS files."
    )
    parser.add_argument(
        "output_csv",
        help="Name or path of the output CSV file, for example report.csv.",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    if output_csv.suffix.lower() != ".csv":
        parser.error("output_csv must end with .csv")
    if not output_csv.parent or output_csv.parent == Path("."):
        output_csv = OUT_DIR / output_csv

    return output_csv


def main():
    program_start = perf_counter()
    report_csv = parse_args()

    report_csv.parent.mkdir(parents=True, exist_ok=True)

    astap_cmd = find_astap()
    if not astap_cmd:
        print("ASTAP not found on PATH.", file=sys.stderr)
        sys.exit(1)

    files = find_input_files(INPUT_DIR)
    if not files:
        print(f"No FITS files found in {INPUT_DIR.resolve()}")
        sys.exit(1)

    rows = []

    for i, src in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {src.name}")

        astap_start = perf_counter()
        wcs, astap_status = solve_with_astap(src, astap_cmd)
        astap_solve_seconds = perf_counter() - astap_start

        center_x_px = center_y_px = None
        center_ra_deg = center_dec_deg = None
        center_radec_seconds = None
        if wcs is not None:
            center_start = perf_counter()
            center_x_px, center_y_px, center_ra_deg, center_dec_deg = (
                image_center_radec(src, wcs)
            )
            center_radec_seconds = perf_counter() - center_start

        print(f" Solved image center (RA/Dec deg): {center_ra_deg}, {center_dec_deg}")

        astride_start = perf_counter()
        show_image_only(src)
        temp_fits = write_blurred_temp_fits(src, BLUR_SIGMA)
        show_image_only(temp_fits)
        try:
            streak = run_astride(temp_fits)
        finally:
            temp_fits.unlink(missing_ok=True)
        astride_with_blur_seconds = perf_counter() - astride_start

        row_start = len(rows)
        group_and_row_start = perf_counter()
        if streak.streaks:
            groups, idx_to_streak = build_groups(streak)
            raw_group_count = len(groups)
            groups, grouped_records, rejected_records = filter_border_artifacts(
                groups,
                idx_to_streak,
                image_shape(src),
            )

            # print(
            #     "  raw_components="
            #     f"{len(streak.streaks)} raw_groups={raw_group_count} "
            #     f"kept_groups={len(grouped_records)} rejected_border_groups="
            #     f"{len(rejected_records)}"
            # )
        else:
            grouped_records = []

        if not grouped_records:
            rows.append(
                {
                    "file": src.name,
                    "astap_status": astap_status,
                    "image_center_ra_deg": center_ra_deg,
                    "image_center_dec_deg": center_dec_deg,
                    "streak_center_x_px": "",
                    "streak_center_y_px": "",
                    "streak_center_ra_deg": "",
                    "streak_center_dec_deg": "",
                }
            )
        else:
            for rec in grouped_records:
                sx = rec["center_x"]
                sy = rec["center_y"]

                if wcs is not None:
                    sra, sdec = pixel_to_radec(wcs, sx, sy)
                else:
                    sra = sdec = None

                rows.append(
                    {
                        "file": src.name,
                        "astap_status": astap_status,
                        "image_center_ra_deg": center_ra_deg,
                        "image_center_dec_deg": center_dec_deg,
                        "streak_center_x_px": sx,
                        "streak_center_y_px": sy,
                        "streak_center_ra_deg": sra,
                        "streak_center_dec_deg": sdec,
                    }
                )

        group_and_row_seconds = perf_counter() - group_and_row_start
        # for row in rows[row_start:]:
        #     row["astap_solve_seconds"] = astap_solve_seconds
        #     row["center_radec_seconds"] = center_radec_seconds
        #     row["astride_with_blur_seconds"] = astride_with_blur_seconds
        #     row["group_and_row_seconds"] = group_and_row_seconds

        # print(
        #     " Timing (s): "
        #     f"ASTAP={astap_solve_seconds:.6f}, "
        #     f"center={center_radec_seconds or 0.0:.6f}, "
        #     f"ASTRiDE+blur={astride_with_blur_seconds:.6f}, "
        #     f"group+row={group_and_row_seconds:.6f}"
        # )

    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    program_total_seconds = perf_counter() - program_start
    print(f"\nDone. Wrote CSV: {report_csv.resolve()}")
    print(f"Total runtime: {program_total_seconds:.6f} s")


if __name__ == "__main__":
    main()
