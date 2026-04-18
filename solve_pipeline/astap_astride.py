#!/usr/bin/env python3
"""
astap_astride.py

For each FITS file in ./input:
- run ASTAP plate solve
- get WCS from solved FITS header or sidecar .wcs
- run ASTRiDE on a blurred temporary FITS
- filter short detections
- merge linked detections using connectivity
- compute one center per merged streak
- convert image center and streak center to RA/Dec
- write CSV report

Output:
    out/report.csv

Notes:
- If ASTAP fails, RA/Dec fields are left blank.
- If ASTRiDE finds two separate streaks, two CSV rows are written
  with the same file name and same image-center RA/Dec.
- Pixel coordinates written to CSV are 0-based Python-style coordinates.
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u
from scipy.ndimage import gaussian_filter
from astride import Streak


# ----------------------------
# Config
# ----------------------------
INPUT_DIR = Path("input")
OUT_DIR = Path("out")
REPORT_CSV = OUT_DIR / "report.csv"

ASTAP_TIMEOUT = 180
ASTAP_CANDIDATES = ["astap", "astap-cli", "astap.exe"]

ASTRIDE_PARAMS = dict(
    remove_bkg="map",
    bkg_box_size=50,
    contour_threshold=1.7,
    min_points=10,
    shape_cut=0.2,
    area_cut=10,
    radius_dev_cut=0.5,
    connectivity_angle=5.0,
    output_path=None,
)

BLUR_SIGMA = 0.8
MIN_LENGTH = 30.0


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
def read_astap_hint_from_header(fits_path: Path):
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header

    ra_raw = hdr.get("OBJCTRA")
    dec_raw = hdr.get("OBJCTDEC")

    if ra_raw is None:
        ra_raw = hdr.get("RA") or hdr.get("CRVAL1")
    if dec_raw is None:
        dec_raw = hdr.get("DEC") or hdr.get("CRVAL2")

    if ra_raw is None or dec_raw is None:
        return None

    if isinstance(ra_raw, str):
        ra_hours = Angle(ra_raw, unit=u.hourangle).hour
    else:
        ra_val = float(ra_raw)
        ra_hours = ra_val / 15.0 if ra_val > 24.0 else ra_val

    if isinstance(dec_raw, str):
        dec_deg = Angle(dec_raw, unit=u.deg).deg
    else:
        dec_deg = float(dec_raw)

    spd_deg = dec_deg + 90.0
    return ra_hours, dec_deg, spd_deg


def header_has_wcs(hdr):
    return all(k in hdr for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")) and (
        "CD1_1" in hdr or "CDELT1" in hdr or "PC1_1" in hdr
    )


def get_wcs_from_source_or_sidecar(src: Path):
    # try in-place FITS header first
    with fits.open(src) as hdul:
        hdr = hdul[0].header.copy()
    if header_has_wcs(hdr):
        return WCS(hdr)

    # fallback: sidecar .wcs
    sidecar = src.with_suffix(".wcs")
    if sidecar.exists():
        side = Header.fromtextfile(str(sidecar))
        for k, v in side.items():
            hdr[k] = v
        if header_has_wcs(hdr):
            return WCS(hdr)

    return None


def solve_with_astap(src: Path, astap_cmd: str):
    args = [astap_cmd, "-f", str(src)]

    hint = read_astap_hint_from_header(src)
    if hint is not None:
        ra_hours, _, spd_deg = hint
        args += [
            "-ra",
            f"{ra_hours:.6f}",
            "-spd",
            f"{spd_deg:.6f}",
            "-r",
            "180",
        ]

    rc, _, _ = run_subprocess(args, ASTAP_TIMEOUT)

    wcs = get_wcs_from_source_or_sidecar(src)
    if wcs is not None:
        return wcs, "ok"
    if rc == 124:
        return None, "timeout"
    return None, f"failed rc={rc}"


def image_center_radec(src: Path, wcs: WCS):
    with fits.open(src) as hdul:
        hdr = hdul[0].header
        nx = hdr["NAXIS1"]
        ny = hdr["NAXIS2"]

    # 0-based pixel center for Astropy high-level WCS
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


def run_astride(src_for_astride: Path):
    streak = Streak(str(src_for_astride), **ASTRIDE_PARAMS)
    streak.detect()

    # Keep only long enough detections
    streak.streaks = [s for s in streak.streaks if s["length"] > MIN_LENGTH]
    return streak


def build_groups(streak):
    """
    Build connected groups from ASTRiDE connectivity.
    Treat connectivity as undirected.
    """
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


def merged_group_center(group, idx_to_streak):
    """
    Compute one center for a connected streak group using PCA
    on all contour points / extreme points.
    """
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

    return float(center[0]), float(center[1]), length


# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

        # ASTAP solve
        wcs, astap_status = solve_with_astap(src, astap_cmd)

        center_x_px = center_y_px = None
        center_ra_deg = center_dec_deg = None
        if wcs is not None:
            center_x_px, center_y_px, center_ra_deg, center_dec_deg = (
                image_center_radec(src, wcs)
            )

        # ASTRiDE detect on blurred temp copy
        temp_fits = write_blurred_temp_fits(src, BLUR_SIGMA)
        try:
            streak = run_astride(temp_fits)
        finally:
            temp_fits.unlink(missing_ok=True)

        if not streak.streaks:
            rows.append(
                {
                    "file": src.name,
                    # "astap_status": astap_status,
                    # "image_center_x_px": center_x_px,
                    # "image_center_y_px": center_y_px,
                    # "image_center_ra_deg": center_ra_deg,
                    # "image_center_dec_deg": center_dec_deg,
                    # "streak_group": "",
                    # "n_segments_in_group": 0,
                    # "segment_indices": "",
                    "streak_center_x_px": "",
                    "streak_center_y_px": "",
                    "streak_center_ra_deg": "",
                    "streak_center_dec_deg": "",
                    "merged_streak_length_px": "",
                }
            )
            continue

        groups, idx_to_streak = build_groups(streak)

        for gi, group in enumerate(groups, start=1):
            sx, sy, merged_len = merged_group_center(group, idx_to_streak)

            if wcs is not None:
                sra, sdec = pixel_to_radec(wcs, sx, sy)
            else:
                sra = sdec = None

            rows.append(
                {
                    "file": src.name,
                    # "astap_status": astap_status,
                    # "image_center_x_px": center_x_px,
                    # "image_center_y_px": center_y_px,
                    # "image_center_ra_deg": center_ra_deg,
                    # "image_center_dec_deg": center_dec_deg,
                    # "streak_group": gi,
                    # "n_segments_in_group": len(group),
                    # "segment_indices": ",".join(str(x) for x in group),
                    "streak_center_x_px": sx,
                    "streak_center_y_px": sy,
                    "streak_center_ra_deg": sra,
                    "streak_center_dec_deg": sdec,
                    "merged_streak_length_px": merged_len,
                }
            )

    with REPORT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                # "astap_status",
                # "image_center_x_px",
                # "image_center_y_px",
                # "image_center_ra_deg",
                # "image_center_dec_deg",
                # "streak_group",
                # "n_segments_in_group",
                # "segment_indices",
                "streak_center_x_px",
                "streak_center_y_px",
                "streak_center_ra_deg",
                "streak_center_dec_deg",
                "merged_streak_length_px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote CSV: {REPORT_CSV.resolve()}")


if __name__ == "__main__":
    main()
