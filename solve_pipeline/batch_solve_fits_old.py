#!/usr/bin/env python3
# batch_solve_fits.py
#
# Batch plate-solve all FITS in ./input using Astrometry.net and ASTAP,
# compute center RA/Dec from each solver's WCS, and export results.xlsx.
#
# Features:
# - Never overwrites your originals
# - Creates ./out/astrometry and ./out/astap
# - Timeouts, per-file logs, clear status fields
# - Works even if ASTAP overwrites in-place (we detect & copy)
#
# Dependencies: astropy, pandas, openpyxl

#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u

# ---------- Config ----------
INPUT_DIR = Path("input")
OUT_DIR = Path("out")
ASTROMETRY_OUT = OUT_DIR / "astrometry"
ASTAP_OUT = OUT_DIR / "astap"
LOG_DIR = OUT_DIR / "logs"

# Solve timeouts (seconds) — adjust for your images/catalogs
ASTROMETRY_TIMEOUT = 300
ASTAP_TIMEOUT = 180

# Command names to try
SOLVE_FIELD = "solve-field"
# ASTAP CLI names differ by platform/build; we try a few
ASTAP_CANDIDATES = ["astap", "astap-cli", "astap.exe"]

# Optional: extra arguments you typically use
ASTROMETRY_ARGS = [
    "--no-plots",
    "--overwrite",
    "--crpix-center",  # center reference pixel
    # "--guess-scale",         # uncomment if your index set is broad
    # "--downsample", "2",     # uncomment if needed for speed
]
# For ASTAP, we try to force an output file with -o when supported
# (Some builds solve in-place; we detect that case.)
ASTAP_ARGS_BASE = [
    "-solve",
    # You can add: "-r", "5"   # search radius deg (if you give an initial guess)
    # Or "-fov", "0.3"        # if you know your field of view in degrees
]

EXCEL_NAME = "results.xlsx"

# ---------- Helpers ----------


def which(cmd: str) -> Optional[str]:
    """Return full path if command exists on PATH."""
    from shutil import which as _which

    return _which(cmd)


def find_astap() -> Optional[str]:
    for c in ASTAP_CANDIDATES:
        p = which(c)
        if p:
            return p
    return None


def run_subprocess(
    args: List[str], timeout: int, log_file: Path
) -> Tuple[int, str, str]:
    """Run a subprocess, capture stdout/stderr, write a per-run log."""
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        stdout, stderr = proc.stdout, proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\nTIMEOUT after {timeout}s"
        rc = 124

    log_file.write_text(
        f"$ {' '.join(args)}\n\n=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n"
    )
    return rc, stdout, stderr


def read_center_radec_from_wcs(fits_path: Path) -> Tuple[float, float]:
    """Open FITS, compute WCS at image center (1-based FITS convention)."""
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr)
        nx = hdr.get("NAXIS1")
        ny = hdr.get("NAXIS2")
        if not nx or not ny:
            raise ValueError("Missing NAXIS1/NAXIS2.")
        # FITS pixel center: (NAXIS+1)/2
        cx = (nx + 1) / 2.0
        cy = (ny + 1) / 2.0
        world = wcs.pixel_to_world(cx, cy)
        ra_deg = float(world.ra.deg)
        dec_deg = float(world.dec.deg)
        return ra_deg, dec_deg


def ensure_dirs():
    for d in [OUT_DIR, ASTROMETRY_OUT, ASTAP_OUT, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_fits(input_dir: Path) -> List[Path]:
    patterns = ["**/*.fits", "**/*.fit", "**/*.fts"]
    fits_files = []
    for p in patterns:
        fits_files.extend(input_dir.glob(p))
    return sorted(fits_files)


def read_astap_hint_from_header(
    fits_path: Path,
) -> Optional[Tuple[float, float, float]]:
    """
    Read RA/Dec hint from FITS header.

    Returns:
        (ra_hours, dec_deg, spd_deg)
    where:
        ra_hours = RA in decimal hours
        dec_deg  = Dec in decimal degrees
        spd_deg  = south pole distance = dec_deg + 90
    """
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header

    # Prefer telescope/header pointing keywords
    ra_raw = hdr.get("OBJCTRA")
    dec_raw = hdr.get("OBJCTDEC")

    # Fallbacks
    if ra_raw is None:
        ra_raw = hdr.get("RA") or hdr.get("CRVAL1")
    if dec_raw is None:
        dec_raw = hdr.get("DEC") or hdr.get("CRVAL2")

    if ra_raw is None or dec_raw is None:
        return None

    # Parse RA
    if isinstance(ra_raw, str):
        # Example: '06 23 22.435'
        ra_hours = Angle(ra_raw, unit=u.hourangle).hour
    else:
        # Could already be numeric. If >24, assume degrees and convert to hours.
        ra_val = float(ra_raw)
        ra_hours = ra_val / 15.0 if ra_val > 24.0 else ra_val

    # Parse Dec
    if isinstance(dec_raw, str):
        # Example: '+05 53 21.66'
        dec_deg = Angle(dec_raw, unit=u.deg).deg
    else:
        dec_deg = float(dec_raw)

    spd_deg = dec_deg + 90.0
    return ra_hours, dec_deg, spd_deg


# ---------- Solvers ----------


# def solve_with_astrometry(src: Path) -> Tuple[Optional[Path], str]:
#     """
#     Use solve-field to produce a new FITS with WCS in ASTROMETRY_OUT.
#     Returns (solved_path or None, status_str).
#     """
#     solved = ASTROMETRY_OUT / f"{src.stem}_astrometry.fits"
#     logf = LOG_DIR / f"{src.stem}_astrometry.log"

#     args = [SOLVE_FIELD, *ASTROMETRY_ARGS, "--new-fits", str(solved), str(src)]

#     if not which(SOLVE_FIELD):
#         return None, "solve-field not found on PATH"

#     rc, stdout, stderr = run_subprocess(args, ASTROMETRY_TIMEOUT, logf)
#     if rc == 0 and solved.exists():
#         # Double-check WCS readable
#         try:
#             read_center_radec_from_wcs(solved)
#             return solved, "ok"
#         except Exception as e:
#             return None, f"Astrometry WCS read error: {e}"
#     # Fallback: sometimes solve-field writes <name>.new if --new-fits failed
#     alt = ASTROMETRY_OUT / f"{src.stem}.new"
#     if alt.exists():
#         try:
#             read_center_radec_from_wcs(alt)
#             alt.rename(solved)
#             return solved, "ok (renamed from .new)"
#         except Exception as e:
#             return None, f"Astrometry alt WCS read error: {e}"

#     # If it failed, give a concise reason
#     reason = "timeout" if rc == 124 else f"rc={rc}"
#     return None, f"Astrometry failed ({reason})"


def solve_with_astap(
    src: Path, astap_cmd: str
) -> Tuple[Optional[Tuple[float, float]], str]:
    """
    Run ASTAP in-place: `astap -f <src> -solve`.
    Then obtain center RA/Dec from either:
      1) the updated source FITS header, or
      2) a sidecar <src>.wcs file merged onto the original header.
    """
    from astropy.io import fits
    from astropy.io.fits import Header

    logf = LOG_DIR / f"{src.stem}_astap.log"

    def run_astap() -> Tuple[int, str, str]:
        args = [astap_cmd, "-f", str(src)]

        hint = read_astap_hint_from_header(src)
        if hint is not None:
            ra_hours, dec_deg, spd_deg = hint
            args += [
                "-ra",
                f"{ra_hours:.6f}",
                "-spd",
                f"{spd_deg:.6f}",
                "-r",
                "10",  # search radius in degrees; tune as needed
            ]

        args += ["-solve"]
        return run_subprocess(args, ASTAP_TIMEOUT, logf)

    def header_has_wcs(h: "fits.Header") -> bool:
        return all(k in h for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")) and (
            "CD1_1" in h or "CDELT1" in h or "PC1_1" in h
        )

    def center_radec_from_header(h_src: "fits.Header") -> Tuple[float, float]:
        nx = h_src.get("NAXIS1")
        ny = h_src.get("NAXIS2")
        if not nx or not ny:
            raise ValueError("Missing NAXIS1/NAXIS2 in source header.")
        cx = (nx + 1) / 2.0
        cy = (ny + 1) / 2.0
        w = WCS(h_src)
        world = w.pixel_to_world(cx, cy)
        return float(world.ra.deg), float(world.dec.deg)

    rc, _, _ = run_astap()

    try:
        with fits.open(src) as hdul:
            hdr = hdul[0].header
            if header_has_wcs(hdr):
                ra, dec = center_radec_from_header(hdr)
                return (ra, dec), "ok (in-place header)"
    except Exception:
        pass

    sidecar = src.with_suffix(".wcs")
    if sidecar.exists():
        try:
            with fits.open(src) as hdul:
                base = hdul[0].header.copy()
            h_side = Header.fromtextfile(str(sidecar))
            for k, v in h_side.items():
                base[k] = v
            ra, dec = center_radec_from_header(base)
            return (ra, dec), "ok (sidecar .wcs)"
        except Exception as e:
            return None, f"ASTAP sidecar WCS read error: {e}"

    if rc == 124:
        return None, "ASTAP failed (timeout)"
    return None, f"ASTAP failed (rc={rc}, no WCS found)"


# ---------- Main ----------


def main():
    ensure_dirs()

    # === Allow custom Excel filename ===
    if len(sys.argv) > 1:
        excel_name = sys.argv[1]
    else:
        excel_name = "results.xlsx"
    xls = OUT_DIR / excel_name

    input_dir = Path(__file__).parent / "input"
    files = find_fits(input_dir)
    if not files:
        print(f"No FITS found in {input_dir} (extensions: .fits .fit .fts).")
        sys.exit(1)

    astap_cmd = find_astap()
    if not astap_cmd:
        print(
            "Warning: ASTAP command not found. ASTAP results will be empty. "
            "Set ASTAP_CANDIDATES or add ASTAP to PATH.",
            file=sys.stderr,
        )

    rows: List[Dict[str, object]] = []

    for i, src in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {src.name}")

        # # ---- Astrometry.net solve ----
        # astrometry_path, astrometry_status = solve_with_astrometry(src)
        # astrometry_ra = astrometry_dec = None
        # if astrometry_path:
        #     try:
        #         astrometry_ra, astrometry_dec = read_center_radec_from_wcs(
        #             astrometry_path
        #         )
        #     except Exception as e:
        #         astrometry_status = f"Astrometry WCS read error: {e}"

        # ---- ASTAP solve ----
        astap_ra = astap_dec = None
        astap_status = "ASTAP not run (cmd not found)"
        if astap_cmd:
            astap_radec, astap_status = solve_with_astap(src, astap_cmd)
            if astap_radec:
                astap_ra, astap_dec = astap_radec

        # ---- Collect results ----
        rows.append(
            {
                "file": str(src.name),
                # "astrometry_ra_deg": astrometry_ra,
                # "astrometry_dec_deg": astrometry_dec,
                # "astrometry_status": astrometry_status,
                "astap_ra_deg": astap_ra,
                "astap_dec_deg": astap_dec,
                # "astap_status": astap_status,
            }
        )

    # ---- Write Excel ----
    df = pd.DataFrame(rows)
    df.to_excel(xls, index=False)
    print(f"\nDone. Wrote Excel: {xls.resolve()}")
    print(f"Per-file logs: {LOG_DIR.resolve()}")


if __name__ == "__main__":
    main()
