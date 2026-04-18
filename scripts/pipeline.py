import os
import cv2
import sys
import subprocess
from pathlib import Path
import warnings
import numpy as np

from astropy.wcs import WCS, FITSFixedWarning
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astride import Streak

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# call solve-field
def solve_with_astrometry(input_fits: str, ra: float, dec: float) -> str:
    """
    Runs solve-field on input_fits and returns path to the solved .new FITS.
    Raises RuntimeError on failure.
    """
    input_fits = str(input_fits)
    inpath = Path(input_fits)
    if not inpath.exists():
        raise FileNotFoundError(f"Input FITS not found: {input_fits}")

    out_new = inpath.with_suffix(".new")  # astrometry.net default
    # cmd = [
    #     "solve-field",
    #     input_fits,
    #     "--continue",
    # ]
    if ra == 0 and dec == 0:
        cmd = [
            "astap",
            "-f",
            input_fits,
            "-solve",
        ]
    else:
        cmd = [
            "astap",
            "-f",
            input_fits,
            "-solve",
            "-ra",
            f"{ra / 15:.3f}",
            "-spd",
            f"{dec + 90:.3f}",
        ]

    print(f"Running ASTAP:\n  {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Show solver log for debugging
    print(proc.stdout)

    if proc.returncode != 0:
        print("ASTAP solve failed.")
        cmd = [
            "solve-field",
            input_fits,
            "--overwrite"
        ]

        print(f"Running Astrometry.net:\n  {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            raise RuntimeError("solve-field failed. Can't solve image.")

    return str(inpath)

def pix2radec(w: WCS, x: float, y: float):
    """
    Robustly convert pixel -> sky (ICRS) in degrees.
    Works whether pixel_to_world returns SkyCoord or (lon, lat) Angles.
    Falls back to low-level all_pix2world if needed.
    """
    obj = w.pixel_to_world(x, y)

    # Case 1: already a SkyCoord
    if isinstance(obj, SkyCoord):
        c = obj.icrs  # transform to ICRS just in case
        return float(c.ra.deg), float(c.dec.deg)

    # Case 2: tuple/list of Angle-like (lon, lat)
    try:
        lon, lat = obj  # unpack
        ra = np.asanyarray(lon.to_value("deg")).astype(float)
        dec = np.asanyarray(lat.to_value("deg")).astype(float)
        return float(ra), float(dec)
    except Exception:
        pass

    # Case 3: fallback to low-level API (always degrees)
    ra, dec = w.all_pix2world(x, y, 0)
    return float(ra), float(dec)

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {Path(sys.argv[0]).name} <input.fits> <ra in deg> <dec in deg>")
        sys.exit(1)

    input_fits = sys.argv[1]
    ra = sys.argv[2]
    dec = sys.argv[3]

    # Solve to get WCS
    solved_fits = solve_with_astrometry(input_fits, float(ra), float(dec))

    # Load the solved image (with TAN-SIP WCS in header)
    hdul = fits.open(solved_fits)
    hdr  = hdul[0].header
    w    = WCS(hdr)

    # Get image dimensions and compute field center in sky coordinates
    # ny, nx = hdul[0].data.shape   # note: FITS images are indexed [y, x]
    # center_px = (nx / 2, ny / 2)  # (x_center, y_center)
    # center_world = w.pixel_to_world(center_px[0], center_px[1])
    # print(center_world)
    # print(f"Field center: RA {center_world.ra.deg:.6f} deg, Dec {center_world.dec.deg:.6f} deg")

    # # 2) Run ASTRiDE to detect streaks in pixel space
    # #    Tweak thresholds to your data; start with defaults, then adjust.
    # streak = Streak(solved_fits)
    # streak.detect()
    # streaks = streak.streaks  # list of dicts, each with polygon/center info

    # # print(streaks)

    # if not streaks:
    #     raise RuntimeError("No streaks detected. Try adjusting thresholds or denoising.")

    # # Heuristic: choose the longest streak by pixel length
    # det = streaks[0]  # you printed one detection
    # Apx = det['extreme_points'][0]  # [x, y]
    # Bpx = det['extreme_points'][1]  # [x, y]
    # Mpx = 0.5*(Apx + Bpx)

    # # 3) Convert pixel → sky (ICRS)
    # #    astropy expects (x, y) with origin=0; ASTRiDE also uses 0-based pixel indices.
    # raA, decA = w.pixel_to_world(Apx[0], Apx[1]).ra.deg, w.pixel_to_world(Apx[0], Apx[1]).dec.deg
    # raB, decB = w.pixel_to_world(Bpx[0], Bpx[1]).ra.deg, w.pixel_to_world(Bpx[0], Bpx[1]).dec.deg
    # raM, decM = w.pixel_to_world(Mpx[0], Mpx[1]).ra.deg, w.pixel_to_world(Mpx[0], Mpx[1]).dec.deg
#############################################################################
    # Center of the image
    ny, nx = hdul[0].data.shape
    cx, cy = nx/2, ny/2
    ra_c, dec_c = pix2radec(w, cx, cy)
    print(f"Field center: RA {ra_c:.6f} deg, Dec {dec_c:.6f} deg")

    # Pick the longest streak (your code picked index 0 regardless)
    def seg_len(p):
        # p is expected to be [[x1,y1],[x2,y2]]
        return float(np.hypot(p[0][0]-p[1][0], p[0][1]-p[1][1]))

    # 2) Run ASTRiDE to detect streaks in pixel space
    #    Tweak thresholds to your data; start with defaults, then adjust.
    streak = Streak(solved_fits)
    streak.detect()
    streaks = streak.streaks  # list of dicts, each with polygon/center info

    # print(streaks)

    if not streaks:
        raise RuntimeError("No streaks detected. Try adjusting thresholds or denoising.")

    for i in range(len(streaks)):
        det = streaks[i]  # you printed one detection
        Apx = det['extreme_points'][0]  # [x, y]
        Bpx = det['extreme_points'][1]  # [x, y]
        Mpx = 0.5*(Apx + Bpx)

        # Pixel -> sky (ICRS) for endpoints and midpoint
        raA, decA = pix2radec(w, Apx[0], Apx[1])
        raB, decB = pix2radec(w, Bpx[0], Bpx[1])
        raM, decM = pix2radec(w, Mpx[0], Mpx[1])
    ##############################################################################

        # 4) Angular length and PA on the sky
        A = SkyCoord(raA, decA, unit="deg", frame="icrs")
        B = SkyCoord(raB, decB, unit="deg", frame="icrs")
        M = SkyCoord(raM, decM, unit="deg", frame="icrs")

        ang_length = A.separation(B)                   # Quantity
        pa = A.position_angle(B).to("deg")             # east of north

        print()
        print(f"Streak {i}:")
        print(f"Endpoints:")
        print(f"  A: RA {A.ra.deg:.6f} deg, Dec {A.dec.deg:.6f} deg ({Apx[0]:.3f}, {Apx[1]:.3f})")
        print(f"  B: RA {B.ra.deg:.6f} deg, Dec {B.dec.deg:.6f} deg ({Bpx[0]:.3f}, {Bpx[1]:.3f})")
        print(f"Midpoint:")
        print(f"  M: RA {M.ra.deg:.6f} deg, Dec {M.dec.deg:.6f} deg ({Mpx[0]:.3f}, {Mpx[1]:.3f})")
        print(f"On-sky length: {ang_length.to('arcmin').value:.3f} arcmin")
        print(f"Position angle: {pa.value:.2f} deg  (E of N)")


if __name__ == "__main__":
    main()
