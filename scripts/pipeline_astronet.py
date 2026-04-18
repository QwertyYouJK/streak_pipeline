import os
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
def solve_with_astrometry(input_fits: str) -> str:
    """
    Runs solve-field on input_fits and returns path to the solved .new FITS.
    Raises RuntimeError on failure.
    """
    input_fits = str(input_fits)
    inpath = Path(input_fits)
    if not inpath.exists():
        raise FileNotFoundError(f"Input FITS not found: {input_fits}")

    out_new = inpath.with_suffix(".new")  # astrometry.net default
    cmd = [
        "solve-field",
        input_fits,
        "--continue",
    ]

    print(f"[solve-field] Running:\n  {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Show solver log for debugging
    print(proc.stdout)

    if proc.returncode != 0 or not out_new.exists():
        raise RuntimeError("solve-field failed or did not produce a .new file.")

    return str(out_new)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <input.fits>")
        sys.exit(1)

    input_fits = sys.argv[1]

    # Solve to get WCS
    solved_fits = solve_with_astrometry(input_fits)

    # Load the solved image (with TAN-SIP WCS in header)
    hdul = fits.open(solved_fits)
    hdr  = hdul[0].header
    w    = WCS(hdr)

    # Get image dimensions and compute field center in sky coordinates
    ny, nx = hdul[0].data.shape   # note: FITS images are indexed [y, x]
    center_px = (nx / 2, ny / 2)  # (x_center, y_center)
    center_world = w.pixel_to_world(center_px[0], center_px[1])

    print(f"Field center: RA {center_world.ra.deg:.6f} deg, Dec {center_world.dec.deg:.6f} deg")


    # 2) Run ASTRiDE to detect streaks in pixel space
    #    Tweak thresholds to your data; start with defaults, then adjust.
    streak = Streak(solved_fits)
    streak.detect()
    streaks = streak.streaks  # list of dicts, each with polygon/center info

    # print(streaks)

    if not streaks:
        raise RuntimeError("No streaks detected. Try adjusting thresholds or denoising.")

    # Heuristic: choose the longest streak by pixel length
    det = streaks[0]  # you printed one detection
    Apx = det['extreme_points'][0]  # [x, y]
    Bpx = det['extreme_points'][1]  # [x, y]
    Mpx = 0.5*(Apx + Bpx)

    # 3) Convert pixel → sky (ICRS)
    #    astropy expects (x, y) with origin=0; ASTRiDE also uses 0-based pixel indices.
    raA, decA = w.pixel_to_world(Apx[0], Apx[1]).ra.deg, w.pixel_to_world(Apx[0], Apx[1]).dec.deg
    raB, decB = w.pixel_to_world(Bpx[0], Bpx[1]).ra.deg, w.pixel_to_world(Bpx[0], Bpx[1]).dec.deg
    raM, decM = w.pixel_to_world(Mpx[0], Mpx[1]).ra.deg, w.pixel_to_world(Mpx[0], Mpx[1]).dec.deg

    # 4) Angular length and PA on the sky
    A = SkyCoord(raA, decA, unit="deg", frame="icrs")
    B = SkyCoord(raB, decB, unit="deg", frame="icrs")
    M = SkyCoord(raM, decM, unit="deg", frame="icrs")

    ang_length = A.separation(B)                   # Quantity
    pa = A.position_angle(B).to("deg")             # east of north

    print(f"Endpoints:")
    print(f"  A: RA {A.ra.deg:.6f} deg, Dec {A.dec.deg:.6f} deg")
    print(f"  B: RA {B.ra.deg:.6f} deg, Dec {B.dec.deg:.6f} deg")
    print(f"Midpoint:")
    print(f"  M: RA {M.ra.deg:.6f} deg, Dec {M.dec.deg:.6f} deg")
    print(f"On-sky length: {ang_length.to('arcmin').value:.3f} arcmin")
    print(f"Position angle: {pa.value:.2f} deg  (E of N)")


if __name__ == "__main__":
    main()