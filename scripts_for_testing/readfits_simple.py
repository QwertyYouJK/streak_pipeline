# Python file for reading FITS files and displaying the image data using Matplotlib
# Usage: python readfits_simple.py <input.fits>

from astropy.io import fits
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <input.fits>")
        sys.exit(1)

    input_fits = sys.argv[1]

    # Open FITS file
    hdul = fits.open(input_fits)

    # Print the structure
    hdul.info()

    # Access image data (usually in the first HDU)
    image_data = hdul[0].data

    # Show the image
    plt.imshow(image_data, cmap="gray")
    # plt.imshow(image_data)
    plt.colorbar()

    inpath = Path(input_fits)
    out_new = inpath.with_suffix(".png")

    plt.savefig(out_new, dpi=300, bbox_inches="tight")
    plt.show()

    # Print header info (metadata)
    print(repr(hdul[0].header))


if __name__ == "__main__":
    main()
