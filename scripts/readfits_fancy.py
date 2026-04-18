
# from astropy.io import fits
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.visualization import ImageNormalize, ZScaleInterval, SqrtStretch

# # Open FITS
# filestring = 'narrow_image_stretch'
# hdul = fits.open(f'{filestring}.fits')
# hdul.info()

# # Read data (as float)
# image_data = hdul[0].data.astype(float)

# # Replace NaN / Inf
# image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

# # Normalize + stretch for visualization
# norm = ImageNormalize(image_data, interval=ZScaleInterval(), stretch=SqrtStretch())

# # Plot
# plt.figure(figsize=(8, 8))
# plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
# plt.colorbar(label='Pixel value')
# plt.axis('off')
# plt.tight_layout()

# # Save first
# plt.savefig(f'{filestring}.png', dpi=300, bbox_inches='tight')

# # Then show
# plt.show()
# plt.close()

# # Print header
# print(repr(hdul[0].header))
# hdul.close()

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (
    ImageNormalize,
    ZScaleInterval,
    AsinhStretch
)

# === SETTINGS ===
filestring = 'narrow_image'
fits_file = f"{filestring}.fits"

# === 1. Open FITS ===
hdul = fits.open(fits_file)
hdul.info()

# Read data as float and clean NaNs/Infs
image_data = np.nan_to_num(hdul[0].data.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

# === 2. Background removal ===
# Subtract median (background sky level)
background = np.median(image_data)
image_data -= background
image_data[image_data < 0] = 0  # no negatives after subtraction

# === 3. Clip outliers for display ===
# Clip between 0.5th and 99.5th percentile to remove extreme hot/cold pixels
vmin, vmax = np.percentile(image_data, (0.5, 99.5))
image_data = np.clip(image_data, vmin, vmax)

# === 4. Normalize and stretch (similar to ASTAP auto-stretch) ===
norm = ImageNormalize(
    image_data,
    interval=ZScaleInterval(),
    stretch=AsinhStretch(0.1)  # gamma-like stretch to boost faint details
)

# === 5. Plot ===
plt.figure(figsize=(8, 8))
plt.imshow(image_data, cmap='gray', origin='lower', norm=norm)
plt.colorbar(label='Scaled pixel value')
plt.axis('off')
plt.tight_layout()

# Save first, then show
plt.savefig(f"{filestring}_contrast.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# === 6. Print header ===
print(repr(hdul[0].header))
hdul.close()
