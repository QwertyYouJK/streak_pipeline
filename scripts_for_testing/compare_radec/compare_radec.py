from astropy.io import fits
from astropy.time import Time
from skyfield.api import EarthSatellite, load, wgs84

SATELLITE_NAME = "STARLINK-2509"
TLE_LINE1 = "1 48318U 21036AU  24249.70984417  .00005472  00000-0  38602-3 0  9997"
TLE_LINE2 = "2 48318  53.0537  50.5691 0001165  89.4796 270.6326 15.06394329185777"
FITS_PATH = "SL-2509(NORAD-48318)_00008.fits"

ts = load.timescale()
satellite = EarthSatellite(TLE_LINE1, TLE_LINE2, SATELLITE_NAME, ts)
tle_epoch = satellite.epoch

with fits.open(FITS_PATH, ignore_missing_end=True) as hdul:
    hdr = hdul[0].header

lat = float(hdr["GPS_LAT"])
lon = float(hdr["GPS_LONG"])
alt = float(hdr["GPS_ALT"])
t = Time(hdr["DATE-AVG"], format="isot", scale="utc")

observer = wgs84.latlon(lat, lon, elevation_m=alt)
ra, dec, _ = (satellite - observer).at(ts.from_astropy(t)).radec()

print(f"Observer location: lat={lat} deg, lon={lon} deg, alt={alt} m")
print(f"TLE epoch: {tle_epoch.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Observation time: {t.iso} UTC")
print(f"RA: {ra.degrees:.8f} deg")
print(f"Dec: {dec.degrees:.8f} deg")
