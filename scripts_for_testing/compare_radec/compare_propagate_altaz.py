# Python script to compare RA/Dec from FITS metadata with propagated Alt/Az from user-provided TLEs.
# Usage: python compare_propagate_altaz.py

from sgp4.api import Satrec, SGP4_ERRORS
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    TEME,
    ITRS,
    AltAz,
    EarthLocation,
    CartesianRepresentation,
    CartesianDifferential,
    SkyCoord,
)

line1 = "1 48318U 21036AU  24247.18864902  .00007348  00000-0  51167-3 0  9998"
line2 = "2 48318  53.0535  61.8894 0001035  82.4630 277.6476 15.06399062185391"

t = Time("2024-09-03T08:42:04.1670085", format="isot", scale="utc")

site = EarthLocation.from_geodetic(
    lon=151.111238 * u.deg,
    lat=-33.770104333333336 * u.deg,
    height=68.19999694824219 * u.m,
)

sat = Satrec.twoline2rv(line1, line2)
err, r_km, v_kms = sat.sgp4(t.jd1, t.jd2)
if err != 0:
    raise RuntimeError(SGP4_ERRORS[err])

teme = TEME(
    CartesianRepresentation(r_km * u.km).with_differentials(
        CartesianDifferential(v_kms * u.km / u.s)
    ),
    obstime=t,
)

itrs_geo = teme.transform_to(ITRS(obstime=t))
topo_itrs_repr = itrs_geo.cartesian.without_differentials() - site.get_itrs(t).cartesian
itrs_topo = ITRS(topo_itrs_repr, obstime=t, location=site)
pred_altaz = itrs_topo.transform_to(AltAz(obstime=t, location=site))

print(f"Propagated Altitude (deg): {pred_altaz.alt.deg:.10f}")
print(f"Propagated Azimuth  (deg): {pred_altaz.az.deg:.10f}")


# your measured RA/Dec
ra_deg = 320.545346722032
dec_deg = -21.4661384264898

# measured RA/Dec in ICRS
coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

# convert to Alt/Az
altaz = coord.transform_to(AltAz(obstime=t, location=site))

print(f"My Altitude (deg): {altaz.alt.deg:.10f}")
print(f"My Azimuth  (deg): {altaz.az.deg:.10f}")
