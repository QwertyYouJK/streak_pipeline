from sgp4.api import Satrec, SGP4_ERRORS
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import (
    TEME,
    GCRS,
    CartesianRepresentation,
    CartesianDifferential,
)

line1 = "1 48318U 21036AU  24247.18864902  .00007348  00000-0  51167-3 0  9998"
line2 = "2 48318  53.0535  61.8894 0001035  82.4630 277.6476 15.06399062185391"

# first measurement time = DATE-AVG from your FITS
t = Time("2024-09-03T08:42:00.9645801", scale="utc")

sat = Satrec.twoline2rv(line1, line2)

# SGP4 gives TEME position/velocity in km and km/s
err, r_km, v_kms = sat.sgp4(t.jd1, t.jd2)
if err != 0:
    raise RuntimeError(SGP4_ERRORS[err])

teme = TEME(
    CartesianRepresentation(r_km * u.km).with_differentials(
        CartesianDifferential(v_kms * u.km / u.s)
    ),
    obstime=t,
)

# Convert to a practical ECI-like frame
gcrs = teme.transform_to(GCRS(obstime=t))

r_m = gcrs.cartesian.xyz.to(u.m).value
v_mps = gcrs.velocity.d_xyz.to(u.m / u.s).value

print("TEME position [km]:", r_km)
print("TEME velocity [km/s]:", v_kms)

print("\nGCRS position [m]:")
print(f"x = {r_m[0]:.6f}")
print(f"y = {r_m[1]:.6f}")
print(f"z = {r_m[2]:.6f}")

print("\nGCRS velocity [m/s]:")
print(f"vx = {v_mps[0]:.6f}")
print(f"vy = {v_mps[1]:.6f}")
print(f"vz = {v_mps[2]:.6f}")

print("\nPaste this into YAML:")
print(
    f"initial_state: [{r_m[0]:.6f}, {r_m[1]:.6f}, {r_m[2]:.6f}, "
    f"{v_mps[0]:.6f}, {v_mps[1]:.6f}, {v_mps[2]:.6f}]"
)
