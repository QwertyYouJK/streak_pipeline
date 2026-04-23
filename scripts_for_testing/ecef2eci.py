from astropy import units as u
from astropy.time import Time
from astropy.coordinates import ITRS, GCRS
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.utils import iers

# Let Astropy fetch/use Earth orientation data if needed
iers.conf.auto_download = True

# ------------------------------------------------------------------
# REPLACE THESE WITH YOUR ACTUAL VALUES
# Epoch must match the exact time of the ECEF state
# ------------------------------------------------------------------
t = Time("2024-09-03T08:42:04.167", scale="utc")

# ECEF state from your external tool
x_km = 5045.499
y_km = 4105.192
z_km = 2197.316
vx_kms = -4.815196
vy_kms = 2.911199
vz_kms = 5.590007

# ------------------------------------------------------------------
# Build an ITRS/ECEF state with velocity
# ------------------------------------------------------------------
rep = CartesianRepresentation(
    x_km * u.km,
    y_km * u.km,
    z_km * u.km,
    differentials=CartesianDifferential(
        vx_kms * u.km / u.s, vy_kms * u.km / u.s, vz_kms * u.km / u.s
    ),
)

ecef = ITRS(rep, obstime=t)

# ------------------------------------------------------------------
# Convert to GCRS (practical ECI target)
# ------------------------------------------------------------------
eci = ecef.transform_to(GCRS(obstime=t))

# Extract position and velocity
r_m = eci.cartesian.xyz.to(u.m)
v_mps = eci.velocity.d_xyz.to(u.m / u.s)

print("GCRS/ECI position [m]:")
print(f"x = {r_m[0].value:.6f}")
print(f"y = {r_m[1].value:.6f}")
print(f"z = {r_m[2].value:.6f}")

print("\nGCRS/ECI velocity [m/s]:")
print(f"vx = {v_mps[0].value:.6f}")
print(f"vy = {v_mps[1].value:.6f}")
print(f"vz = {v_mps[2].value:.6f}")

print("\nYAML initial_state line:")
print(
    f"initial_state: [{r_m[0].value:.6f}, {r_m[1].value:.6f}, {r_m[2].value:.6f}, "
    f"{v_mps[0].value:.6f}, {v_mps[1].value:.6f}, {v_mps[2].value:.6f}]"
)
