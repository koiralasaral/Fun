import numpy as np
import folium
from folium.features import DivIcon
from scipy.interpolate import splprep, splev

# ==============================================================================
# PART 1. Define Waypoints and Interpolate a Smooth Route
# ==============================================================================

# Global coordinates (latitude, longitude) for key Nepal locations.
nepalgunj = np.array([28.05, 81.60])         # Origin: Nepalgunj
chitwan   = np.array([27.53, 84.35])
kathmandu = np.array([27.7172, 85.3240])
everest   = np.array([27.9881, 86.9250])

# Define a set of waypoints from Nepalgunj to Everest.
waypoints = np.array([
    nepalgunj,
    chitwan,
    kathmandu,
    everest
])
lat = waypoints[:, 0]
lon = waypoints[:, 1]

# Spline interpolation (cubic, s=0 for an exact interpolation).
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 200)
lon_smooth, lat_smooth = splev(u_new, tck)
# First derivatives (for tangent estimation).
dlon, dlat = splev(u_new, tck, der=1)
# Second derivatives (for curvature estimation).
d2lon, d2lat = splev(u_new, tck, der=2)

# ==============================================================================
# PART 2. Establish a Local Coordinate System (Origin = Nepalgunj)
# ==============================================================================

# Use Nepalgunj as the origin (O).
O_lat, O_lon = nepalgunj[0], nepalgunj[1]
# For a simple local coordinate system, define:
#   x_local = global longitude difference, y_local = global latitude difference.
x_local = lon_smooth - O_lon
y_local = lat_smooth - O_lat
r_arr = np.sqrt(x_local**2 + y_local**2)   # Radial distance (OP = r) at each point.

# Compute unit tangent vectors from the first derivatives.
T_arr = np.vstack((dlon, dlat)).T
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T

# Compute pedal distance p = |OQ|.
p_arr = []
for i in range(len(r_arr)):
    P_i = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    # Q = (P · T) T.
    Q_i = np.dot(P_i, T_i) * T_i
    p_arr.append(np.abs(np.dot(P_i, T_i)))
p_arr = np.array(p_arr)

# ==============================================================================
# PART 3. Compute Additional Local Geometry at the Middle of the Curve
# ==============================================================================

# Automatically select the middle sample point.
i_sample = len(u_new) // 2  # Middle index

# Use the actual computed local coordinates from the spline.
P_local = np.array([x_local[i_sample], y_local[i_sample]])  # Local coordinates for P.
r_sample = np.linalg.norm(P_local)                           # OP = r.
T_sample = T_unit[i_sample]                                   # Unit tangent at P.
p_sample = p_arr[i_sample]                                    # Pedal distance, OQ = p.

# Compute the foot Q (the projection of P on the tangent).
Q_local = np.dot(P_local, T_sample) * T_sample

# Approximate dr/dp using central difference (if possible).
if i_sample > 0 and i_sample < len(r_arr) - 1:
    dr = r_arr[i_sample + 1] - r_arr[i_sample - 1]
    dp = p_arr[i_sample + 1] - p_arr[i_sample - 1]
    dr_dp = dr / dp if dp != 0 else np.nan
else:
    dr_dp = np.nan

rho_sample = r_sample * dr_dp       # ρ = r * (dr/dp)
PR_sample = 2 * p_sample * dr_dp      # PR = 2p * (dr/dp)

# Compute angles.
theta_val = np.arctan2(P_local[1], P_local[0])  # Polar angle of OP.
psi = np.arccos(np.dot(P_local, T_sample) / (r_sample + 1e-9))  # Angle between OP and tangent.
psi_deg = np.degrees(psi)

# Compute curvature-based osculating circle:
num = np.abs(dlon * d2lat - dlat * d2lon)
den = (dlon**2 + dlat**2)**1.5 + 1e-6
k_arr = num / den
k_sample = k_arr[i_sample]
R0 = 1 / k_sample if k_sample != 0 else np.inf

# Centre of the osculating circle, using the unit normal (rotate T_sample 90° counterclockwise).
N_sample = np.array([-T_sample[1], T_sample[0]])
C_local = P_local + R0 * N_sample

# Compute φ, the angle from C to P.
phi = np.arctan2(P_local[1] - C_local[1], P_local[0] - C_local[0])

# ==============================================================================
# Print Computed Values to the Console
# ==============================================================================
print("Computed Geometry at the Middle Sample Point (index {}):".format(i_sample))
print(f"P_local: {P_local}")
print(f"OP (r): {r_sample:.4f} (local units)")
print(f"Pedal distance (p): {p_sample:.4f} (local units)")
print(f"dr/dp: {dr_dp:.4f}")
print(f"ρ = r*(dr/dp): {rho_sample:.4f}")
print(f"PR = 2p*(dr/dp): {PR_sample:.4f}")
print(f"ψ: {psi:.4f} rad ({psi_deg:.1f}°)")
print(f"θ = {theta_val:.4f} rad ({np.degrees(theta_val):.1f}°)")
print(f"φ = {phi:.4f} rad ({np.degrees(phi):.1f}°)")
print(f"Osculating Circle Radius, R₀: {R0:.4f}")

# ==============================================================================
# PART 4. Convert Local Geometry Back to Global Coordinates
# ==============================================================================

def local_to_global(local_pt):
    # local_pt = [x, y] where x = (global_lon - O_lon), y = (global_lat - O_lat)
    return [O_lat + local_pt[1], O_lon + local_pt[0]]  # global = [lat, lon]

P_global = local_to_global(P_local)
Q_global = local_to_global(Q_local)
C_global = local_to_global(C_local)

# Compute chords on the osculating circle:
# Horizontal chord: fixed y = P_local[1]
disc_h = R0**2 - (P_local[1] - C_local[1])**2
if disc_h >= 0:
    x_h1 = C_local[0] + np.sqrt(disc_h)
    x_h2 = C_local[0] - np.sqrt(disc_h)
    horizontal_chord_local = np.array([[x_h1, P_local[1]], [x_h2, P_local[1]]])
else:
    horizontal_chord_local = np.array([[P_local[0], P_local[1]], [P_local[0], P_local[1]]])
horizontal_chord_global = [local_to_global(pt) for pt in horizontal_chord_local]

# Vertical chord: fixed x = P_local[0]
disc_v = R0**2 - (P_local[0] - C_local[0])**2
if disc_v >= 0:
    y_v1 = C_local[1] + np.sqrt(disc_v)
    y_v2 = C_local[1] - np.sqrt(disc_v)
    vertical_chord_local = np.array([[P_local[0], y_v1], [P_local[0], y_v2]])
else:
    vertical_chord_local = np.array([[P_local[0], P_local[1]], [P_local[0], P_local[1]]])
vertical_chord_global = [local_to_global(pt) for pt in vertical_chord_local]

# Tangent segment at P (a short line segment along T_sample).
seg = 0.02  # length in local units
tangent_start_local = P_local - seg * T_sample
tangent_end_local = P_local + seg * T_sample
tangent_start_global = local_to_global(tangent_start_local)
tangent_end_global = local_to_global(tangent_end_local)

# Radial line [OP]: from Nepalgunj (origin) to P_global.
OP_coords = [[O_lat, O_lon], [P_global[0], P_global[1]]]
# Pedal line [OQ]: from Nepalgunj (origin) to Q_global.
OQ_coords = [[O_lat, O_lon], [Q_global[0], Q_global[1]]]

# ==============================================================================
# PART 5. Create the Folium Map with All Geometric Lines and Multiple Popups
# ==============================================================================

# Center the map between Nepalgunj and Everest.
center_lat_map = (nepalgunj[0] + everest[0]) / 2
center_lon_map = (nepalgunj[1] + everest[1]) / 2
m = folium.Map(location=[center_lat_map, center_lon_map], zoom_start=7)

# Add the route as a blue polyline.
route_coords = list(zip(lat_smooth, lon_smooth))  # (lat, lon) order.
folium.PolyLine(locations=route_coords, color="blue", weight=3,
                popup="Optimized Route (Spline Interpolation)").add_to(m)

# Mark key cities.
for city, popup in zip([nepalgunj, chitwan, kathmandu, everest],
                       ["Nepalgunj (Origin)", "Chitwan", "Kathmandu", "Mount Everest"]):
    folium.Marker(location=city.tolist(),
                  popup=popup,
                  icon=folium.Icon(color="green" if popup=="Nepalgunj (Origin)" else \
                                   "red" if popup=="Mount Everest" else "cadetblue")
                 ).add_to(m)

# Draw the radial line (OP = r).
folium.PolyLine(locations=[[OP_coords[0][0], OP_coords[0][1]], [OP_coords[1][0], OP_coords[1][1]]],
                color="black", weight=4,
                popup="Radial Line OP = r<br>θ = {:.2f}°".format(np.degrees(theta_val))).add_to(m)

# Draw the pedal line (OQ = p).
folium.PolyLine(locations=[[O_lat, O_lon], [Q_global[0], Q_global[1]]],
                color="purple", weight=3, dash_array="5,5",
                popup="Pedal Line OQ = p").add_to(m)

# Draw the tangent segment at P.
folium.PolyLine(locations=[[tangent_start_global[0], tangent_start_global[1]],
                           [tangent_end_global[0], tangent_end_global[1]]],
                color="magenta", weight=3, dash_array="2,4",
                popup="Tangent at P<br>ψ = {:.2f} rad, {:.1f}°".format(psi, psi_deg)).add_to(m)

# Draw the osculating circle as a dashed green polyline.
circle_points = []
for theta_val_c in np.linspace(0, 2*np.pi, 100):
    x_c = C_local[0] + R0 * np.cos(theta_val_c)
    y_c = C_local[1] + R0 * np.sin(theta_val_c)
    pt_global = local_to_global([x_c, y_c])
    circle_points.append(pt_global)
folium.PolyLine(locations=circle_points, color="green", weight=2, dash_array="4,4",
                popup="Osculating Circle<br>R₀ = {:.2f}".format(R0)).add_to(m)

# Draw the horizontal chord (PM) on the osculating circle.
folium.PolyLine(locations=[[horizontal_chord_global[0][0], horizontal_chord_global[0][1]],
                           [horizontal_chord_global[1][0], horizontal_chord_global[1][1]]],
                color="orange", weight=3,
                popup="Horizontal Chord (PM) ∥ x-axis").add_to(m)

# Draw the vertical chord (PN) on the osculating circle.
folium.PolyLine(locations=[[vertical_chord_global[0][0], vertical_chord_global[0][1]],
                           [vertical_chord_global[1][0], vertical_chord_global[1][1]]],
                color="brown", weight=3,
                popup="Vertical Chord (PN) ∥ y-axis").add_to(m)

# Add a marker at the sample point P with a popup showing computed values and formulas.
marker_popup = (
    "<b>Geometry at Sample Point (index {})</b><br>"
    "OP = r = {:.4f}<br>"
    "OQ = p = {:.4f}<br>"
    "dr/dp = {:.4f}<br>"
    "ρ = r·(dr/dp) = {:.4f}<br>"
    "PR = 2p·(dr/dp) = {:.4f}<br>"
    "ψ = {:.4f} rad ({:.1f}°)<br><br>"
    "θ = {:.4f} rad ({:.1f}°)<br>"
    "φ = {:.4f} rad ({:.1f}°)<br><br>"
    "<b>Curvature Equations:</b><br>"
    "ρ = r·(dr/dp)<br>"
    "p + d²p/dφ² = ρ<br>"
    "PR = 2p·(dr/dp)<br>"
    "PN = 2ρ·cos(ψ)<br>"
    "2ρ·sin(ψ)"
).format(i_sample, r_sample, p_sample, dr_dp, rho_sample, PR_sample,
         psi, psi_deg, theta_val, np.degrees(theta_val), phi, np.degrees(phi))
folium.Marker(location=P_global, popup=folium.Popup(marker_popup, max_width=300),
              icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

# Add a static marker showing all formulas (placed at the map center).
formula_text = (
    "<b>Curvature Formulas:</b><br>"
    "ρ = r·(dr/dp)<br>"
    "p + d²p/dφ² = ρ<br>"
    "PR = 2p·(dr/dp)<br>"
    "PN = 2ρ·cos(ψ)<br>"
    "2ρ·sin(ψ)"
)
folium.Marker(location=[center_lat_map, center_lon_map],
              popup=folium.Popup(formula_text, max_width=300),
              icon=folium.Icon(color="orange", icon="question-sign")).add_to(m)

# ==============================================================================
# PART 6. Save the Map
# ==============================================================================

m.save("map_with_full_geometry.html")
print("Folium map saved as 'map_with_full_geometry.html'.")
