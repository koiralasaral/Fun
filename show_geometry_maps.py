import numpy as np
import folium
from folium.features import DivIcon
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
# ==============================================================================
# PART 1. Define Waypoints and Interpolate a Smooth Route
# ==============================================================================

# Global coordinates (latitude, longitude) for key Nepal locations.
nepalgunj = np.array([28.05, 81.60])         # Origin
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

# Spline interpolation (cubic, s=0 for exact interpolation).
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

# Use Nepalgunj as the origin, O.
O_lat, O_lon = nepalgunj[0], nepalgunj[1]

# For a simple local system, define:
#   x_local = global_lon - O_lon,   y_local = global_lat - O_lat.
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
    # Q = (P·T) T.
    Q_i = np.dot(P_i, T_i) * T_i
    p_arr.append(np.abs(np.dot(P_i, T_i)))
p_arr = np.array(p_arr)

# ==============================================================================
# PART 3. Compute Additional Local Geometry at a Sample Point
# ==============================================================================

# Choose a sample index (for example, index 100).
i_sample = 100

# For demonstration, we use a dummy sample point.
# In practice, you could use: P_local = np.array([x_local[i_sample], y_local[i_sample]])
P_local = np.array([0.5, 0.3])  # local coordinate differences (x, y) in degrees.
r_sample = np.linalg.norm(P_local)  # OP = r.
# Assume unit tangent (T_sample) is computed as (dummy values):
T_sample = np.array([0.8, 0.6]) / np.linalg.norm(np.array([0.8, 0.6]))
# And pedal distance p_sample computed from P_local and T_sample:
p_sample = np.abs(np.dot(P_local, T_sample))

# Now compute ψ: the angle between OP and T at P.
psi = np.arccos(np.dot(P_local, T_sample) / (r_sample + 1e-9))
psi_deg = np.degrees(psi)

# Compute the foot Q of the perpendicular from the origin onto the tangent.
Q_local = np.dot(P_local, T_sample) * T_sample

# Approximate dr/dp using a central difference (if possible).
if i_sample > 0 and i_sample < len(r_arr) - 1:
    dr = r_arr[i_sample + 1] - r_arr[i_sample - 1]
    dp = p_arr[i_sample + 1] - p_arr[i_sample - 1]
    dr_dp = dr / dp if dp != 0 else np.nan
else:
    dr_dp = np.nan

rho_sample = r_sample * dr_dp       # ρ = r*(dr/dp)
PR_sample = 2 * p_sample * dr_dp      # PR = 2p*(dr/dp)

# Compute angles.
theta_val = np.arctan2(P_local[1], P_local[0])            # Polar angle (θ) of OP.
psi = np.arccos(np.dot(P_local, T_sample) / (r_sample + 1e-9))  # Angle (ψ) between OP and tangent.
psi_deg = np.degrees(psi)

# Compute curvature using the spline derivatives.
k_arr = np.abs(dlon * d2lat - dlat * d2lon) / ((dlon**2 + dlat**2)**1.5 + 1e-6)
k_sample = k_arr[i_sample]
R0 = 1 / k_sample if k_sample != 0 else np.inf

# Compute centre of the osculating circle: C_local = P_local + R0 * N, where N is the unit normal.
N_sample = np.array([-T_sample[1], T_sample[0]])
C_local = P_local + R0 * N_sample

# Compute φ: angle from the line joining C to P.
phi = np.arctan2(P_local[1] - C_local[1], P_local[0] - C_local[0])

# Compute chords on the osculating circle.
# Horizontal chord (PM): x-values vary; y is fixed at P_local[1].
disc_h = R0**2 - (P_local[1] - C_local[1])**2
if disc_h >= 0:
    x_h1 = C_local[0] + np.sqrt(disc_h)
    x_h2 = C_local[0] - np.sqrt(disc_h)
    horizontal_chord_local = np.array([[x_h1, P_local[1]], [x_h2, P_local[1]]])
else:
    horizontal_chord_local = np.array([[P_local[0], P_local[1]], [P_local[0], P_local[1]]])
# Vertical chord (PN): y-values vary; x is fixed at P_local[0].
disc_v = R0**2 - (P_local[0] - C_local[0])**2
if disc_v >= 0:
    y_v1 = C_local[1] + np.sqrt(disc_v)
    y_v2 = C_local[1] - np.sqrt(disc_v)
    vertical_chord_local = np.array([[P_local[0], y_v1], [P_local[0], y_v2]])
else:
    vertical_chord_local = np.array([[P_local[0], P_local[1]], [P_local[0], P_local[1]]])

# ==============================================================================
# PART 4. Convert Local Geometry Back to Global Coordinates
# ==============================================================================

def local_to_global(local_pt):
    # local_pt = [x, y] where x = (global lon - O_lon), and y = (global lat - O_lat).
    return [O_lat + local_pt[1], O_lon + local_pt[0]]  # global = [lat, lon]

P_global = local_to_global(P_local)
Q_global = local_to_global(Q_local)
C_global = local_to_global(C_local)
horizontal_chord_global = [local_to_global(pt) for pt in horizontal_chord_local]
vertical_chord_global = [local_to_global(pt) for pt in vertical_chord_local]

# Also, compute a short tangent segment at P.
seg = 0.02  # segment length in local units.
tangent_start_local = P_local - seg * T_sample
tangent_end_local   = P_local + seg * T_sample
tangent_start_global = local_to_global(tangent_start_local)
tangent_end_global   = local_to_global(tangent_end_local)

# Radial line [OP]: from Nepalgunj to P_global.
OP_coords = [[O_lat, O_lon], [P_global[0], P_global[1]]]
# Pedal line [OQ]: from Nepalgunj to Q_global.
OQ_coords = [[O_lat, O_lon], [Q_global[0], Q_global[1]]]

# ==============================================================================
# PART 5. Create Matplotlib Plot (Global Coordinates as Axes)
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 8))
# Plot the spline route using global coordinates.
ax.plot(lon_smooth, lat_smooth, 'b-', label="Spline Route")
ax.plot(nepalgunj[1], nepalgunj[0], 'ko', label="Nepalgunj (Origin)")
ax.plot(P_global[1], P_global[0], 'ro', label="Sample Point P")
# Plot radial line OP.
ax.plot([O_lon, P_global[1]], [O_lat, P_global[0]], 'k-', lw=2, label="OP (r)")
ax.text((O_lon+P_global[1])/2, (O_lat+P_global[0])/2, "  r", color="black")
# Plot pedal line OQ.
ax.plot([O_lon, Q_global[1]], [O_lat, Q_global[0]], color="purple", lw=2, linestyle="--", label="OQ (p)")
ax.text(Q_global[1], Q_global[0], " Q", color="purple")
# Plot tangent segment at P.
ax.plot([tangent_start_global[1], tangent_end_global[1]], [tangent_start_global[0], tangent_end_global[0]], 
        color="magenta", lw=2, linestyle="-.", label="Tangent at P")
ax.text(tangent_end_global[1], tangent_end_global[0], " T", color="magenta")
# Plot the osculating circle.
theta_circ = np.linspace(0, 2*np.pi, 100)
circ_x = []
circ_y = []
for t in theta_circ:
    # Compute a point in local coordinates on the circle.
    x_c = C_local[0] + R0 * np.cos(t)
    y_c = C_local[1] + R0 * np.sin(t)
    # Convert to global:
    pt = local_to_global([x_c, y_c])
    circ_x.append(pt[1])  # lon
    circ_y.append(pt[0])  # lat
ax.plot(circ_x, circ_y, 'g--', lw=2, label="Osculating Circle")
ax.plot(C_global[1], C_global[0], 'co', label="Circle Centre C")
ax.text(C_global[1], C_global[0], "  C", color="cyan")
# Plot horizontal chord.
hchord_x = [horizontal_chord_global[0][1], horizontal_chord_global[1][1]]
hchord_y = [horizontal_chord_global[0][0], horizontal_chord_global[1][0]]
ax.plot(hchord_x, hchord_y, 'orange', lw=2, label="Horizontal Chord (PM)")
ax.text(np.mean(hchord_x), horizontal_chord_global[0][0], " PM", color="orange")
# Plot vertical chord.
vchord_x = [vertical_chord_global[0][1], vertical_chord_global[1][1]]
vchord_y = [vertical_chord_global[0][0], vertical_chord_global[1][0]]
ax.plot(vchord_x, vchord_y, 'brown', lw=2, label="Vertical Chord (PN)")
ax.text(vertical_chord_global[0][1], np.mean(vchord_y), " PN", color="brown")
# Annotate angles: show ψ at P, and also label axes with lat,lon.
ax.text(P_global[1], P_global[0], f"\nψ={psi:.2f} rad\n({psi_deg:.1f}°)\nLat:{P_global[0]:.4f}, Lon:{P_global[1]:.4f}", 
        color="red", fontsize=10)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Matplotlib Plot: Spline Route and Geometric Elements")
ax.legend(loc="upper left")
ax.grid(True)

plt.show()

# ==============================================================================
# PART 6. Create the Folium Map with All Geometry and Popups
# ==============================================================================

# Create a Folium map centered between Nepalgunj and Everest.
center_lat_map = (nepalgunj[0] + everest[0]) / 2
center_lon_map = (nepalgunj[1] + everest[1]) / 2
m = folium.Map(location=[center_lat_map, center_lon_map], zoom_start=7)

# Add the spline route as a blue polyline.
route_coords = list(zip(lat_smooth, lon_smooth))   # (lat, lon) pairs.
folium.PolyLine(locations=route_coords, color="blue", weight=3,
                popup="Optimized Route (Spline Interpolation)").add_to(m)

# Add markers for key cities.
for city, popup in zip([nepalgunj, chitwan, kathmandu, everest],
                       ["Nepalgunj (Origin)", "Chitwan", "Kathmandu", "Mount Everest"]):
    folium.Marker(location=city.tolist(),
                  popup=popup + f"<br>Lat: {city[0]:.4f}, Lon: {city[1]:.4f}",
                  icon=folium.Icon(color="green" if popup=="Nepalgunj (Origin)" else "red" if popup=="Mount Everest" else "cadetblue")
                 ).add_to(m)

# Draw the radial line (OP = r).
folium.PolyLine(locations=[[OP_coords[0][0], OP_coords[0][1]], [OP_coords[1][0], OP_coords[1][1]]],
                color="black", weight=4,
                popup="Radial Line OP = r<br>θ = {:.2f}°".format(np.degrees(theta_val))).add_to(m)

# Draw the pedal line (OQ = p).
folium.PolyLine(locations=[[O_lat, O_lon], [Q_global[0], Q_global[1]]],
                color="purple", weight=3, dash_array="5,5",
                popup="Pedal Line OQ = p<br>Lat: {:.4f}, Lon: {:.4f}".format(Q_global[0], Q_global[1])).add_to(m)

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

# Draw the horizontal chord (PM).
folium.PolyLine(locations=[[horizontal_chord_global[0][0], horizontal_chord_global[0][1]],
                           [horizontal_chord_global[1][0], horizontal_chord_global[1][1]]],
                color="orange", weight=3,
                popup="Horizontal Chord (PM) ∥ x-axis").add_to(m)

# Draw the vertical chord (PN).
folium.PolyLine(locations=[[vertical_chord_global[0][0], vertical_chord_global[0][1]],
                           [vertical_chord_global[1][0], vertical_chord_global[1][1]]],
                color="brown", weight=3,
                popup="Vertical Chord (PN) ∥ y-axis").add_to(m)

# Add a marker at the sample point P with a detailed popup.
marker_popup = (
    "<b>Geometry at Sample Point (middle of curve)</b><br>"
    "OP (r) = {:.4f}<br>"
    "OQ (p) = {:.4f}<br>"
    "dr/dp = {:.4f}<br>"
    "ρ = r·(dr/dp) = {:.4f}<br>"
    "PR = 2p·(dr/dp) = {:.4f}<br>"
    "ψ = {:.4f} rad ({:.1f}°)<br>"
    "θ = {:.4f} rad ({:.1f}°)<br>"
    "φ = {:.4f} rad ({:.1f}°)<br>"
    "Osculating Circle R₀ = {:.4f}<br>"
    "Lat: {:.4f}, Lon: {:.4f}"
).format(r_sample, p_sample, dr_dp, rho_sample, PR_sample,
         psi, psi_deg, theta_val, np.degrees(theta_val), phi, np.degrees(phi),
         R0, P_global[0], P_global[1])
folium.Marker(location=P_global, popup=folium.Popup(marker_popup, max_width=300),
              icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

# Add a static marker at the map center displaying general curvature formulas.
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

# Save the Folium map.
folium.LatLngPopup().add_to(m)
m.save("map_with_full_geometry.html")
print("Folium map saved as 'map_with_full_geometry.html'. Open this file in your browser to view it.")