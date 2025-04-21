import numpy as np
import folium
from folium.features import DivIcon
from scipy.interpolate import splprep, splev

# ------------------------------
# A. Define the Route Waypoints
# ------------------------------
# Global coordinates: [lat, lon]
nepalgunj = np.array([28.05, 81.60])
chitwan   = np.array([27.53, 84.35])
kathmandu = np.array([27.7172, 85.3240])
everest   = np.array([27.9881, 86.9250])
waypoints = np.array([
    nepalgunj,
    chitwan,
    kathmandu,
    everest
])
lat = waypoints[:, 0]
lon = waypoints[:, 1]

# ------------------------------
# B. Spline Interpolation
# ------------------------------
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 200)
lon_smooth, lat_smooth = splev(u_new, tck)
dlon, dlat = splev(u_new, tck, der=1)
d2lon, d2lat = splev(u_new, tck, der=2)

# ------------------------------
# C. Local Coordinates (Origin = Nepalgunj)
# ------------------------------
O_lat, O_lon = nepalgunj[0], nepalgunj[1]
x_local = lon_smooth - O_lon
y_local = lat_smooth - O_lat
r_arr = np.sqrt(x_local**2 + y_local**2)

# ------------------------------
# D. Tangents and Pedal Distance
# ------------------------------
T_arr = np.vstack((dlon, dlat)).T
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T
p_arr = []
for i in range(len(r_arr)):
    P_local = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    p_arr.append(np.abs(np.dot(P_local, T_i)))
p_arr = np.array(p_arr)

# ------------------------------
# E. Compute Geometry at a Sample Point (index 100)
# ------------------------------
i_sample = 100
P_local = np.array([x_local[i_sample], y_local[i_sample]])
r_sample = r_arr[i_sample]
T_sample = T_unit[i_sample]
p_sample = p_arr[i_sample]
Q_local = np.dot(P_local, T_sample) * T_sample

# Finite difference for dr/dp:
if i_sample > 0 and i_sample < len(r_arr)-1:
    dr = r_arr[i_sample+1] - r_arr[i_sample-1]
    dp = p_arr[i_sample+1] - p_arr[i_sample-1]
    dr_dp = dr/dp if dp != 0 else np.nan
else:
    dr_dp = np.nan
rho_sample = r_sample * dr_dp
PR_sample = 2 * p_sample * dr_dp
psi = np.arccos(np.dot(P_local, T_sample)/(r_sample+1e-9))
psi_deg = np.degrees(psi)

# Curvature and osculating circle:
num = np.abs(dlon*d2lat - dlat*d2lon)
den = (dlon**2+dlat**2)**1.5 + 1e-6
k_arr = num/den
k_sample = k_arr[i_sample]
R0 = 1/k_sample if k_sample != 0 else np.inf
N_sample = np.array([-T_sample[1], T_sample[0]])
C_local = P_local + R0 * N_sample

# Horizontal chord calculation:
disc = R0**2 - (P_local[1] - C_local[1])**2
if disc>=0:
    chord_x1 = C_local[0] + np.sqrt(disc)
    chord_x2 = C_local[0] - np.sqrt(disc)
    chord_coords = [[chord_x1, P_local[1]], [chord_x2, P_local[1]]]
else:
    chord_coords = [[P_local[0], P_local[1]], [P_local[0], P_local[1]]]

# Convert local geometry back to global coordinates.
# Global coordinate = [O_lat + y, O_lon + x]
P_global = [O_lat + P_local[1], O_lon + P_local[0]]
Q_global = [O_lat + Q_local[1], O_lon + Q_local[0]]
C_global = [O_lat + C_local[1], O_lon + C_local[0]]
# Tangent segment endpoints:
seg = 0.02
tang_start_local = P_local - seg * T_sample
tang_end_local   = P_local + seg * T_sample
tang_start_global = [O_lat + tang_start_local[1], O_lon + tang_start_local[0]]
tang_end_global   = [O_lat + tang_end_local[1], O_lon + tang_end_local[0]]
# Radial line: from Nepalgunj to P_global.
OP_coords = [[O_lon, O_lat], [P_global[1], P_global[0]]]
# Pedal line: from Nepalgunj to Q_global.
OQ_coords = [[O_lon, O_lat], [Q_global[1], Q_global[0]]]
# Chord on osculating circle (convert local chord points to global).
chord_global = []
for pt in chord_coords:
    chord_global.append([O_lon+pt[0], O_lat+pt[1]])

# ------------------------------
# F. Create the Folium Map
# ------------------------------
# Create a Folium map centered between Nepalgunj and Everest.
center_lat = (nepalgunj[0] + everest[0]) / 2
center_lon = (nepalgunj[1] + everest[1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

# Add the route polyline.
route_coords = list(zip(lat_smooth, lon_smooth))  # (lat, lon) pairs
folium.PolyLine(locations=route_coords, color="blue", weight=3, popup="Optimized Route").add_to(m)

# Add static markers.
folium.Marker(location=nepalgunj.tolist(), popup="Nepalgunj (Origin)", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(location=everest.tolist(), popup="Mount Everest", icon=folium.Icon(color="red")).add_to(m)
folium.Marker(location=chitwan.tolist(), popup="Chitwan", icon=folium.Icon(color="darkpurple")).add_to(m)
folium.Marker(location=kathmandu.tolist(), popup="Kathmandu", icon=folium.Icon(color="cadetblue")).add_to(m)

# Draw the radial line (OP).
folium.PolyLine(locations=[[O_lat, O_lon], [P_global[0], P_global[1]]],
                color="black", weight=4, popup="OP = r").add_to(m)
# Draw the pedal line (OQ).
folium.PolyLine(locations=[[O_lat, O_lon], [Q_global[0], Q_global[1]]],
                color="purple", weight=3, dash_array="5,5", popup="OQ = p").add_to(m)
# Draw the tangent segment.
folium.PolyLine(locations=[[tang_start_global[0], tang_start_global[1]], 
                           [tang_end_global[0], tang_end_global[1]]],
                color="magenta", weight=3, dash_array="2,4", popup=f"Tangent at P<br>ψ = {psi:.2f} rad ({psi_deg:.1f}°)").add_to(m)
# Draw the osculating circle.
circle_points = []
for theta_val in np.linspace(0, 2*np.pi, 100):
    x_circ = C_local[0] + R0*np.cos(theta_val)
    y_circ = C_local[1] + R0*np.sin(theta_val)
    circle_points.append([O_lat + y_circ, O_lon + x_circ])
folium.PolyLine(locations=circle_points, color="green", weight=2, dash_array="4,4", popup="Osculating Circle").add_to(m)
# Draw the horizontal chord.
folium.PolyLine(locations=[[chord_global[0][0], chord_global[0][1]],
                           [chord_global[1][0], chord_global[1][1]]],
                color="orange", weight=3, popup="Chord (PM ∥ x-axis)").add_to(m)

# Add a marker at the sample point, with popup showing computed values and formulas.
popup_text = (
    "<b>Geometry at Sample Point (index 100)</b><br>"
    f"OP (r) = {r_sample:.4f}<br>"
    f"OQ (p) = {p_sample:.4f}<br>"
    f"dr/dp = {dr_dp:.4f}<br>"
    f"ρ = r·(dr/dp) = {rho_sample:.4f}<br>"
    f"PR = 2p·(dr/dp) = {PR_sample:.4f}<br>"
    f"ψ = {psi:.4f} rad ({psi_deg:.1f}°)<br><br>"
    "<b>Curvature Equations:</b><br>"
    "ρ = r·(dr/dp)<br>"
    "p + d²p/dφ² = ρ<br>"
    "PR = 2p·(dr/dp)<br>"
    "PN = 2ρ·cos(ψ)<br>"
    "2ρ·sin(ψ)"
)
folium.Marker(location=P_global, 
              popup=folium.Popup(popup_text, max_width=300),
              icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

# Add a static marker showing formulas near the center.
formula_popup = ("<b>Curvature Formulas:</b><br>"
                 "ρ = r·(dr/dp)<br>"
                 "p + d²p/dφ² = ρ<br>"
                 "PR = 2p·(dr/dp)<br>"
                 "PN = 2ρ·cos(ψ)<br>"
                 "2ρ·sin(ψ)")
folium.Marker(location=[center_lat, center_lon],
              popup=folium.Popup(formula_popup, max_width=300),
              icon=folium.Icon(color="orange", icon="question-sign")).add_to(m)

# Save the map.
m.save("map_with_geometry.html")
print("Folium map saved as 'map_with_geometry.html'. Open this file in your browser to view the route and geometry.")
