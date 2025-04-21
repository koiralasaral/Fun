import numpy as np
import folium
from folium.features import DivIcon
from scipy.interpolate import splprep, splev

# =======================================================
# PART 1: Define Waypoints and Spline Interpolation
# =======================================================
# Define global coordinates (latitude, longitude) for:
katmandu = np.array([27.7172, 85.3240])
everest  = np.array([27.9881, 86.9250])
# Chitwan (approximate coordinates)
chitwan  = np.array([27.5291, 84.3540])

# For our route from Kathmandu to Everest we use several intermediate points
# (These are chosen to create a smooth curve even if not EXACTLY following any real road.)
waypoints = np.array([
    katmandu,
    [27.7300, 85.5000],
    [27.7500, 85.8000],
    [27.7700, 86.0000],
    [27.8000, 86.2000],
    [27.8300, 86.4500],
    [27.8600, 86.6500],
    [27.9000, 86.7500],
    everest
])

# Separate latitude and longitude from waypoints.
lat = waypoints[:, 0]
lon = waypoints[:, 1]

# Spline interpolation (cubic; s=0 means exact interpolation through points)
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 100)
lon_smooth, lat_smooth = splev(u_new, tck)
# First derivative (for tangent) and second derivative (for curvature estimation)
dlon, dlat = splev(u_new, tck, der=1)
d2lon, d2lat = splev(u_new, tck, der=2)

# =======================================================
# PART 2: Compute Local Geometry (with Kathmandu as Origin)
# =======================================================
# Define a “local” coordinate system by subtracting Kathmandu’s coordinates.
# Let O = (0,0) represent Kathmandu.
x_local = lon_smooth - katmandu[1]  # x = difference in longitude
y_local = lat_smooth - katmandu[0]    # y = difference in latitude

# Compute the distance from O to each point: r = OP.
r_arr = np.sqrt(x_local**2 + y_local**2)

# The unit tangent vector at each point (from global spline derivative, applied to local differences)
T_arr = np.vstack((dlon, dlat)).T
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T

# Compute the pedal distance, p = OQ, at each point:
# For each point P = (x, y) and its unit tangent T, the foot Q = (P·T)T.
p_arr = []
for i in range(len(r_arr)):
    P_i = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    Q_i = np.dot(P_i, T_i) * T_i
    p_arr.append(np.linalg.norm(Q_i))
p_arr = np.array(p_arr)

# =======================================================
# PART 3: Compute Additional Quantities at a Sample Point
# =======================================================
# Pick a sample index along the spline (e.g., index 50)
i_sample = 50
P_local = np.array([x_local[i_sample], y_local[i_sample]])
r_val   = r_arr[i_sample]
T_sample = T_unit[i_sample]

# Compute the foot Q for the sample point
Q = np.dot(P_local, T_sample) * T_sample
p_val = np.linalg.norm(Q)

# Estimate the derivative dr/dp at the sample using central differences.
if i_sample > 0 and i_sample < len(r_arr)-1:
    dr = r_arr[i_sample+1] - r_arr[i_sample-1]
    dp = p_arr[i_sample+1] - p_arr[i_sample-1]
    dr_dp = dr / dp if dp != 0 else np.nan
else:
    dr_dp = np.nan

# Define ρ (rho):  ρ = r * (dr/dp)
rho_val = r_val * dr_dp

# Define PR = 2 p (dr/dp)
PR_val = 2 * p_val * dr_dp

# Compute the angle ψ between OP and the tangent T.
# ψ = arccos( (P · T) / |P| )
psi_val = np.arccos(np.dot(P_local, T_sample) / r_val)

# Compute 2ρ sin(ψ) and 2ρ cos(ψ)
val_2rho_sin_psi = 2 * rho_val * np.sin(psi_val)
val_2rho_cos_psi = 2 * rho_val * np.cos(psi_val)

# For completeness, note the theoretical relations:
#   ρ = r (dr/dp)
#   p + d²p/dφ² = ρ   (as a theoretical relation)
#   PR = 2p (dr/dp)
#   PN = 2ρ cos(ψ)   (and 2ρ sin(ψ) may appear in other relations)

# =======================================================
# PART 4: Print Intermediate Values to Console
# =======================================================
print("=== Computed Geometric Quantities at Sample Point ===")
print(f"OP = r       : {r_val:.4f} (local units)")
print(f"OQ = p       : {p_val:.4f} (local units)")
print(f"dr/dp      : {dr_dp:.4f}")
print(f"ρ = r·dr/dp : {rho_val:.4f}")
print(f"PR = 2p·dr/dp  : {PR_val:.4f}")
print(f"2ρ·sin(ψ)  : {val_2rho_sin_psi:.4f}")
print(f"2ρ·cos(ψ)  : {val_2rho_cos_psi:.4f}")
print(f"ψ (radians): {psi_val:.4f}  ({np.degrees(psi_val):.2f}°)")
print("\nCurvature Formulae (as used in our analysis):")
print("   ρ = r (dr/dp)")
print("   p + d²p/dφ² = ρ  (theoretical relation)")
print("   PR = 2p (dr/dp)")
print("   PN = 2ρ cos(ψ)   (and similarly, 2ρ sin(ψ))")

# =======================================================
# PART 5: Create Folium Map with Route and Annotated Markers
# =======================================================
# Prepare a polyline for the smoothed route.
# Folium expects GeoJSON coordinates as [lat, lon]
route_coords = list(zip(lat_smooth, lon_smooth))

# Center map between Kathmandu and Everest.
center_lat = (katmandu[0] + everest[0]) / 2
center_lon = (katmandu[1] + everest[1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')

# Add the route as a blue polyline.
folium.PolyLine(
    locations=route_coords,
    color='blue',
    weight=3,
    opacity=0.7,
    popup="Optimized Route (Spline Interpolation)"
).add_to(m)

# Add markers for Kathmandu (origin) and Mount Everest.
folium.Marker(
    location=katmandu.tolist(),
    popup="Kathmandu (Origin)",
    icon=folium.Icon(color='green', icon='info-sign')
).add_to(m)

folium.Marker(
    location=everest.tolist(),
    popup="Mount Everest",
    icon=folium.Icon(color='red', icon='flag')
).add_to(m)

# Add a marker for Chitwan.
folium.Marker(
    location=chitwan.tolist(),
    popup="Chitwan",
    icon=folium.Icon(color='darkpurple', icon='info-sign')
).add_to(m)

# Convert the sample point (P_local, in local coordinates) back to global coordinates.
# Recall: x_local = lon - katmandu[1], y_local = lat - katmandu[0]
sample_lon = P_local[0] + katmandu[1]
sample_lat = P_local[1] + katmandu[0]
sample_global = [sample_lat, sample_lon]

# Prepare a popup with computed geometric equations and values.
popup_text = f"""
<b>Geometric Quantities at Sample Point</b><br>
Curve: Parametric spline interpolation<br>
OP = r = {r_val:.4f}<br>
OQ = p = {p_val:.4f}<br>
dr/dp = {dr_dp:.4f}<br>
<b>ρ = r·dr/dp = {rho_val:.4f}</b><br>
PR = 2p·dr/dp = {PR_val:.4f}<br>
2ρ·sin(ψ) = {val_2rho_sin_psi:.4f}<br>
2ρ·cos(ψ) = {val_2rho_cos_psi:.4f}<br>
ψ = {psi_val:.4f} rad ({np.degrees(psi_val):.2f}°)<br>
<br>
<b>Curvature Equations:</b><br>
ρ = r·(dr/dp)<br>
p + d²p/dφ² = ρ<br>
PR = 2p·(dr/dp)<br>
PN = 2ρ·cos(ψ)<br>
2ρ·sin(ψ)
"""

folium.Marker(
    location=sample_global,
    popup=folium.Popup(popup_text, max_width=300),
    icon=folium.Icon(color='blue', icon='info-sign')
).add_to(m)

# Also add a marker near the map center for the summary of formulas.
formula_text = """
<b>Curvature Formulae:</b><br>
ρ = r·(dr/dp)<br>
p + d²p/dφ² = ρ<br>
PR = 2p·(dr/dp)<br>
PN = 2ρ·cos(ψ)<br>
2ρ·sin(ψ)
"""
folium.Marker(
    location=[center_lat, center_lon],
    popup=folium.Popup(formula_text, max_width=300),
    icon=folium.Icon(color='orange', icon='question-sign')
).add_to(m)

# =======================================================
# PART 6: Save the Folium Map
# =======================================================
m.save("path_to_everest.html")
print("\nFolium map has been saved as 'path_to_everest.html'.\nOpen this file in your browser to view the annotated route with markers for Kathmandu, Chitwan, and Mount Everest.")
