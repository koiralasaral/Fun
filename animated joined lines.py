import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
from scipy.interpolate import splprep, splev

# =======================================================
# PART 1: Define Waypoints and Spline Interpolation
# =======================================================
# Global coordinates for key places.
katmandu = np.array([27.7172, 85.3240])        # Kathmandu (origin)
everest  = np.array([27.9881, 86.9250])         # Mount Everest
chitwan  = np.array([27.5291, 84.3540])         # Chitwan marker

# Define waypoints for a curved route from Kathmandu to Everest.
# (These intermediate points add curvature and need not be an exact road.)
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

lat = waypoints[:, 0]
lon = waypoints[:, 1]

# Use SciPy spline to interpolate a smooth route (cubic spline, s=0 for exact interpolation)
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 100)
lon_smooth, lat_smooth = splev(u_new, tck)
# Compute first derivative (for tangent estimates)
dlon, dlat = splev(u_new, tck, der=1)

# =======================================================
# PART 2: Set Up a Local Coordinate System (with Kathmandu as O)
# =======================================================
# For geometric computations we form local (x,y) coordinates by subtracting Kathmandu.
O_lon, O_lat = katmandu[1], katmandu[0]
x_local = lon_smooth - O_lon  # local x = difference in longitude
y_local = lat_smooth - O_lat  # local y = difference in latitude
r_arr = np.sqrt(x_local**2 + y_local**2)  # distance from O to each point

# Compute unit tangent vectors (from derivatives; using the same differences)
T_arr = np.vstack((dlon, dlat)).T  # Tangent vectors in global space (which matches local differences)
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T    # Normalize each tangent vector

# Compute pedal distance p (OQ). For each point P = (x,y), its foot Q on the tangent is given by Q = (P·T)T.
p_arr = []
for i in range(len(r_arr)):
    P_local = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    Q_i = np.dot(P_local, T_i) * T_i
    p_arr.append(np.linalg.norm(Q_i))
p_arr = np.array(p_arr)

# =======================================================
# PART 3: Build Animated Polyline (Joined Lines)
# =======================================================
# We create an animated feature for the cumulative route.
# For each sample from 0 to n, we produce a polyline joining the points from index 0 to i.
features = []
num_samples = len(u_new)

# Set the start time and specify an interval for animation frames.
start_time = datetime(2023, 1, 1, 0, 0, 0)
interval = timedelta(seconds=3)

for i in range(num_samples):
    # Build a polyline from the beginning (sample 0) to sample i
    poly_coords = []
    for j in range(i+1):
        # Convert global coordinates as [lon, lat] for geojson.
        poly_coords.append([lon_smooth[j], lat_smooth[j]])
    cur_time = (start_time + i * interval).isoformat()
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[pt[0], pt[1]] for pt in poly_coords]  # note: GeoJSON expects [lon, lat]
        },
        "properties": {
            "time": cur_time,
            "style": {
                "color": "blue",
                "weight": 4,
                "opacity": 0.8
            },
            "popup": f"Route up to sample {i}"
        }
    }
    features.append(feature)

animated_feature_collection = {
    "type": "FeatureCollection",
    "features": features
}

# =======================================================
# PART 4: Compute and Draw Static Geometric Lines at a Sample Point
# =======================================================
# Choose a sample index (e.g., 50) to compute additional geometry.
i_sample = 50
P_local = np.array([x_local[i_sample], y_local[i_sample]])
r_val = r_arr[i_sample]
T_sample = T_unit[i_sample]

# Compute Q: foot of perpendicular from O to tangent at P.
dot_val = np.dot(P_local, T_sample)
Q_local = dot_val * T_sample
p_val = np.linalg.norm(Q_local)

# Estimate dr/dp using finite differences (central difference).
if i_sample > 0 and i_sample < num_samples - 1:
    dr = r_arr[i_sample+1] - r_arr[i_sample-1]
    dp = p_arr[i_sample+1] - p_arr[i_sample-1]
    dr_dp = dr / dp if dp != 0 else np.nan
else:
    dr_dp = np.nan
rho_val = r_val * dr_dp             # ρ = r · (dr/dp)
PR_val = 2 * p_val * dr_dp          # PR = 2 p (dr/dp)
psi = np.arccos(dot_val / (r_val + 1e-9))  # angle between OP and tangent (radians)
psi_deg = np.degrees(psi)
val2rho_sin = 2 * rho_val * np.sin(psi)
val2rho_cos = 2 * rho_val * np.cos(psi)

# For static drawing, convert the sample point back to global coordinates.
P_global = [lat_smooth[i_sample], lon_smooth[i_sample]]
# Radial line (OP): from Kathmandu to P.
OP_coords = [[katmandu[1], katmandu[0]], [lon_smooth[i_sample], lat_smooth[i_sample]]]

# Pedal line (OQ): Q in global coordinates.
Q_global = [O_lat + Q_local[1], O_lon + Q_local[0]]  # recall: x_local = lon - O_lon, y_local = lat - O_lat

# Tangent segment at P (a short segment in both directions from P).
seg_len = 0.02  # in degrees (approx)
tangent_start_local = P_local - seg_len * T_sample
tangent_end_local   = P_local + seg_len * T_sample
tangent_start_global = [O_lat + tangent_start_local[1], O_lon + tangent_start_local[0]]
tangent_end_global   = [O_lat + tangent_end_local[1], O_lon + tangent_end_local[0]]

# Build static GeoJSON features (these will be added as separate layers on the map)
static_features = []

def make_line_feature(coords, popup_text, color, weight=3, dashArray=None):
    style = {"color": color, "weight": weight, "opacity": 0.8}
    if dashArray is not None:
        style["dashArray"] = dashArray
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords  # already in [lon, lat] order
        },
        "properties": {
            "popup": popup_text,
            "style": style
        }
    }

# Radial line OP
static_features.append(make_line_feature(OP_coords, "OP = r", "black", 4))
# Pedal line OQ
static_features.append(make_line_feature([[katmandu[1], katmandu[0]], [Q_global[1], Q_global[0]]],
                                           "OQ = p", "purple", 3, dashArray="5,5"))
# Tangent segment
static_features.append(make_line_feature([[tangent_start_global[1], tangent_start_global[0]],
                                            [tangent_end_global[1], tangent_end_global[0]]],
                                           f"Tangent at P<br>ψ = {psi:.3f} rad ({psi_deg:.1f}°)", "magenta", 3, dashArray="2,4"))

# Combine static features into a FeatureCollection (they will be added as a separate layer that always appears)
static_feature_collection = {
    "type": "FeatureCollection",
    "features": static_features
}

# =======================================================
# PART 5: Create the Folium Map and Add Layers
# =======================================================
# Center the map roughly between Kathmandu and Everest.
center_lat = (katmandu[0] + everest[0]) / 2
center_lon = (katmandu[1] + everest[1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')

# Add the animated joined-line feature.
TimestampedGeoJson(
    data=animated_feature_collection,
    period='PT3S',  # time interval 3 seconds
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY-MM-DDTHH:mm:ss',
    time_slider_drag_update=True
).add_to(m)

# Add the static geometric lines (using GeoJson layer so they always show).
folium.GeoJson(static_feature_collection,
               name="Geometric Lines",
               tooltip=folium.GeoJsonTooltip(fields=["popup"])
               ).add_to(m)

# Add static markers for Kathmandu, Mount Everest, and Chitwan.
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
folium.Marker(
    location=chitwan.tolist(),
    popup="Chitwan",
    icon=folium.Icon(color='darkpurple', icon='cloud')
).add_to(m)

# Add a static marker at the sample point with a popup showing computed formulas.
popup_text = f"""
<b>Geometry at Sample Point (index {i_sample})</b><br>
OP = r = {r_val:.4f}<br>
OQ = p = {p_val:.4f}<br>
dr/dp = {dr_dp:.4f}<br>
ρ = r·dr/dp = {rho_val:.4f}<br>
PR = 2p·dr/dp = {PR_val:.4f}<br>
2ρ·sin(ψ) = {val2rho_sin:.4f}<br>
2ρ·cos(ψ) = {val2rho_cos:.4f}<br>
ψ = {psi:.4f} rad ({psi_deg:.1f}°)<br>
<br>
Curvature Equations:<br>
ρ = r·(dr/dp)<br>
p + d²p/dφ² = ρ<br>
PR = 2p·(dr/dp)<br>
PN = 2ρ·cos(ψ)<br>
2ρ·sin(ψ)
"""
folium.Marker(
    location=P_global,
    popup=folium.Popup(popup_text, max_width=300),
    icon=folium.Icon(color='blue', icon='info-sign')
).add_to(m)

# Also add a marker showing the formulas (placed near the map center).
formula_text = ("<b>Curvature Formulas:</b><br>"
                "ρ = r·(dr/dp)<br>"
                "p + d²p/dφ² = ρ<br>"
                "PR = 2p·(dr/dp)<br>"
                "PN = 2ρ·cos(ψ)<br>"
                "2ρ·sin(ψ)")
folium.Marker(
    location=[center_lat, center_lon],
    popup=folium.Popup(formula_text, max_width=300),
    icon=folium.Icon(color='orange', icon='question-sign')
).add_to(m)

# Save the map to an HTML file.
m.save("animated_joined_lines.html")
print("Folium animated map saved as 'animated_joined_lines.html'.\nOpen this file in your browser to view the animation.")
