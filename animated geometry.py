import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
from scipy.interpolate import splprep, splev

# ================================
# PART 1: Define Route and Spline
# ================================
# Global coordinates (lat, lon) for key places.
katmandu = np.array([27.7172, 85.3240])
everest  = np.array([27.9881, 86.9250])
chitwan  = np.array([27.5291, 84.3540])

# Define waypoints along a curved route from Kathmandu to Everest.
# (These intermediate waypoints are chosen to yield an interesting spline; they are not necessarily a real road.)
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

# Separate latitudes and longitudes.
lat = waypoints[:, 0]
lon = waypoints[:, 1]

# Spline interpolation (cubic with s=0 to pass exactly through waypoints)
tck, u = splprep([lon, lat], s=0, k=3)
u_new = np.linspace(0, 1, 100)
lon_smooth, lat_smooth = splev(u_new, tck)
# First derivative (used to approximate the tangent at each point)
dlon, dlat = splev(u_new, tck, der=1)

# ================================
# PART 2: Establish a Local Coordinate System
# ================================
# We take Kathmandu as our origin, O.
O_lat, O_lon = katmandu[0], katmandu[1]
# Define local coordinates as differences (x = difference in longitude, y = difference in latitude)
x_local = lon_smooth - O_lon   # local x (in degrees)
y_local = lat_smooth - O_lat   # local y (in degrees)
# Distance from O to each point (in “local units” – degrees)
r_arr = np.sqrt(x_local**2 + y_local**2)

# Compute unit tangent vectors from derivatives.
T_arr = np.vstack((dlon, dlat)).T   # 2D tangent vector at each sample
T_norm = np.linalg.norm(T_arr, axis=1)
T_unit = (T_arr.T / T_norm).T       # normalized tangent

# For each sample compute the pedal distance p = |OQ|.
# For a point P (with local coordinates [x, y]) and unit tangent T,
# the foot Q is given by: Q = (P·T) T.
p_arr = []
for i in range(len(r_arr)):
    P_local = np.array([x_local[i], y_local[i]])
    T_i = T_unit[i]
    Q_i = np.dot(P_local, T_i) * T_i
    p_arr.append(np.linalg.norm(Q_i))
p_arr = np.array(p_arr)

# ================================
# PART 3: Build GeoJSON Features for Animation
# ================================
# We will create animated features for each sample point along the spline.
# At each time step we include:
#  - A moving point (P) on the route.
#  - The radial line from O to P (OP).
#  - The line from O to Q (OQ).
#  - A short segment representing the tangent at P.
#
# For the animation timestamps we'll use an interval (e.g., 5 seconds per step).
start_time = datetime(2023, 1, 1, 0, 0, 0)
interval = timedelta(seconds=5)

features = []  # list to hold all GeoJSON features

num_samples = len(u_new)
for i in range(num_samples):
    # Global coordinates for the sample point P are simply the spline coordinates:
    P_global = [lat_smooth[i], lon_smooth[i]]  # [lat, lon]
    # Local coordinates (difference from Kathmandu)
    x_i = x_local[i]  # = lon_smooth[i] - O_lon
    y_i = y_local[i]  # = lat_smooth[i] - O_lat
    # Unit tangent at sample i:
    T_i = T_unit[i]  # [t_x, t_y] in local coordinate units (long. diff, lat. diff)
    
    # Compute the foot Q of the perpendicular from O to the tangent.
    # P_local = [x_i, y_i]
    dot_val = x_i * T_i[0] + y_i * T_i[1]
    Q_local = dot_val * T_i  # projection vector in local coords
    # Convert Q_local to global: global Q = [O_lat + Q_local_y, O_lon + Q_local_x]
    Q_global = [O_lat + Q_local[1], O_lon + Q_local[0]]
    
    # Compute a short tangent segment at P.
    d_segment = 0.02   # segment length in local coordinate units (degrees)
    # Endpoints of the tangent segment in local coordinates:
    tangent1_local = np.array([x_i, y_i]) - d_segment * T_i
    tangent2_local = np.array([x_i, y_i]) + d_segment * T_i
    # Convert these endpoints to global coordinates:
    tangent1_global = [O_lat + tangent1_local[1], O_lon + tangent1_local[0]]
    tangent2_global = [O_lat + tangent2_local[1], O_lon + tangent2_local[0]]
    
    # Compute the angle ψ between OP and the tangent.
    # ψ = arccos((P_local · T_i)/|P_local|) where P_local = [x_i, y_i]
    r_val = np.sqrt(x_i**2 + y_i**2)
    dot_OP_T = (x_i * T_i[0] + y_i * T_i[1])
    psi = np.arccos(dot_OP_T / (r_val + 1e-9))  # add epsilon to avoid zero division
    psi_deg = np.degrees(psi)
    
    # Create a timestamp for this sample.
    cur_time = (start_time + i * interval).isoformat()
    
    # -------- Create a feature for point P --------
    feature_P = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [P_global[1], P_global[0]]  # GeoJSON uses [lon, lat]
        },
        "properties": {
            "time": cur_time,
            "popup": f"P<br>ψ = {psi:.3f} rad ({psi_deg:.1f}°)",
            "icon": "circle",
            "iconstyle": {
                "fillColor": "red",
                "fillOpacity": 0.8,
                "stroke": True,
                "radius": 4
            }
        }
    }
    features.append(feature_P)
    
    # -------- Feature for radial line OP (from Kathmandu (O) to P) --------
    feature_OP = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            # Coordinates: from O_global to P_global, in [lon, lat] order.
            "coordinates": [
                [O_lon, O_lat],
                [P_global[1], P_global[0]]
            ]
        },
        "properties": {
            "time": cur_time,
            "popup": "OP = r",
            "style": {
                "color": "black",
                "weight": 3,
                "opacity": 0.7
            }
        }
    }
    features.append(feature_OP)
    
    # -------- Feature for pedal line OQ (from O to Q) --------
    feature_OQ = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [O_lon, O_lat],
                [Q_global[1], Q_global[0]]
            ]
        },
        "properties": {
            "time": cur_time,
            "popup": "OQ = p",
            "style": {
                "color": "purple",
                "weight": 2,
                "opacity": 0.7,
                "dashArray": "5, 5"
            }
        }
    }
    features.append(feature_OQ)
    
    # -------- Feature for the tangent line at P --------
    feature_tangent = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [tangent1_global[1], tangent1_global[0]],
                [tangent2_global[1], tangent2_global[0]]
            ]
        },
        "properties": {
            "time": cur_time,
            "popup": f"Tangent at P<br>ψ = {psi:.3f} rad",
            "style": {
                "color": "magenta",
                "weight": 2,
                "opacity": 0.7,
                "dashArray": "2, 4"
            }
        }
    }
    features.append(feature_tangent)
    
    # Optionally, you could add a feature to show an arc or text annotation.
    # For simplicity, we embed the angle info in the popup.
    
# Combine all features into a FeatureCollection.
feature_collection = {
    "type": "FeatureCollection",
    "features": features
}

# ================================
# PART 4: Create the Folium Map
# ================================
# Center the map roughly between Kathmandu and Everest.
center_lat = (katmandu[0] + everest[0]) / 2
center_lon = (katmandu[1] + everest[1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')

# Add the animated features with TimestampedGeoJson.
TimestampedGeoJson(
    data=feature_collection,
    period='PT5S',  # Each timestamp is 5 seconds apart.
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY-MM-DDTHH:mm:ss',
    time_slider_drag_update=True
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

# Also add a static marker with the curvature formulas.
formulas_text = (
    "Curvature Formulas:<br>"
    "ρ = r·(dr/dp)<br>"
    "p + d²p/dφ² = ρ<br>"
    "PR = 2p·(dr/dp)<br>"
    "PN = 2ρ·cos(ψ)<br>"
    "2ρ·sin(ψ)"
)
folium.Marker(
    location=[center_lat, center_lon],
    popup=folium.Popup(formulas_text, max_width=300),
    icon=folium.Icon(color='orange', icon='question-sign')
).add_to(m)

# Save the animated map to an HTML file.
m.save("animated_geometry.html")
print("Folium animated map saved as 'animated_geometry.html'.\nOpen this file in your browser to view the animation.")

