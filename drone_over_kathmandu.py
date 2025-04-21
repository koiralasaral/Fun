import time
import json
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import TimestampedGeoJson
from scipy.interpolate import splprep, splev
from datetime import datetime, timedelta

# --- Define Waypoints (Real-life Example: Kathmandu, Nepal) ---
# Sample waypoints (latitude, longitude)
# (Note: These coordinates approximate a local flight path starting near central Kathmandu.)
waypoints = np.array([
    [27.7048, 85.3200],
    [27.7070, 85.3210],
    [27.7095, 85.3220],
    [27.7120, 85.3228],
    [27.7150, 85.3235],
    [27.7180, 85.3240],
    [27.7210, 85.3245],
    [27.7240, 85.3250]
])

print("=== Drone Path Optimization: Original Waypoints (lat, lon) ===")
print(waypoints)

# Generate timestamps for each original waypoint.
start_time = datetime.now()
times = [start_time + timedelta(seconds=10*i) for i in range(len(waypoints))]
time_strings = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times]

# --- Spline Smoothing for Trajectory Optimization ---
# (We use the spline technique from our maths code techniques, and also compute curvature.)
# For interpolation, we use longitude as x and latitude as y.
lon = waypoints[:, 1]
lat = waypoints[:, 0]

# spline parameterization (cubic spline smoothing, with smoothing factor s=0 for interpolation)
tck, u = splprep([lon, lat], s=0, k=3)
# Interpolate 100 points along the spline
u_new = np.linspace(0, 1, 100)
lon_smooth, lat_smooth = splev(u_new, tck)

# Compute first and second derivatives (for curvature calculation)
dx, dy = splev(u_new, tck, der=1)
d2x, d2y = splev(u_new, tck, der=2)
# Avoid division by zero by adding a small epsilon.
curvature = np.abs(dx * d2y - dy * d2x) / (np.power(dx*dx + dy*dy, 1.5) + 1e-6)

# Print some intermediate computed values.
print("\n=== Intermediate Spline Values ===")
print("First 5 interpolated (lat, lon) points with approximate curvature:")
for i in range(5):
    print(f"Point {i}: (lat = {lat_smooth[i]:.6f}, lon = {lon_smooth[i]:.6f}), curvature ~ {curvature[i]:.6f}")

# --- Plot the Optimized (Smoothed) Drone Trajectory ---
plt.figure(figsize=(8,6))
plt.plot(lon, lat, 'ro', label='Original Waypoints')
plt.plot(lon_smooth, lat_smooth, 'b-', label='Optimized Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Drone Optimized Trajectory over Kathmandu')
plt.legend()
plt.grid(True)
plt.show()

# --- Prepare Folium Timestamped Animation ---
# We now prepare a GeoJSON feature collection for the smooth path.
features = []
num_points = len(lon_smooth)
# Assume constant speed along the path.
total_time = (times[-1] - start_time).total_seconds()
dt = total_time / (num_points - 1)
for i in range(num_points):
    point_time = start_time + timedelta(seconds=i*dt)
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [lon_smooth[i], lat_smooth[i]]
        },
        'properties': {
            'time': point_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'popup': f"Point {i}",
            'icon': 'circle',
            'iconstyle': {
                'fillColor': 'blue',
                'fillOpacity': 0.6,
                'stroke': 'true',
                'radius': 4
            }
        }
    }
    features.append(feature)

data = {
    'type': 'FeatureCollection',
    'features': features
}

# Create a folium Map centered at the average location.
center_lat = np.mean(lat)
center_lon = np.mean(lon)
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

TimestampedGeoJson(
    data,
    period='PT10S',  # 10-second period between points
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY/MM/DD HH:mm:ss',
    time_slider_drag_update=True
).add_to(m)

# Save the map to an HTML file.
m.save("drone_path.html")
print("\nDrone path animation saved as 'drone_path.html'. Open this file in a web browser to view the animated flight path.")
