import folium
from folium.plugins import TimestampedGeoJson
import numpy as np
import datetime

# --------------------------------------------------------------
# Simulation Parameters
# --------------------------------------------------------------
num_frames = 100  # number of simulation frames
t = np.linspace(0, 20, num_frames)

# Define target path in 2D:
# For example: target's x and y coordinates oscillate sinusoidally.
target_positions = np.column_stack((
    5 * np.sin(0.2 * t),   # x position
    5 * np.cos(0.2 * t)    # y position
))

# Drone's initial position (2D)
drone_pos = np.array([0.0, 0.0])
alpha = 0.1  # proportional gain for the drone to "chase" the target

# Lists to store the positions over time
drone_path = []
target_path = []

# Simulate the drone tracking the target over time.
for i in range(num_frames):
    target = target_positions[i]
    # Simple proportional control:
    drone_pos = drone_pos + alpha * (target - drone_pos)
    drone_path.append(drone_pos.copy())
    target_path.append(target.copy())

# --------------------------------------------------------------
# Generate Timestamps for Each Simulation Frame
# --------------------------------------------------------------
base_time = datetime.datetime.now()
timestamps = [ (base_time + datetime.timedelta(seconds=i)).isoformat() for i in range(num_frames) ]

# --------------------------------------------------------------
# Build GeoJSON Features for Drone and Target
# --------------------------------------------------------------
# Note: In GeoJSON the coordinates are usually in [longitude, latitude]
# Here we treat our x as longitude and y as latitude.
drone_features = []
target_features = []

for i in range(num_frames):
    drone_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [ float(drone_path[i][0]), float(drone_path[i][1]) ]
        },
        "properties": {
            "times": [timestamps[i]],
            "popup": f"Drone: {drone_path[i]}"
        }
    })
    target_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [ float(target_path[i][0]), float(target_path[i][1]) ]
        },
        "properties": {
            "times": [timestamps[i]],
            "popup": f"Target: {target_path[i]}"
        }
    })

# Combine all features into one FeatureCollection.
features = drone_features + target_features
geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

# --------------------------------------------------------------
# Create a Folium Map and Add the TimestampedGeoJson Layer
# --------------------------------------------------------------
# Center the map at (0,0) and choose an appropriate zoom level.
m = folium.Map(location=[0, 0], zoom_start=6)

TimestampedGeoJson(
    data=geojson_data,
    period='PT1S',            # time step period (1 second)
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options='YYYY-MM-DD HH:mm:ss',
    time_slider_drag_update=True
).add_to(m)

# Optionally, draw full polylines for the drone and target paths.
# Note: folium expects coordinate pairs in (lat, lon) order.
drone_polyline = folium.PolyLine(
    locations=[ (pt[1], pt[0]) for pt in drone_path ],
    color="blue", weight=2, opacity=0.7, tooltip="Drone Path"
)
target_polyline = folium.PolyLine(
    locations=[ (pt[1], pt[0]) for pt in target_path ],
    color="red", weight=2, opacity=0.7, tooltip="Target Path"
)
drone_polyline.add_to(m)
target_polyline.add_to(m)

# Save map to an HTML file
m.save("drone_target_tracking.html")
print("Map has been saved to drone_target_tracking.html")
