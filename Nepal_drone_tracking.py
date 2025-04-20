import folium
from folium.plugins import TimestampedGeoJson
import numpy as np
import datetime

# --------------------------------------------------------------
# Simulation Parameters
# --------------------------------------------------------------

# Base location: Kathmandu, Nepal (approximate center)
base_lat = 27.7172   # Latitude
base_lon = 85.3240   # Longitude

num_frames = 100  # Total simulation frames
t = np.linspace(0, 20, num_frames)

# We'll simulate positions as small offsets (in degrees) from the base.
# Assume 0.01° ~ 1.1 km so that a 0.05° amplitude gives a range of roughly a few kilometers.
offset_scale = 0.05

# Define target path (relative offsets): the target oscillates in both longitude (x offset)
# and latitude (y offset).
target_offsets = np.column_stack((
    offset_scale * np.sin(0.2 * t),  # x offset
    offset_scale * np.cos(0.2 * t)   # y offset
))

# The drone starts at zero offset relative to the base location.
drone_offset = np.array([0.0, 0.0])
alpha = 0.1  # Proportional gain for chasing the target

drone_offsets = []
target_offsets_list = []

for i in range(num_frames):
    target_off = target_offsets[i]
    # Drone moves gradually toward the target offset.
    drone_offset = drone_offset + alpha * (target_off - drone_offset)
    drone_offsets.append(drone_offset.copy())
    target_offsets_list.append(target_off.copy())

drone_offsets = np.array(drone_offsets)
target_offsets_list = np.array(target_offsets_list)

# Compute global positions by adding the base coordinates.
# GeoJSON expects coordinates as [longitude, latitude].
drone_global = np.column_stack((base_lon + drone_offsets[:, 0],
                                base_lat + drone_offsets[:, 1]))
target_global = np.column_stack((base_lon + target_offsets_list[:, 0],
                                 base_lat + target_offsets_list[:, 1]))

# --------------------------------------------------------------
# Print Input and Output Information
# --------------------------------------------------------------
print("Base Coordinates (Nepal):", (base_lat, base_lon))
print("Number of Frames:", num_frames)
print("\nInitial Target Global Position:", target_global[0])
print("Final Target Global Position:  ", target_global[-1])
print("\nInitial Drone Global Position:", drone_global[0])
print("Final Drone Global Position:  ", drone_global[-1])

# --------------------------------------------------------------
# Generate Timestamps for Each Simulation Frame
# --------------------------------------------------------------
base_time = datetime.datetime.now()
timestamps = [ (base_time + datetime.timedelta(seconds=i)).isoformat() for i in range(num_frames) ]

# --------------------------------------------------------------
# Build GeoJSON Features for Drone and Target
# --------------------------------------------------------------
drone_features = []
target_features = []
for i in range(num_frames):
    drone_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [ float(drone_global[i, 0]), float(drone_global[i, 1]) ]
        },
        "properties": {
            "times": [timestamps[i]],
            "popup": f"Drone: {drone_global[i]}"
        }
    })
    target_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [ float(target_global[i, 0]), float(target_global[i, 1]) ]
        },
        "properties": {
            "times": [timestamps[i]],
            "popup": f"Target: {target_global[i]}"
        }
    })

# Create separate FeatureCollections for the drone and target.
drone_geojson = {"type": "FeatureCollection", "features": drone_features}
target_geojson = {"type": "FeatureCollection", "features": target_features}

# --------------------------------------------------------------
# Create a Folium Map Centered in Nepal with Attractive Styling
# --------------------------------------------------------------
# Use the "CartoDB Positron" tile for a clean look.
m = folium.Map(location=[base_lat, base_lon], zoom_start=14, tiles="CartoDB positron")

# Create a TimestampedGeoJson layer for the drone.
drone_layer = TimestampedGeoJson(
    data=drone_geojson,
    period='PT1S',            # Each time step is 1 second.
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY-MM-DD HH:mm:ss',
    time_slider_drag_update=True
)
drone_layer.add_to(m)

# Create a separate TimestampedGeoJson layer for the target.
target_layer = TimestampedGeoJson(
    data=target_geojson,
    period='PT1S',
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY-MM-DD HH:mm:ss',
    time_slider_drag_update=True
)
target_layer.add_to(m)

# Add attractive polylines for the drone and target paths.
# Note: folium expects coordinate pairs as (latitude, longitude).
drone_polyline = folium.PolyLine(
    locations=[ (pt[1], pt[0]) for pt in drone_global ],
    color="blue", weight=3, opacity=0.8, tooltip="Drone Path"
)
target_polyline = folium.PolyLine(
    locations=[ (pt[1], pt[0]) for pt in target_global ],
    color="red", weight=3, opacity=0.8, tooltip="Target Path"
)
drone_polyline.add_to(m)
target_polyline.add_to(m)

# --------------------------------------------------------------
# Save the Animated Map
# --------------------------------------------------------------
m.save("drone_target_tracking_nepal_attractive.html")
print("\nMap has been saved to drone_target_tracking_nepal_attractive.html")
