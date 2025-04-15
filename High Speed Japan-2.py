import pandas as pd
import folium
import geopandas as gpd
from geopy.distance import geodesic

# Load railway track data
railway_data = gpd.read_file("N05-16_RailroadSection2.geojson")  # Replace with actual file path

# Load train schedule data
train_schedule_url = "https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/train-schedule.csv"  # Replace with actual dataset URL
train_data = pd.read_csv(train_schedule_url)

# Merge train speed data
train_data["speed_kmh"] = train_data["train_type"].map({"Bullet Train": 320, "Express": 250, "Regional": 180})

# Define station coordinates
stations = {
    "Tokyo": (35.681391, 139.766103),
    "Osaka": (34.702485, 135.495951)
}

# Calculate distance and travel time
distance_km = geodesic(stations["Tokyo"], stations["Osaka"]).km
train_speed = train_data["speed_kmh"].mean()
travel_time_hours = distance_km / train_speed

# Create Folium map
m = folium.Map(location=stations["Tokyo"], zoom_start=6)

# Plot railway track
for _, row in railway_data.iterrows():
    if row['geometry'].geom_type == 'LineString':
        coords = [(lat, lon) for lon, lat in row['geometry'].coords]
        folium.PolyLine(coords, color="red", weight=3).add_to(m)

# Add station markers
for name, coords in stations.items():
    folium.Marker(location=coords, popup=name).add_to(m)

# Display train schedule details
print(f"Distance: {distance_km:.2f} km")
print(f"Average train speed: {train_speed:.2f} km/h")
print(f"Estimated travel time: {travel_time_hours:.2f} hours")

# Save map
m.save("train_schedule_map.html")
print("Map saved! Open 'train_schedule_map.html' to view.")
