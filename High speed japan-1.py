import pandas as pd
import folium
import geopandas as gpd
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import time

# Load railway track data
railway_data = gpd.read_file("C:\\Users\\LENOVO\\Downloads\\hotosm_jpn_railways_lines_geojson\\hotosm_jpn_railways_lines_geojson.geojson")  # Replace with the actual path to your GeoJSON file

# Load Shinkansen station data
stations_df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\Shinkansen_stations_inJapan_geo.csv")  # Replace with the actual path to your CSV file

# Create a dictionary of station coordinates
stations = {
    row['Station_Name']: (row['Latitude'], row['Longitude'])  # Replace 'Station Name' with the actual column name for station names in your CSV file
    for _, row in stations_df.iterrows()
}

# Define train speeds (in km/h)
train_speeds = {
    "Nozomi": 300,
    "Hikari": 285,
    "Kodama": 275
}

# Select two stations for analysis
start_station = "Tokyo"
end_station = "Tazawako"

# Calculate distance between stations
distance_km = geodesic(stations[start_station], stations[end_station]).km

# Calculate travel times for different train types
travel_times = {
    train: distance_km / speed
    for train, speed in train_speeds.items()
}

# Create Folium map centered at the start station
m = folium.Map(location=stations[start_station], zoom_start=6)
folium.LatLngPopup().add_to(m)

# Plot railway tracks
for _, row in railway_data.iterrows():
    if row.geometry and row.geometry.geom_type == 'LineString':
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(coords, color="blue", weight=2).add_to(m)

# Add markers for the selected stations
for station in [start_station, end_station]:
    folium.Marker(
        location=stations[station],
        popup=station,
        icon=folium.Icon(color='red' if station == start_station else 'green')
    ).add_to(m)

# Display travel information
print(f"Distance between {start_station} and {end_station}: {distance_km:.2f} km")
for train, time in travel_times.items():
    print(f"{train} train estimated travel time: {time:.2f} hours")

# Save the map to an HTML file
m.save("shinkansen_route_map.html")
print("Map saved as 'shinkansen_route_map.html'.")
# Filter railway tracks to only show those between Tokyo and Tazawako

start_point = Point(stations[start_station][1], stations[start_station][0])
end_point = Point(stations[end_station][1], stations[end_station][0])

filtered_tracks = []
for _, row in railway_data.iterrows():
    if row.geometry and row.geometry.geom_type == 'LineString':
        line = row.geometry
        if line.distance(start_point) < 0.1 or line.distance(end_point) < 0.1:
            filtered_tracks.append(line)

# Plot filtered railway tracks
for line in filtered_tracks:
    coords = [(lat, lon) for lon, lat in line.coords]
    folium.PolyLine(coords, color="blue", weight=2).add_to(m)
    # Add train travel animation
    train_icon = folium.Icon(color='orange', icon='train', prefix='fa')

    # Generate intermediate points along the route
    route_line = LineString([start_point, end_point])
    num_points = 100
    intermediate_points = [
        route_line.interpolate(i / num_points, normalized=True)
        for i in range(num_points + 1)
    ]

    # Add train marker to the map
    train_marker = folium.Marker(
        location=(intermediate_points[0].y, intermediate_points[0].x),
        icon=train_icon
    )
    train_marker.add_to(m)

    # Simulate train movement
    for point in intermediate_points:
        train_marker.location = (point.y, point.x)
        m.save("shinkansen_route_map.html")
        time.sleep(0.1)  # Adjust speed of animation