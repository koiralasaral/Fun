import pandas as pd
import folium
import os
from folium.plugins import TimestampedGeoJson
from geopy.distance import geodesic

# Read the CSV file (Replace with your correct path)
csv_path = "C:\\Users\\LENOVO\\Downloads\\Shinkansen_stations_inJapan_geo.csv"
df_stations = pd.read_csv(csv_path)

# Sort stations north to south by Latitude
df_stations = df_stations.sort_values("Latitude", ascending=False)

# Color-coding regions
region_colors = {
    "Hokkaido": "purple",
    "Tohoku": "blue",
    "Kanto": "green",
    "Chubu": "orange",
    "Kansai": "red",
    "Chugoku": "pink",
    "Shikoku": "brown",
    "Kyushu": "black"
}

# Function to assign colors based on Prefecture
def get_station_color(prefecture):
    for region, color in region_colors.items():
        if region in prefecture:
            return color
    return "gray"  # Default color if not found in predefined regions

df_stations["Color"] = df_stations["Prefecture"].apply(get_station_color)

# Create a Folium map centered on Japan
m = folium.Map(location=[37.0, 138.0], zoom_start=5)

# **🔹 Add Station Markers with Color Coding**
for _, row in df_stations.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=row["Color"],
        fill=True,
        fill_color=row["Color"],
        fill_opacity=0.7,
        popup=folium.Popup(f"<b>{row['Station_Name']}</b><br>Region: {row['Prefecture']}", max_width=300),
    ).add_to(m)

# **🔹 Add a Polyline to Trace the Full Journey**
route_coords = [(row["Latitude"], row["Longitude"]) for _, row in df_stations.iterrows()]
folium.PolyLine(route_coords, color="blue", weight=3, opacity=0.8, tooltip="Shinkansen Route").add_to(m)

# **🔹 Calculate Travel Time Between Stations**
df_stations["Travel_Time"] = 0.0
for i in range(1, len(df_stations)):
    coord1 = (df_stations.iloc[i - 1]["Latitude"], df_stations.iloc[i - 1]["Longitude"])
    coord2 = (df_stations.iloc[i]["Latitude"], df_stations.iloc[i]["Longitude"])
    distance_km = geodesic(coord1, coord2).kilometers
    df_stations.at[i, "Travel_Time"] = distance_km / (300 / 3600)  # Based on 300 km/h speed

# Update timestamps for each station
current_time = pd.Timestamp("2023-01-01 08:00:00")
timestamps = [current_time]
for travel_time in df_stations["Travel_Time"][1:]:
    current_time += pd.to_timedelta(travel_time, unit="s")
    timestamps.append(current_time)
df_stations["Timestamp"] = timestamps

# **🔹 Prepare GeoJSON Features for Animated Movement**
features = []
for _, row in df_stations.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [row["Longitude"], row["Latitude"]]},
        "properties": {
            "time": row["Timestamp"].isoformat(),
            "popup": f"<b>{row['Station_Name']}</b><br>Travel Time: {round(row['Travel_Time'] / 60, 1)} min",
            "tooltip": f"Passing {row['Station_Name']}"
        }
    })

# **🔹 Add Animated Movement of Train Along the Route**
TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="PT10S",  # Adjust movement timing
    add_last_point=True,
).add_to(m)

# **🔹 Save the Fully Enhanced Animated Map**
m.save("shinkansen_animation_final.html")
print("Final animation saved! Open 'shinkansen_animation_final.html' in a browser to view.")