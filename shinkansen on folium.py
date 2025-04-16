import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Load the Shinkansen Data ---
stations_data = [
    ["Shin-Hakodate-Hokuto", 41.90464401, 140.6486969],
    ["Kikonai", 41.67826462, 140.43756104],
    ["Okutsugaru-Imabetsu", 41.14491653, 140.51577759],
    ["Shin-Aomori", 40.82733536, 140.6934967],
    ["Hachinohe", 40.51228333, 141.48840332],
    ["Sendai", 38.26795197, 140.86953735],
    ["Tokyo", 35.68083191, 139.76693726],
    ["Shin-Osaka", 34.73361206, 135.75778198],
    ["Hakata", 33.58980179, 130.42068481]
]

# Convert to DataFrame & Sort by Latitude (North to South)
df_stations = pd.DataFrame(stations_data, columns=["Station", "Latitude", "Longitude"])
df_stations.sort_values("Latitude", ascending=False, inplace=True)

# --- 2. Create a Folium Map ---
# Centered around Japan
m = folium.Map(location=[37.0, 138.0], zoom_start=5)

# Add markers for each station
for _, row in df_stations.iterrows():
    folium.Marker([row["Latitude"], row["Longitude"]], popup=row["Station"]).add_to(m)

# Save the map as an HTML file to open in a browser
m.save("shinkansen_map.html")

# --- 3. Set Up Animation ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(min(df_stations["Longitude"]) - 1, max(df_stations["Longitude"]) + 1)
ax.set_ylim(min(df_stations["Latitude"]) - 1, max(df_stations["Latitude"]) + 1)
ax.set_title("Shinkansen Journey (North to South)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

train_marker, = ax.plot([], [], marker="o", markersize=10, color="red")

# Background: Show station locations
ax.scatter(df_stations["Longitude"], df_stations["Latitude"], color="blue", marker="x", label="Stations")
ax.legend()

# Animation function
def animate(i):
    train_marker.set_data(df_stations.iloc[i]["Longitude"], df_stations.iloc[i]["Latitude"])
    ax.set_title(f"Shinkansen Passing: {df_stations.iloc[i]['Station']}")
    return train_marker,

# --- 4. Create Animation Over 2 Minutes (120 Seconds) ---
num_stations = len(df_stations)
ani = animation.FuncAnimation(fig, animate, frames=num_stations, interval=120000/num_stations, repeat=False)

plt.show()