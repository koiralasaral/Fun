import folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from geopy.distance import geodesic

# Define all major train stations along the London-Paris route
stations = {
    "London St Pancras": (51.5317, -0.1263),
    "Ebbsfleet International": (51.4431, 0.3236),
    "Ashford International": (51.1432, 0.8744),
    "Calais-Fréthun": (50.9200, 1.8628),
    "Lille Europe": (50.6364, 3.0701),
    "Paris Gare du Nord": (48.8808, 2.3553)
}

# Extract station coordinates
train_track = list(stations.values())

# Interpolate movement for smoother transitions
animation_steps = 60  # 1-minute animation (updates every second)
latitudes = np.linspace(train_track[0][0], train_track[-1][0], animation_steps)
longitudes = np.linspace(train_track[0][1], train_track[-1][1], animation_steps)

# 🚆 **Matplotlib Animation**
fig, ax = plt.subplots(figsize=(6, 6))
x_values = [loc[1] for loc in train_track]  # Longitude values
y_values = [loc[0] for loc in train_track]  # Latitude values

def update(frame):
    ax.clear()
    ax.plot(x_values[:frame + 1], y_values[:frame + 1], color="red", marker="o", linestyle="-")
    ax.set_title("Animated Train Journey: London → Paris")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

ani = animation.FuncAnimation(fig, update, frames=animation_steps, interval=1000)  # Smooth updates

# Save animation
ani.save("train_journey_animation.gif", writer="pillow")

# 🎉 **Display Animation**
plt.show()

# 🌍 **Folium Map Visualization**
m = folium.Map(location=stations["London St Pancras"], zoom_start=5)

# Plot railway track in Folium
folium.PolyLine(train_track, color="blue", weight=5, opacity=0.8).add_to(m)

# Add station markers
for name, coords in stations.items():
    folium.Marker(location=coords, popup=name, icon=folium.Icon(color="blue")).add_to(m)

# Save Folium map
m.save("train_animation_map.html")
print("Map saved! Open 'train_animation_map.html' to view.")
print("Animation saved! Check 'train_journey_animation.gif'.")