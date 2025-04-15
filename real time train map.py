import requests
import folium
import pandas as pd

# TrainFlow API endpoint (replace with actual working API URL)
import os
import requests

# Retrieve credentials from environment variables
username = os.getenv("koiralasaral")
password = os.getenv("L2s4Yslfbz")

# TrainFlow API endpoint
api_url = "https://api.rtt.io/api/v1/json/search/LBG"

# Set headers securely
headers = {
    "Username": username,
    "Password": password
}

# Fetch real-time train data securely
response = requests.get(api_url, headers=headers, timeout=10)

# Fetch real-time train data
try:
    response = requests.get(api_url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
    train_data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching train data: {e}")
    train_data = {"trains": []}  # Fallback to an empty train list

# Ensure there is train location data
if not train_data["trains"]:
    print("No real-time train data available. Try again later.")
    exit()

# Extract train locations safely
train_locations = [(train.get("latitude"), train.get("longitude")) for train in train_data["trains"] if train.get("latitude") and train.get("longitude")]

# Create Folium map centered at the first train location
m = folium.Map(location=train_locations[0], zoom_start=6)

# Plot train locations on the map
for loc in train_locations:
    folium.Marker(location=loc, popup="Train", icon=folium.Icon(color="blue")).add_to(m)

# Save and display the interactive map
m.save("real_time_train_map.html")
print("Map saved! Open 'real_time_train_map.html' to view.")