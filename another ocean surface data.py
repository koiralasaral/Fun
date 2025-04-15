import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os

# Step 1: Define file path and coordinate ranges
file_path = "C:/Users/LENOVO/Downloads/GEBCO_2024_tid_n54.9819_s54.942_w14.7783_e15.0614.nc"  # Update file path
data = Dataset(file_path, mode='r')

# Define multiple latitude and longitude ranges
coordinate_ranges = [
    {"lat_min": 20, "lat_max": 26, "lon_min": 119, "lon_max": 123},  # Example: Taiwan
    {"lat_min": 30, "lat_max": 40, "lon_min": 130, "lon_max": 140},  # Example: Japan
    {"lat_min": -10, "lat_max": 10, "lon_min": 100, "lon_max": 110},  # Example: Equator region
]

# Step 2: Extract dataset variables
lats = data.variables['lat'][:]  # Ensure variable name matches dataset
lons = data.variables['lon'][:]  # Ensure variable name matches dataset
depth = data.variables['elevation'][:]  # Replace with correct variable name for depth/elevation

# Step 3: Loop through each coordinate range
for coords in coordinate_ranges:
    lat_min, lat_max = coords["lat_min"], coords["lat_max"]
    lon_min, lon_max = coords["lon_min"], coords["lon_max"]

    # Mask data for the specific region
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)

    if np.sum(lat_mask) > 0 and np.sum(lon_mask) > 0:
        # Create a subset of the data
        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]
        lon_grid, lat_grid = np.meshgrid(lons[lon_indices], lats[lat_indices])
        depth_region = depth[np.ix_(lat_indices, lon_indices)]

        # Visualize the subset data
        plt.figure(figsize=(10, 8))
        plt.contourf(lon_grid, lat_grid, depth_region, cmap='terrain')
        plt.colorbar(label="Depth (m)")
        plt.title(f"Depth Data: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid()
        plt.show()
    else:
        print(f"No data found for region: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")