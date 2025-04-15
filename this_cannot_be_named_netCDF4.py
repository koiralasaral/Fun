import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Load bathymetric data
file_path = "C:\\Users\\LENOVO\\Downloads\\GEBCO_04_Apr_2025_c3f5515709af\\gebco_2024_n65.7521_s65.6917_w22.3768_e22.4556.nc"
data = Dataset(file_path, mode='r')

# Extract variables
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
depth = data.variables['elevation'][:]  # Ensure this matches your dataset

# Print dataset ranges
print(f"Latitudes in dataset: {lats.min()} to {lats.max()}")
print(f"Longitudes in dataset: {lons.min()} to {lons.max()}")

# Define region
lat_min, lat_max = 65.6917, 65.7521
lon_min, lon_max = 22.3768, 22.4556

# Debug the ranges
print(f"Query Latitude range: {lat_min} to {lat_max}")
print(f"Query Longitude range: {lon_min} to {lon_max}")

# Expand region if needed
lat_min -= 0.01
lat_max += 0.01
lon_min -= 0.01
lon_max += 0.01

# Create masks
lat_mask = (lats >= lat_min) & (lats <= lat_max)
lon_mask = (lons >= lon_min) & (lons <= lon_max)

print(f"Latitudes selected: {np.sum(lat_mask)}")
print(f"Longitudes selected: {np.sum(lon_mask)}")

if np.sum(lat_mask) > 1 and np.sum(lon_mask) > 1:
    # Subset data
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    lon_grid, lat_grid = np.meshgrid(lons[lon_indices], lats[lat_indices])
    depth_subset = depth[np.ix_(lat_indices, lon_indices)]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(lon_grid, lat_grid, depth_subset, cmap='terrain')
    plt.colorbar(label="Depth (m)")
    plt.title(f"Bathymetry Visualization\nLat: {lat_min}-{lat_max}, Lon: {lon_min}-{lon_max}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()
else:
    print("No valid data found for the specified region.")