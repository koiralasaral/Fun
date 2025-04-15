import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Step 1: Define the directory containing .nc files
nc_files_directory = "C:\\Users\\LENOVO\\Downloads\\GEBCO_04_Apr_2025_6ec5b7288313"  # Replace with the path to your directory

# Define multiple latitude and longitude ranges
coordinate_ranges = [
    {"lat_min": 65.641, "lat_max": 65.66, "lon_min": 23.123, "lon_max": 23.143},  # Example: Taiwan
    {"lat_min": 65.706, "lat_max": 65.712, "lon_min": 23.12, "lon_max": 23.127},  # Example: Japan
    {"lat_min": 65.46, "lat_max": 65.544, "lon_min": 23.111, "lon_max": 23.17},  # Example: Equatorial region
]

# Step 2: Loop through each .nc file in the directory
for nc_file in os.listdir(nc_files_directory):
    if nc_file.endswith(".nc"):  # Process only .nc files
        file_path = os.path.join(nc_files_directory, nc_file)
        print(f"Processing file: {file_path}")

        # Load the NetCDF file
        data = Dataset(file_path, mode='r')

        # Extract variables (ensure consistency across files)
        try:
            lats = data.variables['lat'][:]
            lons = data.variables['lon'][:]
            depth = data.variables['elevation'][:]  # Replace with correct variable name (e.g., 'tid' or 'bathymetry')

            # Step 3: Loop through each coordinate range
            for coords in coordinate_ranges:
                lat_min, lat_max = coords["lat_min"], coords["lat_max"]
                lon_min, lon_max = coords["lon_min"], coords["lon_max"]

                # Mask the data for the specific region
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                lon_mask = (lons >= lon_min) & (lons <= lon_max)

                if np.sum(lat_mask) > 0 and np.sum(lon_mask) > 0:
                    # Subset the data
                    lat_indices = np.where(lat_mask)[0]
                    lon_indices = np.where(lon_mask)[0]
                    lon_grid, lat_grid = np.meshgrid(lons[lon_indices], lats[lat_indices])
                    depth_region = depth[np.ix_(lat_indices, lon_indices)]

                    # Visualize the data
                    plt.figure(figsize=(10, 8))
                    plt.contourf(lon_grid, lat_grid, depth_region, cmap='terrain')
                    plt.colorbar(label="Depth (m)")
                    plt.title(f"{nc_file} - Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.grid()
                    plt.show()
                else:
                    print(f"No data found for region: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")
        except KeyError as e:
            print(f"Variable not found in file {nc_file}: {e}")
        finally:
            data.close()